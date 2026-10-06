//! Gpu-to-gpu copies through Vulkan on the render gpu.
//!
//! [`MultiRenderer`](super::MultiRenderer) hands a frame to another gpu by rendering into a buffer the target
//! gpu can import. When no such buffer exists -- the proprietary NVIDIA driver will not allocate `LINEAR`
//! buffers through GBM, and its block-linear layouts cannot be imported elsewhere -- the only remaining path
//! used to be reading every frame back through the cpu.
//!
//! This module restores a gpu path in the opposite direction, the one PRIME uses: the *target* gpu allocates a
//! `LINEAR` buffer, and the *render* gpu copies each finished frame into it with its own copy engine. OpenGL
//! on that driver refuses to render into a foreign `LINEAR` buffer, but its Vulkan driver imports one as a
//! transfer destination just fine, so the copy is done with Vulkan while rendering stays on OpenGL.
//!
//! The same device also uploads `wl_shm` buffers (see [`VulkanCopier::shm_upload`]). OpenGL can only
//! upload those with the cpu, copying every pixel on the compositor's thread and competing with the client
//! for memory bandwidth -- a full 4K frame costs 7-20 ms there. Instead the client's memory is imported
//! into Vulkan as it is (`VK_EXT_external_memory_host`) and the gpu's copy engine pulls the damaged rows
//! into an image that OpenGL samples, synchronised through sync_files in both directions.
//!
//! Frame copies line up the same way where the driver allows it (see [`CopyJob::copy_async`]): the copy
//! waits for OpenGL on the gpu, and the display waits for the copy through the sync_file it signals, so the
//! compositor's thread waits for neither.

use std::{
    any::Any,
    fmt,
    os::fd::{AsFd, AsRawFd, FromRawFd, IntoRawFd, OwnedFd},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use ash::{ext, khr, vk};
use tracing::{debug, info};

use crate::{
    backend::{
        allocator::{Buffer, Fourcc, dmabuf::Dmabuf},
        drm::DrmNode,
        renderer::sync::{Fence, Interrupted},
    },
    utils::{Buffer as BufferCoords, Rectangle},
};

/// Errors from setting up or running Vulkan copies.
#[derive(Debug, thiserror::Error)]
pub enum VulkanCopyError {
    /// The Vulkan loader could not be loaded.
    #[error("loading the Vulkan library failed: {0}")]
    Loading(#[from] ash::LoadingError),
    /// A Vulkan call failed.
    #[error("Vulkan call failed: {0}")]
    Vk(#[from] vk::Result),
    /// The device or buffer cannot be used for copies.
    #[error("{0}")]
    Unsupported(&'static str),
}

struct Inner {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    memory_fd: khr::external_memory_fd::Device,
    /// sync_file import and export for semaphores, if the device has them (always with `shm`).
    semaphore_fd: Option<khr::external_semaphore_fd::Device>,
    queue_family: u32,
    queue: Mutex<vk::Queue>,
    pool: Mutex<vk::CommandPool>,
    /// What `wl_shm` uploads need, if the device has it.
    shm: Option<ShmSupport>,
    /// Set when a copy failed, which in practice means the device was lost (e.g. across a suspend).
    lost: AtomicBool,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // SAFETY: every CopyJob, HostImport and ShmImage holds an Arc to us, so none of the objects created
        // from this device outlive it; waiting for idle makes sure no submitted copy is still using the pools.
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(shm) = self.shm.as_mut() {
                for slot in shm.slots.get_mut().unwrap().slots.drain(..) {
                    self.device.destroy_fence(slot.fence, None);
                    self.device.destroy_semaphore(slot.wait, None);
                    self.device.destroy_semaphore(slot.signal, None);
                }
                self.device
                    .destroy_command_pool(*shm.pool.get_mut().unwrap(), None);
            }
            self.device
                .destroy_command_pool(*self.pool.get_mut().unwrap(), None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// What uploading `wl_shm` buffers needs beyond the copies.
struct ShmSupport {
    host: ext::external_memory_host::Device,
    /// Imported host pointers and sizes must be multiples of this.
    alignment: u64,
    queue_family: u32,
    /// A queue on the copy engine (a transfer-only family), so uploads run alongside rendering. `None` when
    /// the device has no such family and uploads share the main queue.
    queue: Option<Mutex<vk::Queue>>,
    pool: Mutex<vk::CommandPool>,
    slots: Mutex<UploadSlots>,
}

/// Command buffers, fences and semaphores for uploads, reused round-robin.
#[derive(Default)]
struct UploadSlots {
    slots: Vec<UploadSlot>,
    next: usize,
}

struct UploadSlot {
    cmd: vk::CommandBuffer,
    /// Signalled when the slot's last upload is done; unsignalled and unused before the first one.
    fence: vk::Fence,
    submitted: bool,
    /// Takes a temporary sync_file payload: whatever read the destination before is done with it.
    wait: vk::Semaphore,
    /// Exported as a sync_file after every submit: the upload landed.
    signal: vk::Semaphore,
}

/// Uploads in flight at once; the oldest slot is waited for when all are in use.
const MAX_UPLOAD_SLOTS: usize = 8;

/// Client memory kept imported (and so pinned) at most, across all clients.
const MAX_IMPORTED_BYTES: u64 = 512 << 20;

/// Vulkan device for the render gpu, used to copy frames into buffers the target gpu allocated, and to
/// upload `wl_shm` buffers.
pub struct VulkanCopier {
    inner: Arc<Inner>,
    /// Client memory imported for uploads, most recently used last.
    imports: Mutex<Vec<CachedImport>>,
}

struct CachedImport {
    key: usize,
    import: Arc<HostImport>,
}

impl fmt::Debug for VulkanCopier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanCopier")
            .field("queue_family", &self.inner.queue_family)
            .finish_non_exhaustive()
    }
}

const DEVICE_EXTENSIONS: [&std::ffi::CStr; 4] = [
    khr::external_memory_fd::NAME,
    ext::external_memory_dma_buf::NAME,
    ext::image_drm_format_modifier::NAME,
    ext::queue_family_foreign::NAME,
];

/// Enabled when present: sync_files in and out of the device, for frame copies and `wl_shm` uploads that
/// line up with other drivers on the gpu instead of on the cpu.
const SYNC_FD_EXTENSION: &std::ffi::CStr = khr::external_semaphore_fd::NAME;

/// Enabled when present (and sync_files are), for `wl_shm` uploads.
const SHM_EXTENSION: &std::ffi::CStr = ext::external_memory_host::NAME;

/// What `open_device` hands back.
struct OpenedDevice {
    device: ash::Device,
    memory_fd: khr::external_memory_fd::Device,
    semaphore_fd: Option<khr::external_semaphore_fd::Device>,
    queue_family: u32,
    queue: vk::Queue,
    pool: vk::CommandPool,
    shm: Option<ShmSupport>,
}

impl VulkanCopier {
    /// Open the Vulkan device backing `node`.
    pub fn new(node: &DrmNode) -> Result<Self, VulkanCopyError> {
        // Logged up front: drivers may block in here (the NVIDIA one connects to $DISPLAY).
        info!(?node, "Opening a Vulkan device for copies");
        // SAFETY: loading the system Vulkan loader.
        let entry = unsafe { ash::Entry::load()? };
        let app = vk::ApplicationInfo::default()
            .application_name(c"smithay-multigpu-copy")
            .api_version(vk::API_VERSION_1_2);
        // SAFETY: valid create info, no layers or extensions.
        let instance = unsafe {
            entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)?
        };

        match Self::open_device(&entry, &instance, node) {
            Ok(opened) => Ok(VulkanCopier {
                inner: Arc::new(Inner {
                    _entry: entry,
                    instance,
                    device: opened.device,
                    memory_fd: opened.memory_fd,
                    semaphore_fd: opened.semaphore_fd,
                    queue_family: opened.queue_family,
                    queue: Mutex::new(opened.queue),
                    pool: Mutex::new(opened.pool),
                    shm: opened.shm,
                    lost: AtomicBool::new(false),
                }),
                imports: Mutex::new(Vec::new()),
            }),
            Err(err) => {
                // SAFETY: nothing else was created from this instance.
                unsafe { instance.destroy_instance(None) };
                Err(err)
            }
        }
    }

    fn open_device(
        _entry: &ash::Entry,
        instance: &ash::Instance,
        node: &DrmNode,
    ) -> Result<OpenedDevice, VulkanCopyError> {
        let (major, minor) = (node.major() as i64, node.minor() as i64);
        // SAFETY: valid instance.
        let physical = unsafe { instance.enumerate_physical_devices()? }
            .into_iter()
            .find(|&pd| {
                // SAFETY: valid physical device.
                let exts = unsafe { instance.enumerate_device_extension_properties(pd) }.unwrap_or_default();
                let has = |name: &std::ffi::CStr| {
                    exts.iter()
                        .any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == name))
                };
                if !has(ext::physical_device_drm::NAME) || !DEVICE_EXTENSIONS.iter().all(|n| has(n)) {
                    return false;
                }
                let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
                let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut drm);
                // SAFETY: valid physical device, drm properties are supported (checked above).
                unsafe { instance.get_physical_device_properties2(pd, &mut props) };
                (drm.has_render == vk::TRUE && drm.render_major == major && drm.render_minor == minor)
                    || (drm.has_primary == vk::TRUE
                        && drm.primary_major == major
                        && drm.primary_minor == minor)
            })
            .ok_or(VulkanCopyError::Unsupported(
                "no Vulkan device with dma-buf import matches the render node",
            ))?;

        // SAFETY: valid physical device.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let queue_family = families
            .iter()
            .position(|f| {
                f.queue_flags
                    .intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER)
            })
            .ok_or(VulkanCopyError::Unsupported("no queue family can do transfers"))?
            as u32;

        // SAFETY: valid physical device.
        let exts = unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
        let has_ext = |name: &std::ffi::CStr| {
            exts.iter()
                .any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == name))
        };
        // sync_files both ways: frame copies wait for OpenGL and hand the display a fence on the gpu, and
        // `wl_shm` uploads line up with OpenGL the same way.
        let sync_fd = has_ext(SYNC_FD_EXTENSION) && {
            let mut props = vk::ExternalSemaphoreProperties::default();
            // SAFETY: valid physical device; external semaphores are core in Vulkan 1.1.
            unsafe {
                instance.get_physical_device_external_semaphore_properties(
                    physical,
                    &vk::PhysicalDeviceExternalSemaphoreInfo::default()
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD),
                    &mut props,
                )
            };
            props.external_semaphore_features.contains(
                vk::ExternalSemaphoreFeatureFlags::EXPORTABLE | vk::ExternalSemaphoreFeatureFlags::IMPORTABLE,
            )
        };
        // `wl_shm` uploads: importing host memory, on top of sync_files.
        let shm_alignment = (sync_fd && has_ext(SHM_EXTENSION))
            .then(|| {
                let mut host = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
                let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut host);
                // SAFETY: valid physical device, the extension is supported (checked above).
                unsafe { instance.get_physical_device_properties2(physical, &mut props) };
                // The rounded-up size must stay within the client's mapping, which ends on a page boundary.
                let page = rustix::param::page_size() as u64;
                let alignment = host.min_imported_host_pointer_alignment;
                (alignment > 0 && alignment <= page && page.is_multiple_of(alignment)).then_some(alignment)
            })
            .flatten();
        // Prefer the copy engine for uploads, so they do not queue up behind rendering.
        let shm_family = shm_alignment.map(|_| {
            families
                .iter()
                .position(|f| {
                    f.queue_flags.contains(vk::QueueFlags::TRANSFER)
                        && !f
                            .queue_flags
                            .intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                })
                .map_or(queue_family, |family| family as u32)
        });

        let priorities = [1.0f32];
        let mut queues = vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&priorities),
        ];
        if let Some(family) = shm_family.filter(|&family| family != queue_family) {
            queues.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&priorities),
            );
        }
        let mut ext_names = DEVICE_EXTENSIONS.map(|n| n.as_ptr()).to_vec();
        if sync_fd {
            ext_names.push(SYNC_FD_EXTENSION.as_ptr());
        }
        if shm_alignment.is_some() {
            ext_names.push(SHM_EXTENSION.as_ptr());
        }
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&ext_names);
        // SAFETY: valid physical device and create info; the extensions were checked above.
        let device = unsafe { instance.create_device(physical, &create_info, None)? };
        // SAFETY: the queue was requested at creation.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let create_pool = |family| {
            let pool_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(family);
            // SAFETY: valid device.
            unsafe { device.create_command_pool(&pool_info, None) }
        };
        let pool = match create_pool(queue_family) {
            Ok(pool) => pool,
            Err(err) => {
                // SAFETY: nothing else was created from the device.
                unsafe { device.destroy_device(None) };
                return Err(err.into());
            }
        };
        let memory_fd = khr::external_memory_fd::Device::new(instance, &device);
        let semaphore_fd = sync_fd.then(|| khr::external_semaphore_fd::Device::new(instance, &device));

        let shm = match (shm_alignment, shm_family) {
            (Some(alignment), Some(family)) => match create_pool(family) {
                Ok(shm_pool) => Some(ShmSupport {
                    host: ext::external_memory_host::Device::new(instance, &device),
                    alignment,
                    queue_family: family,
                    // SAFETY: the queue was requested at creation.
                    queue: (family != queue_family)
                        .then(|| Mutex::new(unsafe { device.get_device_queue(family, 0) })),
                    pool: Mutex::new(shm_pool),
                    slots: Mutex::new(UploadSlots::default()),
                }),
                Err(err) => {
                    debug!(?err, "No command pool for wl_shm uploads");
                    None
                }
            },
            _ => None,
        };
        info!(
            ?node,
            async_copies = semaphore_fd.is_some(),
            shm_uploads = shm.is_some(),
            shm_copy_engine = shm.as_ref().is_some_and(|shm| shm.queue.is_some()),
            "Vulkan copies available for this render node"
        );
        Ok(OpenedDevice {
            device,
            memory_fd,
            semaphore_fd,
            queue_family,
            queue,
            pool,
            shm,
        })
    }

    /// Whether a copy on this device failed; it should be replaced by a new one.
    pub fn is_lost(&self) -> bool {
        self.inner.lost.load(Ordering::Relaxed)
    }

    /// Prepare copies from `src` (rendered on the render gpu) into `dst` (allocated by the target gpu).
    ///
    /// Both buffers must be single-plane, and have the same size and memory layout (e.g. `Argb2101010` and
    /// `Xrgb2101010` will do).
    pub fn prepare(&self, src: &Dmabuf, dst: &Dmabuf) -> Result<CopyJob, VulkanCopyError> {
        let format = vk_format(src.format().code).ok_or(VulkanCopyError::Unsupported(
            "no Vulkan equivalent for the buffer format",
        ))?;
        if src.size() != dst.size() || vk_format(dst.format().code) != Some(format) {
            return Err(VulkanCopyError::Unsupported("buffers differ in size or layout"));
        }
        let inner = &self.inner;

        let src_image = import_image(inner, src, format, vk::ImageUsageFlags::TRANSFER_SRC)?;
        let dst_image = match import_image(inner, dst, format, vk::ImageUsageFlags::TRANSFER_DST) {
            Ok(image) => image,
            Err(err) => {
                src_image.destroy(inner);
                return Err(err);
            }
        };

        let pool = inner.pool.lock().unwrap();
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(*pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: valid device and pool (the pool is externally synchronised by the lock).
        let cmd = unsafe { inner.device.allocate_command_buffers(&alloc) };
        drop(pool);
        // SAFETY: valid device.
        let fence = unsafe { inner.device.create_fence(&vk::FenceCreateInfo::default(), None) };
        match (cmd, fence) {
            (Ok(cmd), Ok(fence)) => Ok(CopyJob {
                inner: inner.clone(),
                src: src_image,
                dst: dst_image,
                cmd: cmd[0],
                fence,
                in_flight: AtomicBool::new(false),
                semaphores: inner.semaphore_fd.as_ref().and_then(|_| {
                    create_copy_semaphores(inner)
                        .inspect_err(|err| debug!(?err, "No semaphores for copies the cpu does not wait for"))
                        .ok()
                }),
                width: src.width() as i32,
                height: src.height() as i32,
            }),
            (cmd, fence) => {
                // SAFETY: the objects were just created and never used.
                unsafe {
                    if let Ok(cmd) = &cmd {
                        let pool = inner.pool.lock().unwrap();
                        inner.device.free_command_buffers(*pool, cmd);
                    }
                    if let Ok(fence) = fence {
                        inner.device.destroy_fence(fence, None);
                    }
                }
                src_image.destroy(inner);
                dst_image.destroy(inner);
                Err(cmd.err().or(fence.err()).unwrap().into())
            }
        }
    }

    /// Whether this device can upload `wl_shm` buffers (see [`VulkanCopier::shm_upload`]).
    pub fn supports_shm_uploads(&self) -> bool {
        self.inner.shm.is_some()
    }

    /// Client memory the gpu can read directly: the `len` bytes mapped at `ptr`, cached under `key`.
    ///
    /// `owner` is only called on a cache miss. What it returns is held as long as the import, and the import
    /// is forgotten once it holds the last reference: the client destroyed or resized the memory, so it will
    /// not come back. Until then it stays imported however long the client pauses, since importing pins the
    /// memory again (2-6 ms for a 4K buffer) on the compositor's thread.
    ///
    /// # Safety
    ///
    /// `ptr..ptr + len` must be a mapping of whole pages that stays mapped as long as the value returned by
    /// `owner` lives, and no other mapping may be passed with the same `key` meanwhile.
    pub unsafe fn import_host_memory(
        &self,
        key: usize,
        ptr: *const u8,
        len: usize,
        owner: impl FnOnce() -> Arc<dyn Any + Send + Sync>,
    ) -> Result<Arc<HostImport>, VulkanCopyError> {
        let mut imports = self.imports.lock().unwrap();
        let import = match imports.iter().position(|cached| cached.key == key) {
            Some(idx) => {
                let cached = imports.remove(idx);
                let import = cached.import.clone();
                imports.push(cached);
                import
            }
            None => {
                // SAFETY: forwarded from the caller.
                let import = Arc::new(unsafe { self.import(ptr, len, owner())? });
                imports.push(CachedImport {
                    key,
                    import: import.clone(),
                });
                import
            }
        };

        // Forget memory the client let go of, then the least recently used beyond the budget; imports still
        // in flight are kept alive by their upload's fence.
        let current = imports.len() - 1;
        let mut idx = 0;
        imports.retain(|cached| {
            idx += 1;
            let keep = idx - 1 == current || !cached.import.is_orphaned();
            if !keep {
                debug!(
                    len = cached.import.len,
                    "Forgetting client memory the client let go of"
                );
            }
            keep
        });
        while imports.len() > 1
            && imports.iter().map(|cached| cached.import.len).sum::<u64>() > MAX_IMPORTED_BYTES
        {
            imports.remove(0);
        }
        Ok(import)
    }

    /// # Safety
    ///
    /// See [`VulkanCopier::import_host_memory`].
    unsafe fn import(
        &self,
        ptr: *const u8,
        len: usize,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<HostImport, VulkanCopyError> {
        let inner = &self.inner;
        let shm = inner
            .shm
            .as_ref()
            .ok_or(VulkanCopyError::Unsupported("no wl_shm uploads on this device"))?;
        if !(ptr as u64).is_multiple_of(shm.alignment) {
            return Err(VulkanCopyError::Unsupported(
                "client memory is not aligned for importing",
            ));
        }
        let len = (len as u64).next_multiple_of(shm.alignment);
        let handle_type = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        let device = &inner.device;

        let mut host_props = vk::MemoryHostPointerPropertiesEXT::default();
        // SAFETY: valid device; the pointer is a live mapping (guaranteed by the caller). ash has no
        // wrapper for this one.
        unsafe {
            (shm.host.fp().get_memory_host_pointer_properties_ext)(
                shm.host.device(),
                handle_type,
                ptr.cast(),
                &mut host_props,
            )
            .result()?
        };
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);
        let create_info = vk::BufferCreateInfo::default()
            .push_next(&mut external)
            .size(len)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: valid device and create info.
        let buffer = unsafe { device.create_buffer(&create_info, None)? };
        let memory = (|| {
            // SAFETY: valid buffer.
            let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
            let bits = reqs.memory_type_bits & host_props.memory_type_bits;
            if bits == 0 {
                return Err(VulkanCopyError::Unsupported(
                    "no memory type can hold the client memory",
                ));
            }
            let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
                .handle_type(handle_type)
                .host_pointer(ptr as *mut _);
            let alloc = vk::MemoryAllocateInfo::default()
                .push_next(&mut import)
                .allocation_size(len)
                .memory_type_index(bits.trailing_zeros());
            // SAFETY: valid device; the mapping outlives the memory (guaranteed by the caller).
            let memory = unsafe { device.allocate_memory(&alloc, None)? };
            // SAFETY: freshly allocated memory of a compatible type, bound once.
            if let Err(err) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
                // SAFETY: the memory was never bound.
                unsafe { device.free_memory(memory, None) };
                return Err(err.into());
            }
            Ok(memory)
        })();
        match memory {
            Ok(memory) => {
                debug!(len, "Imported client memory for wl_shm uploads");
                Ok(HostImport {
                    inner: inner.clone(),
                    buffer,
                    memory,
                    len,
                    owner,
                })
            }
            Err(err) => {
                // SAFETY: the buffer was never bound or used.
                unsafe { device.destroy_buffer(buffer, None) };
                Err(err)
            }
        }
    }

    /// Import `dmabuf`, allocated on this gpu, as the destination of `wl_shm` uploads.
    pub fn shm_image(&self, dmabuf: &Dmabuf) -> Result<Arc<ShmImage>, VulkanCopyError> {
        if self.inner.shm.is_none() {
            return Err(VulkanCopyError::Unsupported("no wl_shm uploads on this device"));
        }
        let format = shm_vk_format(dmabuf.format().code).ok_or(VulkanCopyError::Unsupported(
            "no Vulkan equivalent for the buffer format",
        ))?;
        let image = import_image(&self.inner, dmabuf, format, vk::ImageUsageFlags::TRANSFER_DST)?;
        Ok(Arc::new(ShmImage {
            inner: self.inner.clone(),
            image,
            width: dmabuf.width() as i32,
            height: dmabuf.height() as i32,
        }))
    }

    /// Copy `regions` of a `wl_shm` buffer into `dst`.
    ///
    /// The buffer starts `offset` bytes into `src` and has `stride` bytes per row, 4 bytes per pixel.
    /// `dst_free` must signal once `dst` may be overwritten, i.e. whatever read it before is done. The
    /// returned fence signals once the copy landed, and keeps `src` and `dst` alive until then; `None` means
    /// no region intersected the buffer and nothing was copied.
    ///
    /// After a failure the device counts as lost (see [`VulkanCopier::is_lost`]).
    pub fn shm_upload(
        &self,
        src: &Arc<HostImport>,
        offset: u64,
        stride: u32,
        dst: &Arc<ShmImage>,
        regions: &[Rectangle<i32, BufferCoords>],
        dst_free: OwnedFd,
    ) -> Result<Option<ShmUploadFence>, VulkanCopyError> {
        let result = self.submit_shm_upload(src, offset, stride, dst, regions, dst_free);
        if matches!(result, Err(VulkanCopyError::Vk(_))) {
            self.inner.lost.store(true, Ordering::Relaxed);
        }
        result
    }

    fn submit_shm_upload(
        &self,
        src: &Arc<HostImport>,
        offset: u64,
        stride: u32,
        dst: &Arc<ShmImage>,
        regions: &[Rectangle<i32, BufferCoords>],
        dst_free: OwnedFd,
    ) -> Result<Option<ShmUploadFence>, VulkanCopyError> {
        let inner = &self.inner;
        let shm = inner
            .shm
            .as_ref()
            .ok_or(VulkanCopyError::Unsupported("no wl_shm uploads on this device"))?;
        // Always there with `shm`.
        let semaphore_fd = inner
            .semaphore_fd
            .as_ref()
            .ok_or(VulkanCopyError::Unsupported("no sync_files on this device"))?;
        if !Arc::ptr_eq(&src.inner, inner) || !Arc::ptr_eq(&dst.inner, inner) {
            return Err(VulkanCopyError::Unsupported(
                "the memory or image belongs to a replaced device",
            ));
        }
        if !offset.is_multiple_of(4) || !stride.is_multiple_of(4) {
            return Err(VulkanCopyError::Unsupported("the buffer is not 4-byte aligned"));
        }
        let bounds = Rectangle::<i32, BufferCoords>::from_size((dst.width, dst.height).into());
        let mut copies = Vec::with_capacity(regions.len());
        for r in regions.iter().filter_map(|r| r.intersection(bounds)) {
            if r.size.w <= 0 || r.size.h <= 0 {
                continue;
            }
            let start = offset + r.loc.y as u64 * stride as u64 + r.loc.x as u64 * 4;
            let end = start + (r.size.h as u64 - 1) * stride as u64 + r.size.w as u64 * 4;
            if end > src.len {
                return Err(VulkanCopyError::Unsupported(
                    "the buffer reaches past the client memory",
                ));
            }
            copies.push(vk::BufferImageCopy {
                buffer_offset: start,
                buffer_row_length: stride / 4,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D {
                    x: r.loc.x,
                    y: r.loc.y,
                    z: 0,
                },
                image_extent: vk::Extent3D {
                    width: r.size.w as u32,
                    height: r.size.h as u32,
                    depth: 1,
                },
            });
        }
        if copies.is_empty() {
            return Ok(None);
        }

        let device = &inner.device;
        let family = shm.queue_family;
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        // As for frame copies: the image is shared with OpenGL, so take it over from the foreign queue family
        // and hand it back, in GENERAL to keep what is not damaged.
        let barrier = |from, to| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(if from == family {
                    vk::AccessFlags::TRANSFER_WRITE
                } else {
                    vk::AccessFlags::empty()
                })
                .dst_access_mask(if to == family {
                    vk::AccessFlags::TRANSFER_WRITE
                } else {
                    vk::AccessFlags::empty()
                })
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(from)
                .dst_queue_family_index(to)
                .image(dst.image.image)
                .subresource_range(range)
        };
        let foreign = vk::QUEUE_FAMILY_FOREIGN_EXT;

        let mut slots = shm.slots.lock().unwrap();
        let slot = slots.take(inner, shm)?;
        // SAFETY: the slot's command buffer, fence and semaphores are idle (`take` waited for its last
        // upload), the queue is externally synchronised by its lock, and `src`/`dst` stay alive until the
        // returned fence signals.
        unsafe {
            device.reset_command_buffer(slot.cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                slot.cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_pipeline_barrier(
                slot.cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(foreign, family)],
            );
            device.cmd_copy_buffer_to_image(
                slot.cmd,
                src.buffer,
                dst.image.image,
                vk::ImageLayout::GENERAL,
                &copies,
            );
            device.cmd_pipeline_barrier(
                slot.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(family, foreign)],
            );
            device.end_command_buffer(slot.cmd)?;

            // On success the semaphore owns the fd.
            semaphore_fd.import_semaphore_fd(
                &vk::ImportSemaphoreFdInfoKHR::default()
                    .semaphore(slot.wait)
                    .flags(vk::SemaphoreImportFlags::TEMPORARY)
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
                    .fd(dst_free.as_raw_fd()),
            )?;
            let _ = dst_free.into_raw_fd();

            let waits = [slot.wait];
            let wait_stages = [vk::PipelineStageFlags::TRANSFER];
            let cmds = [slot.cmd];
            let signals = [slot.signal];
            let submit = [vk::SubmitInfo::default()
                .wait_semaphores(&waits)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&cmds)
                .signal_semaphores(&signals)];
            {
                let queue = shm.queue.as_ref().unwrap_or(&inner.queue).lock().unwrap();
                device.queue_submit(*queue, &submit, slot.fence)?;
            }
            slot.submitted = true;

            let fd = semaphore_fd.get_semaphore_fd(
                &vk::SemaphoreGetFdInfoKHR::default()
                    .semaphore(slot.signal)
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD),
            )?;
            debug!(regions = copies.len(), "wl_shm upload submitted");
            Ok(Some(ShmUploadFence {
                // SAFETY: vkGetSemaphoreFdKHR hands us a new fd.
                fd: OwnedFd::from_raw_fd(fd),
                _src: src.clone(),
                _dst: dst.clone(),
            }))
        }
    }
}

impl UploadSlots {
    /// The next slot to record an upload in, once its previous upload is done.
    fn take(&mut self, inner: &Inner, shm: &ShmSupport) -> Result<&mut UploadSlot, VulkanCopyError> {
        let device = &inner.device;
        if self.slots.len() < MAX_UPLOAD_SLOTS && self.next == self.slots.len() {
            let pool = shm.pool.lock().unwrap();
            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(*pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let mut export = vk::ExportSemaphoreCreateInfo::default()
                .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
            // SAFETY: valid device and pool (the pool is externally synchronised by the lock); on failure
            // the objects created so far are destroyed again.
            unsafe {
                let cmd = device.allocate_command_buffers(&alloc)?[0];
                let objects = (|| {
                    let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
                    let wait = device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .inspect_err(|_| device.destroy_fence(fence, None))?;
                    let signal = device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut export), None)
                        .inspect_err(|_| {
                            device.destroy_fence(fence, None);
                            device.destroy_semaphore(wait, None);
                        })?;
                    Ok::<_, vk::Result>((fence, wait, signal))
                })();
                match objects {
                    Ok((fence, wait, signal)) => self.slots.push(UploadSlot {
                        cmd,
                        fence,
                        submitted: false,
                        wait,
                        signal,
                    }),
                    Err(err) => {
                        device.free_command_buffers(*pool, &[cmd]);
                        return Err(err.into());
                    }
                }
            }
        }
        let idx = self.next % self.slots.len();
        self.next = (idx + 1) % MAX_UPLOAD_SLOTS;
        let slot = &mut self.slots[idx];
        if slot.submitted {
            // SAFETY: valid fence, submitted with the slot's last upload.
            unsafe {
                device.wait_for_fences(&[slot.fence], true, u64::MAX)?;
                device.reset_fences(&[slot.fence])?;
            }
            slot.submitted = false;
        }
        Ok(slot)
    }
}

/// Client memory imported with [`VulkanCopier::import_host_memory`].
pub struct HostImport {
    inner: Arc<Inner>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    len: u64,
    /// Keeps the memory mapped (see [`VulkanCopier::import_host_memory`]).
    owner: Arc<dyn Any + Send + Sync>,
}

impl HostImport {
    /// Whether only this import still holds on to its memory, so the client is done with it.
    fn is_orphaned(&self) -> bool {
        Arc::strong_count(&self.owner) == 1
    }
}

impl fmt::Debug for HostImport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostImport")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl Drop for HostImport {
    fn drop(&mut self) {
        // SAFETY: not in use: every upload reading it holds an Arc until its copy landed. Dropped before
        // `owner`, so the mapping is still there.
        unsafe {
            self.inner.device.destroy_buffer(self.buffer, None);
            self.inner.device.free_memory(self.memory, None);
        }
    }
}

/// The destination of `wl_shm` uploads, see [`VulkanCopier::shm_image`].
pub struct ShmImage {
    inner: Arc<Inner>,
    image: Image,
    width: i32,
    height: i32,
}

impl fmt::Debug for ShmImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShmImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl ShmImage {
    /// Whether the image was made by `copier`'s device (and not one it replaced, e.g. after a suspend).
    pub fn is_from(&self, copier: &VulkanCopier) -> bool {
        Arc::ptr_eq(&self.inner, &copier.inner)
    }
}

impl Drop for ShmImage {
    fn drop(&mut self) {
        // Not in use: every upload writing it holds an Arc until its copy landed.
        self.image.destroy(&self.inner);
    }
}

/// Signalled once a `wl_shm` upload landed, as a sync_file.
///
/// Keeps the client memory and the destination alive until then; dropping it before blocks until the copy
/// is done.
pub struct ShmUploadFence {
    fd: OwnedFd,
    _src: Arc<HostImport>,
    _dst: Arc<ShmImage>,
}

impl fmt::Debug for ShmUploadFence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShmUploadFence")
            .field("fd", &self.fd)
            .finish_non_exhaustive()
    }
}

impl Fence for ShmUploadFence {
    fn is_signaled(&self) -> bool {
        poll_sync_file(&self.fd, Some(&ZERO_TIMEOUT)).unwrap_or(false)
    }

    fn wait(&self) -> Result<(), Interrupted> {
        wait_sync_file(&self.fd)
    }

    fn is_exportable(&self) -> bool {
        true
    }

    fn export(&self) -> Option<OwnedFd> {
        self.fd.as_fd().try_clone_to_owned().ok()
    }
}

impl Drop for ShmUploadFence {
    fn drop(&mut self) {
        // The copy must not outlive what it reads and writes.
        if !self.is_signaled() {
            let _ = self.wait();
        }
    }
}

/// Formats `wl_shm` uploads handle: 4 bytes per pixel, copied as they are.
fn shm_vk_format(fourcc: Fourcc) -> Option<vk::Format> {
    match fourcc {
        Fourcc::Argb8888
        | Fourcc::Xrgb8888
        | Fourcc::Abgr8888
        | Fourcc::Xbgr8888
        | Fourcc::Argb2101010
        | Fourcc::Xrgb2101010
        | Fourcc::Abgr2101010
        | Fourcc::Xbgr2101010 => vk_format(fourcc),
        _ => None,
    }
}

/// Whether `wl_shm` buffers in `fourcc` can be uploaded through Vulkan.
pub fn shm_format_supported(fourcc: Fourcc) -> bool {
    shm_vk_format(fourcc).is_some()
}

struct Image {
    image: vk::Image,
    memory: vk::DeviceMemory,
}

impl Image {
    fn destroy(&self, inner: &Inner) {
        // SAFETY: the image is not in use (copies are waited for before returning).
        unsafe {
            inner.device.destroy_image(self.image, None);
            inner.device.free_memory(self.memory, None);
        }
    }
}

fn import_image(
    inner: &Inner,
    buffer: &Dmabuf,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Result<Image, VulkanCopyError> {
    if buffer.num_planes() != 1 {
        return Err(VulkanCopyError::Unsupported(
            "multi-planar buffers are not supported",
        ));
    }
    let offset = buffer.offsets().next().unwrap() as u64;
    let stride = buffer.strides().next().unwrap() as u64;
    let plane = [vk::SubresourceLayout {
        offset,
        size: 0,
        row_pitch: stride,
        array_pitch: 0,
        depth_pitch: 0,
    }];
    let mut modifier = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
        .drm_format_modifier(buffer.format().modifier.into())
        .plane_layouts(&plane);
    let mut external = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let create_info = vk::ImageCreateInfo::default()
        .push_next(&mut modifier)
        .push_next(&mut external)
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: buffer.width(),
            height: buffer.height(),
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let device = &inner.device;
    // SAFETY: valid device and create info.
    let image = unsafe { device.create_image(&create_info, None)? };

    let result = (|| {
        let fd: OwnedFd = buffer
            .handles()
            .next()
            .unwrap()
            .try_clone_to_owned()
            .map_err(|_| VulkanCopyError::Unsupported("could not duplicate the dma-buf fd"))?;
        let mut fd_props = vk::MemoryFdPropertiesKHR::default();
        // SAFETY: valid device and fd.
        unsafe {
            inner.memory_fd.get_memory_fd_properties(
                vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                fd.as_raw_fd(),
                &mut fd_props,
            )?
        };
        // SAFETY: valid image.
        let reqs = unsafe { device.get_image_memory_requirements(image) };
        let bits = reqs.memory_type_bits & fd_props.memory_type_bits;
        if bits == 0 {
            return Err(VulkanCopyError::Unsupported(
                "no memory type can hold the dma-buf",
            ));
        }
        let memory_type = bits.trailing_zeros();
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(fd.as_raw_fd());
        let alloc = vk::MemoryAllocateInfo::default()
            .push_next(&mut dedicated)
            .push_next(&mut import)
            .allocation_size(reqs.size)
            .memory_type_index(memory_type);
        // SAFETY: valid device and allocate info; on success Vulkan owns the fd.
        let memory = unsafe { device.allocate_memory(&alloc, None)? };
        let _ = fd.into_raw_fd();
        // SAFETY: freshly allocated memory of a compatible type, bound once.
        if let Err(err) = unsafe { device.bind_image_memory(image, memory, 0) } {
            // SAFETY: the memory was never bound.
            unsafe { device.free_memory(memory, None) };
            return Err(err.into());
        }
        Ok(memory)
    })();

    match result {
        Ok(memory) => Ok(Image { image, memory }),
        Err(err) => {
            // SAFETY: the image was never bound or used.
            unsafe { device.destroy_image(image, None) };
            Err(err)
        }
    }
}

fn vk_format(fourcc: Fourcc) -> Option<vk::Format> {
    // Only the byte layout matters for a copy, so alpha and padding formats share an entry.
    Some(match fourcc {
        Fourcc::Argb8888 | Fourcc::Xrgb8888 => vk::Format::B8G8R8A8_UNORM,
        Fourcc::Abgr8888 | Fourcc::Xbgr8888 => vk::Format::R8G8B8A8_UNORM,
        Fourcc::Argb2101010 | Fourcc::Xrgb2101010 => vk::Format::A2R10G10B10_UNORM_PACK32,
        Fourcc::Abgr2101010 | Fourcc::Xbgr2101010 => vk::Format::A2B10G10R10_UNORM_PACK32,
        _ => return None,
    })
}

/// A prepared copy between one pair of buffers.
pub struct CopyJob {
    inner: Arc<Inner>,
    src: Image,
    dst: Image,
    cmd: vk::CommandBuffer,
    /// Signalled when the last submitted copy is done.
    fence: vk::Fence,
    /// A copy was submitted, and its fence has not been seen signalled (and reset) since.
    in_flight: AtomicBool,
    /// For copies the cpu does not wait for (see [`CopyJob::copy_async`]); `None` without sync_files.
    semaphores: Option<CopySemaphores>,
    width: i32,
    height: i32,
}

struct CopySemaphores {
    /// Takes a temporary sync_file payload: the render gpu finished writing the source.
    wait: vk::Semaphore,
    /// Exported as a sync_file after every submit: the copy landed.
    signal: vk::Semaphore,
}

impl fmt::Debug for CopyJob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CopyJob")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("async", &self.semaphores.is_some())
            .finish_non_exhaustive()
    }
}

impl CopyJob {
    /// Copy `regions` of the source buffer into the destination buffer, and wait for the copy to land.
    ///
    /// The caller must make sure the render gpu has finished writing the source first. After a failure
    /// the device counts as lost (see [`CopyJob::is_lost`]).
    pub fn copy(&self, regions: &[Rectangle<i32, BufferCoords>]) -> Result<(), VulkanCopyError> {
        let result = self.submit_and_wait(regions);
        if result.is_err() {
            self.inner.lost.store(true, Ordering::Relaxed);
        }
        result
    }

    /// Copy `regions` of the source buffer into the destination buffer, without waiting on the cpu.
    ///
    /// The gpu starts the copy once `ready` signals, a sync_file for the render gpu's writes to the source;
    /// without one, the caller must make sure the source is complete. The returned fence signals once the
    /// copy landed, and can go to the display as a sync_file. `None` if there was nothing to copy.
    ///
    /// Fails with [`VulkanCopyError::Unsupported`] if the device has no sync_files, and [`CopyJob::copy`]
    /// has to do instead. After any other failure the device counts as lost.
    pub fn copy_async(
        self: &Arc<Self>,
        regions: &[Rectangle<i32, BufferCoords>],
        ready: Option<OwnedFd>,
    ) -> Result<Option<FrameCopyFence>, VulkanCopyError> {
        let (Some(semaphores), Some(semaphore_fd)) =
            (self.semaphores.as_ref(), self.inner.semaphore_fd.as_ref())
        else {
            return Err(VulkanCopyError::Unsupported("no sync_files for this copy"));
        };
        let result = self.submit_async(semaphores, semaphore_fd, regions, ready);
        if result.is_err() {
            self.inner.lost.store(true, Ordering::Relaxed);
        }
        result
    }

    /// Whether a copy on this job's device failed; the job should be replaced along with the device.
    pub fn is_lost(&self) -> bool {
        self.inner.lost.load(Ordering::Relaxed)
    }

    /// Make the command buffer and fence reusable. Normally the last copy landed long ago: the display only
    /// lets go of a buffer after showing it, and it shows it only after the copy into it is done.
    fn reclaim(&self) -> Result<(), VulkanCopyError> {
        if !self.in_flight.load(Ordering::Relaxed) {
            return Ok(());
        }
        let device = &self.inner.device;
        // SAFETY: valid fence, submitted with the last copy.
        unsafe {
            if !device.get_fence_status(self.fence)? {
                debug!("Reusing a copy job whose last copy is still running");
                device.wait_for_fences(&[self.fence], true, u64::MAX)?;
            }
            device.reset_fences(&[self.fence])?;
        }
        self.in_flight.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// Record the copy of `regions` into the command buffer: the number of rectangles, `None` if there is
    /// nothing to copy.
    fn record(&self, regions: &[Rectangle<i32, BufferCoords>]) -> Result<Option<usize>, VulkanCopyError> {
        let bounds = Rectangle::<i32, BufferCoords>::from_size((self.width, self.height).into());
        let copies = regions
            .iter()
            .filter_map(|r| r.intersection(bounds))
            .filter(|r| r.size.w > 0 && r.size.h > 0)
            .map(|r| {
                let subresource = vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let offset = vk::Offset3D {
                    x: r.loc.x,
                    y: r.loc.y,
                    z: 0,
                };
                vk::ImageCopy {
                    src_subresource: subresource,
                    src_offset: offset,
                    dst_subresource: subresource,
                    dst_offset: offset,
                    extent: vk::Extent3D {
                        width: r.size.w as u32,
                        height: r.size.h as u32,
                        depth: 1,
                    },
                }
            })
            .collect::<Vec<_>>();
        if copies.is_empty() {
            return Ok(None);
        }

        let device = &self.inner.device;
        let family = self.inner.queue_family;
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        // Both buffers are shared with other drivers (GL on this gpu, the target gpu), so take them over
        // from the foreign queue family for the copy and hand them back afterwards. GENERAL keeps the
        // contents: only damaged regions are copied, the rest of the destination must survive.
        let barrier = |image, access, from, to| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(if from == family {
                    access
                } else {
                    vk::AccessFlags::empty()
                })
                .dst_access_mask(if to == family {
                    access
                } else {
                    vk::AccessFlags::empty()
                })
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(from)
                .dst_queue_family_index(to)
                .image(image)
                .subresource_range(range)
        };
        let foreign = vk::QUEUE_FAMILY_FOREIGN_EXT;
        let acquire = [
            barrier(self.src.image, vk::AccessFlags::TRANSFER_READ, foreign, family),
            barrier(self.dst.image, vk::AccessFlags::TRANSFER_WRITE, foreign, family),
        ];
        let release = [
            barrier(self.src.image, vk::AccessFlags::TRANSFER_READ, family, foreign),
            barrier(self.dst.image, vk::AccessFlags::TRANSFER_WRITE, family, foreign),
        ];

        // SAFETY: the command buffer belongs to this job and is idle (`reclaim` saw its last copy done).
        unsafe {
            device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &acquire,
            );
            device.cmd_copy_image(
                self.cmd,
                self.src.image,
                vk::ImageLayout::GENERAL,
                self.dst.image,
                vk::ImageLayout::GENERAL,
                &copies,
            );
            device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &release,
            );
            device.end_command_buffer(self.cmd)?;
        }
        Ok(Some(copies.len()))
    }

    fn submit_and_wait(&self, regions: &[Rectangle<i32, BufferCoords>]) -> Result<(), VulkanCopyError> {
        self.reclaim()?;
        let Some(count) = self.record(regions)? else {
            return Ok(());
        };
        let device = &self.inner.device;
        // SAFETY: the command buffer was just recorded, the fence is unsignalled (`reclaim`), and the queue is
        // externally synchronised by the lock.
        unsafe {
            let cmds = [self.cmd];
            let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
            {
                let queue = self.inner.queue.lock().unwrap();
                device.queue_submit(*queue, &submit, self.fence)?;
            }
            let waited = device.wait_for_fences(&[self.fence], true, u64::MAX);
            device.reset_fences(&[self.fence])?;
            waited?;
        }
        debug!(regions = count, "vulkan copy done");
        Ok(())
    }

    fn submit_async(
        self: &Arc<Self>,
        semaphores: &CopySemaphores,
        semaphore_fd: &khr::external_semaphore_fd::Device,
        regions: &[Rectangle<i32, BufferCoords>],
        ready: Option<OwnedFd>,
    ) -> Result<Option<FrameCopyFence>, VulkanCopyError> {
        self.reclaim()?;
        let Some(count) = self.record(regions)? else {
            return Ok(None);
        };
        let device = &self.inner.device;
        // SAFETY: the command buffer was just recorded, the fence is unsignalled (`reclaim`), the semaphores
        // belong to this job and have no pending operations (every signal was exported, which consumes it, and
        // every temporary payload was waited for), and the queue is externally synchronised by the lock.
        unsafe {
            let waited_on = match ready {
                Some(ready) => {
                    let imported = semaphore_fd.import_semaphore_fd(
                        &vk::ImportSemaphoreFdInfoKHR::default()
                            .semaphore(semaphores.wait)
                            .flags(vk::SemaphoreImportFlags::TEMPORARY)
                            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
                            .fd(ready.as_raw_fd()),
                    );
                    match imported {
                        // On success the semaphore owns the fd.
                        Ok(()) => {
                            let _ = ready.into_raw_fd();
                            true
                        }
                        Err(err) => {
                            debug!(?err, "Cannot wait for rendering on the gpu, waiting on the cpu");
                            let _ = wait_sync_file(&ready);
                            false
                        }
                    }
                }
                None => false,
            };
            let wait_list = [semaphores.wait];
            let waits: &[vk::Semaphore] = if waited_on { &wait_list } else { &[] };
            let wait_stages = [vk::PipelineStageFlags::TRANSFER];
            let cmds = [self.cmd];
            let signals = [semaphores.signal];
            let submit = [vk::SubmitInfo::default()
                .wait_semaphores(waits)
                .wait_dst_stage_mask(&wait_stages[..waits.len()])
                .command_buffers(&cmds)
                .signal_semaphores(&signals)];
            {
                let queue = self.inner.queue.lock().unwrap();
                device.queue_submit(*queue, &submit, self.fence)?;
            }
            self.in_flight.store(true, Ordering::Relaxed);

            let fd = semaphore_fd.get_semaphore_fd(
                &vk::SemaphoreGetFdInfoKHR::default()
                    .semaphore(semaphores.signal)
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD),
            )?;
            debug!(regions = count, gpu_wait = waited_on, "vulkan copy submitted");
            Ok(Some(FrameCopyFence {
                // SAFETY: vkGetSemaphoreFdKHR hands us a new fd.
                fd: OwnedFd::from_raw_fd(fd),
                _job: self.clone(),
            }))
        }
    }
}

impl Drop for CopyJob {
    fn drop(&mut self) {
        let inner = &self.inner;
        // SAFETY: a copy still in flight is waited for first, so none of the objects is in use any more.
        unsafe {
            if *self.in_flight.get_mut() {
                let _ = inner.device.wait_for_fences(&[self.fence], true, u64::MAX);
            }
            let pool = inner.pool.lock().unwrap();
            inner.device.free_command_buffers(*pool, &[self.cmd]);
            drop(pool);
            inner.device.destroy_fence(self.fence, None);
            if let Some(semaphores) = self.semaphores.take() {
                inner.device.destroy_semaphore(semaphores.wait, None);
                inner.device.destroy_semaphore(semaphores.signal, None);
            }
        }
        self.src.destroy(inner);
        self.dst.destroy(inner);
    }
}

/// Signalled once a frame copy landed, as a sync_file (see [`CopyJob::copy_async`]).
///
/// Holds on to the job, so the buffers it reads and writes stay imported until the copy is done.
pub struct FrameCopyFence {
    fd: OwnedFd,
    _job: Arc<CopyJob>,
}

impl fmt::Debug for FrameCopyFence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrameCopyFence")
            .field("fd", &self.fd)
            .finish_non_exhaustive()
    }
}

impl Fence for FrameCopyFence {
    fn is_signaled(&self) -> bool {
        poll_sync_file(&self.fd, Some(&ZERO_TIMEOUT)).unwrap_or(false)
    }

    fn wait(&self) -> Result<(), Interrupted> {
        wait_sync_file(&self.fd)
    }

    fn is_exportable(&self) -> bool {
        true
    }

    fn export(&self) -> Option<OwnedFd> {
        self.fd.as_fd().try_clone_to_owned().ok()
    }
}

const ZERO_TIMEOUT: rustix::time::Timespec = rustix::time::Timespec {
    tv_sec: 0,
    tv_nsec: 0,
};

/// Whether `fd`, a sync_file, signalled within `timeout` (forever if `None`).
fn poll_sync_file(fd: &OwnedFd, timeout: Option<&rustix::time::Timespec>) -> rustix::io::Result<bool> {
    let mut fds = [rustix::event::PollFd::new(fd, rustix::event::PollFlags::IN)];
    rustix::event::poll(&mut fds, timeout).map(|n| n > 0)
}

/// Block until `fd`, a sync_file, signals.
fn wait_sync_file(fd: &OwnedFd) -> Result<(), Interrupted> {
    loop {
        match poll_sync_file(fd, None) {
            Ok(true) => return Ok(()),
            Ok(false) | Err(rustix::io::Errno::INTR) => continue,
            Err(_) => return Err(Interrupted),
        }
    }
}

fn create_copy_semaphores(inner: &Inner) -> Result<CopySemaphores, vk::Result> {
    let device = &inner.device;
    let mut export =
        vk::ExportSemaphoreCreateInfo::default().handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
    // SAFETY: valid device; on failure the semaphore created first is destroyed again.
    unsafe {
        let wait = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
        match device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut export), None) {
            Ok(signal) => Ok(CopySemaphores { wait, signal }),
            Err(err) => {
                device.destroy_semaphore(wait, None);
                Err(err)
            }
        }
    }
}
