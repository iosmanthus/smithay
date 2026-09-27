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

use std::{
    fmt,
    os::fd::{AsRawFd, IntoRawFd, OwnedFd},
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
    queue_family: u32,
    queue: Mutex<vk::Queue>,
    pool: Mutex<vk::CommandPool>,
    /// Set when a copy failed, which in practice means the device was lost (e.g. across a suspend).
    lost: AtomicBool,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // SAFETY: every CopyJob holds an Arc to us, so none of the objects created from this device
        // outlive it; waiting for idle makes sure no submitted copy is still using the pool.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device
                .destroy_command_pool(*self.pool.get_mut().unwrap(), None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Vulkan device for the render gpu, used to copy frames into buffers the target gpu allocated.
pub struct VulkanCopier {
    inner: Arc<Inner>,
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
            Ok((device, memory_fd, queue_family, queue, pool)) => Ok(VulkanCopier {
                inner: Arc::new(Inner {
                    _entry: entry,
                    instance,
                    device,
                    memory_fd,
                    queue_family,
                    queue: Mutex::new(queue),
                    pool: Mutex::new(pool),
                    lost: AtomicBool::new(false),
                }),
            }),
            Err(err) => {
                // SAFETY: nothing else was created from this instance.
                unsafe { instance.destroy_instance(None) };
                Err(err)
            }
        }
    }

    #[allow(clippy::type_complexity)]
    fn open_device(
        _entry: &ash::Entry,
        instance: &ash::Instance,
        node: &DrmNode,
    ) -> Result<
        (
            ash::Device,
            khr::external_memory_fd::Device,
            u32,
            vk::Queue,
            vk::CommandPool,
        ),
        VulkanCopyError,
    > {
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

        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let ext_names = DEVICE_EXTENSIONS.map(|n| n.as_ptr());
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&ext_names);
        // SAFETY: valid physical device and create info; the extensions were checked above.
        let device = unsafe { instance.create_device(physical, &create_info, None)? };
        // SAFETY: the queue was requested at creation.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family);
        // SAFETY: valid device.
        let pool = match unsafe { device.create_command_pool(&pool_info, None) } {
            Ok(pool) => pool,
            Err(err) => {
                // SAFETY: nothing else was created from the device.
                unsafe { device.destroy_device(None) };
                return Err(err.into());
            }
        };
        let memory_fd = khr::external_memory_fd::Device::new(instance, &device);
        info!(?node, "Vulkan copies available for this render node");
        Ok((device, memory_fd, queue_family, queue, pool))
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
    fence: vk::Fence,
    width: i32,
    height: i32,
}

impl fmt::Debug for CopyJob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CopyJob")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl CopyJob {
    /// Copy `regions` of the source buffer into the destination buffer, and wait for the copy to land.
    ///
    /// The caller must make sure the render gpu has finished writing the source first. After a failure
    /// the device counts as lost (see [`CopyJob::is_lost`]).
    pub fn copy(&self, regions: &[Rectangle<i32, BufferCoords>]) -> Result<(), VulkanCopyError> {
        let result = self.submit(regions);
        if result.is_err() {
            self.inner.lost.store(true, Ordering::Relaxed);
        }
        result
    }

    /// Whether a copy on this job's device failed; the job and its buffers should be set up again.
    pub fn is_lost(&self) -> bool {
        self.inner.lost.load(Ordering::Relaxed)
    }

    fn submit(&self, regions: &[Rectangle<i32, BufferCoords>]) -> Result<(), VulkanCopyError> {
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
            return Ok(());
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

        // SAFETY: the command buffer and fence belong to this job and are idle (every copy waits for its
        // fence before returning), and the queue is externally synchronised by the lock.
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
        debug!(regions = copies.len(), "vulkan copy done");
        Ok(())
    }
}

impl Drop for CopyJob {
    fn drop(&mut self) {
        let inner = &self.inner;
        // SAFETY: copies always wait for their fence, so nothing here is in flight.
        unsafe {
            let pool = inner.pool.lock().unwrap();
            inner.device.free_command_buffers(*pool, &[self.cmd]);
            drop(pool);
            inner.device.destroy_fence(self.fence, None);
        }
        self.src.destroy(inner);
        self.dst.destroy(inner);
    }
}
