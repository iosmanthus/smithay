use std::cell::RefCell;
use std::os::unix::io::OwnedFd;
use std::sync::Arc;

use tracing::debug;
use wayland_protocols::ext::data_control::v1::server::ext_data_control_offer_v1::{
    self, ExtDataControlOfferV1,
};
use wayland_protocols::wp::primary_selection::zv1::server::zwp_primary_selection_offer_v1::{
    self, ZwpPrimarySelectionOfferV1 as PrimaryOffer,
};
use wayland_protocols_wlr::data_control::v1::server::zwlr_data_control_offer_v1::{
    self, ZwlrDataControlOfferV1,
};
use wayland_server::DisplayHandle;
use wayland_server::backend::ObjectId;
use wayland_server::backend::protocol::Message;
use wayland_server::backend::{ClientId, Handle, ObjectData};
use wayland_server::protocol::wl_data_offer;
use wayland_server::protocol::wl_seat::WlSeat;
use wayland_server::{Resource, protocol::wl_data_offer::WlDataOffer};
use wl_data_offer::Request as DataOfferRequest;
use zwlr_data_control_offer_v1::Request as DataControlRequest;
use zwp_primary_selection_offer_v1::Request as PrimaryRequest;

use crate::input::Seat;

use super::device::{SelectionDevice, SelectionDeviceKind};
use super::private::selection_dispatch;
use super::seat_data::SeatData;
use super::source::{CompositorSelectionProvider, SelectionSourceProvider};
use super::{SelectionHandler, SelectionRead, SelectionReadSource, SelectionSource, SelectionTarget};

#[derive(Debug, Clone)]
pub enum OfferReplySource<U: Clone + Send + Sync + 'static> {
    /// The selection is backend by client source.
    Client(SelectionSourceProvider),
    /// The selection is backed by the compositor.
    Compositor(CompositorSelectionProvider<U>),
}

impl<U: Clone + Send + Sync + 'static> OfferReplySource<U> {
    /// Get the copy of the underlying mime types.
    pub fn mime_types(&self) -> Vec<String> {
        match self {
            OfferReplySource::Client(source) => source.mime_types(),
            OfferReplySource::Compositor(source) => source.mime_types.clone(),
        }
    }
}

/// Offer representing various selection offers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SelectionOffer {
    DataDevice(WlDataOffer),
    Primary(PrimaryOffer),
    WlrDataControl(ZwlrDataControlOfferV1),
    ExtDataControl(ExtDataControlOfferV1),
}

impl SelectionOffer {
    /// An offer of the selection `data` of `target` with `mime_types`, made on `device` when the
    /// selection had changed `generation` times. The caller sends the mime types.
    pub fn new<D>(
        dh: &DisplayHandle,
        device: &SelectionDevice,
        client_id: ClientId,
        data: OfferReplySource<D::SelectionUserData>,
        mime_types: Vec<String>,
        target: SelectionTarget,
        generation: u64,
    ) -> Self
    where
        D: SelectionHandler + 'static,
    {
        // NOTE: the types are tied to the `SelectionDevice`, so every
        // RTTI like checking is safe and reliable.

        let device_kind = device.device_kind();
        let data = Arc::new(OfferReplyData {
            device_kind,
            seat: device.seat(),
            source: data,
            mime_types,
            target,
            generation,
        });
        let backend = dh.backend_handle();

        let interface = match device_kind {
            SelectionDeviceKind::DataDevice => WlDataOffer::interface(),
            SelectionDeviceKind::PrimarySelection => PrimaryOffer::interface(),
            SelectionDeviceKind::WlrDataControl => ZwlrDataControlOfferV1::interface(),
            SelectionDeviceKind::ExtDataControl => ExtDataControlOfferV1::interface(),
        };

        let offer = backend
            .create_object::<D>(client_id, interface, device.version(), data)
            .unwrap();

        match device_kind {
            SelectionDeviceKind::DataDevice => Self::DataDevice(WlDataOffer::from_id(dh, offer).unwrap()),
            SelectionDeviceKind::PrimarySelection => Self::Primary(PrimaryOffer::from_id(dh, offer).unwrap()),
            SelectionDeviceKind::WlrDataControl => {
                Self::WlrDataControl(ZwlrDataControlOfferV1::from_id(dh, offer).unwrap())
            }
            SelectionDeviceKind::ExtDataControl => {
                Self::ExtDataControl(ExtDataControlOfferV1::from_id(dh, offer).unwrap())
            }
        }
    }

    pub fn offer(&self, mime_type: String) {
        selection_dispatch!(self; Self(offer) => offer.offer(mime_type))
    }
}

struct OfferReplyData<U: Clone + Send + Sync + 'static> {
    device_kind: SelectionDeviceKind,
    source: OfferReplySource<U>,
    /// The mime types the offer announced; a source can add types later, which this offer did not
    /// announce.
    mime_types: Vec<String>,
    seat: WlSeat,
    target: SelectionTarget,
    generation: u64,
}

impl<D> ObjectData<D> for OfferReplyData<D::SelectionUserData>
where
    D: SelectionHandler + 'static,
{
    fn request(
        self: Arc<Self>,
        dh: &Handle,
        handle: &mut D,
        _: ClientId,
        msg: Message<ObjectId, OwnedFd>,
    ) -> Option<Arc<dyn ObjectData<D>>> {
        let dh = DisplayHandle::from(dh.clone());

        // NOTE: we can't parse message more than once, since it expects the `OwnedFd` which
        // we can't clone. To achieve that, we use RTTI passed along the selection data, to
        // make the parsing work only once.
        let (reader, mime_type, fd, object_name) = match self.device_kind {
            SelectionDeviceKind::DataDevice => {
                if let Ok((resource, DataOfferRequest::Receive { mime_type, fd })) =
                    WlDataOffer::parse_request(&dh, msg)
                {
                    (resource.client(), mime_type, fd, "wl_data_offer")
                } else {
                    return None;
                }
            }
            SelectionDeviceKind::PrimarySelection => {
                if let Ok((resource, PrimaryRequest::Receive { mime_type, fd })) =
                    PrimaryOffer::parse_request(&dh, msg)
                {
                    (resource.client(), mime_type, fd, "primary_selection_offer")
                } else {
                    return None;
                }
            }
            SelectionDeviceKind::WlrDataControl => {
                if let Ok((resource, DataControlRequest::Receive { mime_type, fd })) =
                    ZwlrDataControlOfferV1::parse_request(&dh, msg)
                {
                    (resource.client(), mime_type, fd, "wlr_data_control_offer")
                } else {
                    return None;
                }
            }
            SelectionDeviceKind::ExtDataControl => {
                if let Ok((resource, ext_data_control_offer_v1::Request::Receive { mime_type, fd })) =
                    ExtDataControlOfferV1::parse_request(&dh, msg)
                {
                    (resource.client(), mime_type, fd, "ext_data_control_offer")
                } else {
                    return None;
                }
            }
        };

        if !self.mime_types.contains(&mime_type) {
            debug!("Denying a {object_name}.receive with a mime type that was not offered");
            return None;
        }

        // Every read goes through the handler, which may control access: without a seat or a
        // reader there is nobody to ask, and the read is refused.
        let (Some(reader), Some(seat)) = (reader, Seat::<D>::from_resource(&self.seat)) else {
            debug!("Denying a {object_name}.receive without a seat");
            return None;
        };
        let current = seat
            .user_data()
            .get::<RefCell<SeatData<D::SelectionUserData>>>()
            .is_some_and(|data| data.borrow().generation(self.target) == self.generation);
        let source = match &self.source {
            OfferReplySource::Client(provider) => SelectionReadSource::Client(SelectionSource {
                provider: provider.clone(),
            }),
            OfferReplySource::Compositor(source) => {
                debug_assert_eq!(source.ty, self.target);
                SelectionReadSource::Compositor(source.user_data.clone())
            }
        };
        let read = SelectionRead {
            target: self.target,
            device_kind: self.device_kind,
            reader,
            source,
            mime_type,
            generation: self.generation,
            current,
        };
        handle.receive_selection(read, fd, seat);

        None
    }

    fn destroyed(self: Arc<Self>, _: &Handle, _: &mut D, _: ClientId, _: ObjectId) {}
}
