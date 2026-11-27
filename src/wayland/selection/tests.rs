//! Selection reads through [`SelectionHandler::receive_selection`], with clients in the same
//! process. A data-control client owns the selection (it needs no keyboard focus to set it); a
//! `wl_data_device` client, focused with [`set_data_device_focus`], or another data-control
//! client reads it.

use std::fs::File;
use std::io::{Read, Write};
use std::os::unix::io::{AsFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::sync::Arc;

use wayland_client::protocol::{
    wl_callback, wl_data_device, wl_data_device_manager, wl_data_offer, wl_registry, wl_seat,
};
use wayland_client::{Connection, Dispatch, EventQueue, QueueHandle, event_created_child};
use wayland_protocols::ext::data_control::v1::client::{
    ext_data_control_device_v1, ext_data_control_manager_v1, ext_data_control_offer_v1,
    ext_data_control_source_v1,
};
use wayland_server::backend::{ClientData, ClientId, DisconnectReason};
use wayland_server::protocol::wl_surface::WlSurface;
use wayland_server::{Client, Display, DisplayHandle};

use super::data_device::{
    DataDeviceHandler, DataDeviceState, WaylandDndGrabHandler, set_data_device_focus,
    set_data_device_selection,
};
use super::ext_data_control::{DataControlHandler, DataControlState};
use super::{DataDeviceKind, SelectionHandler, SelectionRead, SelectionReadSource, SelectionTarget};
use crate::input::{Seat, SeatHandler, SeatState};
use crate::wayland::compositor::{CompositorClientState, CompositorHandler, CompositorState};
use crate::wayland::pointer_constraints::PointerConstraintsHandler;

const TEXT: &str = "text/plain;charset=utf-8";

/// What the compositor saw of one read.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Seen {
    target: SelectionTarget,
    device_kind: DataDeviceKind,
    reader: ClientId,
    /// The owner of a client source; `None` for the compositor's selection.
    source_client: Option<ClientId>,
    compositor_data: Option<&'static str>,
    mime_type: String,
    current: bool,
}

/// How the compositor serves reads.
enum Serve {
    Forward,
    Write(&'static str),
}

struct State {
    compositor: CompositorState,
    seat_state: SeatState<State>,
    data_device: DataDeviceState,
    data_control: DataControlState,
    seen: Vec<Seen>,
    serve: Serve,
}

impl SeatHandler for State {
    type KeyboardFocus = WlSurface;
    type PointerFocus = WlSurface;
    type TouchFocus = WlSurface;

    fn seat_state(&mut self) -> &mut SeatState<Self> {
        &mut self.seat_state
    }
}

impl SelectionHandler for State {
    type SelectionUserData = &'static str;

    fn send_selection(
        &mut self,
        _ty: SelectionTarget,
        _mime_type: String,
        fd: OwnedFd,
        _seat: Seat<Self>,
        user_data: &&'static str,
    ) {
        File::from(fd).write_all(user_data.as_bytes()).unwrap();
    }

    fn receive_selection(&mut self, read: SelectionRead<&'static str>, fd: OwnedFd, seat: Seat<Self>) {
        let (source_client, compositor_data) = match &read.source {
            SelectionReadSource::Client(source) => (source.client().map(|c| c.id()), None),
            SelectionReadSource::Compositor(data) => (None, Some(*data)),
        };
        self.seen.push(Seen {
            target: read.target,
            device_kind: read.device_kind,
            reader: read.reader.id(),
            source_client,
            compositor_data,
            mime_type: read.mime_type.clone(),
            current: read.current,
        });
        match self.serve {
            Serve::Forward => read.forward(self, fd, seat),
            Serve::Write(data) => File::from(fd).write_all(data.as_bytes()).unwrap(),
        }
    }
}

// The seat's focus types are surfaces, which need these.
impl CompositorHandler for State {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor
    }

    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        &client.get_data::<TestClientData>().unwrap().compositor
    }

    fn commit(&mut self, _surface: &WlSurface) {}
}

impl PointerConstraintsHandler for State {}

impl WaylandDndGrabHandler for State {}

impl DataDeviceHandler for State {
    fn data_device_state(&mut self) -> &mut DataDeviceState {
        &mut self.data_device
    }
}

impl DataControlHandler for State {
    fn data_control_state(&mut self) -> &mut DataControlState {
        &mut self.data_control
    }
}

crate::delegate_dispatch2!(State);

#[derive(Default)]
struct TestClientData {
    compositor: CompositorClientState,
}

impl ClientData for TestClientData {
    fn initialized(&self, _client_id: ClientId) {}
    fn disconnected(&self, _client_id: ClientId, _reason: DisconnectReason) {}
}

/// The client side of a test client.
#[derive(Default)]
struct App {
    seat: Option<wl_seat::WlSeat>,
    data_device_manager: Option<wl_data_device_manager::WlDataDeviceManager>,
    data_control_manager: Option<ext_data_control_manager_v1::ExtDataControlManagerV1>,
    /// The last `wl_data_device.selection`.
    selection: Option<wl_data_offer::WlDataOffer>,
    /// The last `ext_data_control_device_v1.selection`.
    control_selection: Option<ext_data_control_offer_v1::ExtDataControlOfferV1>,
    /// `send` events of this client's sources.
    sends: Vec<(String, OwnedFd)>,
    synced: u32,
}

struct TestClient {
    id: ClientId,
    client: Client,
    connection: Connection,
    queue: EventQueue<App>,
    app: App,
}

impl TestClient {
    fn qh(&self) -> QueueHandle<App> {
        self.queue.handle()
    }
}

struct Fixture {
    display: Display<State>,
    state: State,
    seat: Seat<State>,
    clients: Vec<TestClient>,
}

impl Fixture {
    fn new() -> Self {
        let display = Display::<State>::new().unwrap();
        let dh = display.handle();
        let mut seat_state = SeatState::new();
        let seat = seat_state.new_wl_seat(&dh, "seat");
        let state = State {
            compositor: CompositorState::new::<State>(&dh),
            seat_state,
            data_device: DataDeviceState::new::<State>(&dh),
            data_control: DataControlState::new::<State, _>(&dh, None, |_| true),
            seen: Vec::new(),
            serve: Serve::Forward,
        };
        Self {
            display,
            state,
            seat,
            clients: Vec::new(),
        }
    }

    fn dh(&self) -> DisplayHandle {
        self.display.handle()
    }

    /// A connected client with the seat and both managers bound.
    fn add_client(&mut self) -> usize {
        let (server, client) = UnixStream::pair().unwrap();
        client.set_nonblocking(true).unwrap();
        let server_client = self
            .display
            .handle()
            .insert_client(server, Arc::new(TestClientData::default()))
            .unwrap();
        let id = server_client.id();
        let connection = Connection::from_socket(client).unwrap();
        let queue = connection.new_event_queue();
        connection.display().get_registry(&queue.handle(), ());
        self.clients.push(TestClient {
            id,
            client: server_client,
            connection,
            queue,
            app: App::default(),
        });
        let index = self.clients.len() - 1;
        self.roundtrip(index);
        self.roundtrip(index);
        index
    }

    fn server_client(&self, index: usize) -> Client {
        self.clients[index].client.clone()
    }

    /// Exchange messages until the server has handled everything client `index` sent.
    fn roundtrip(&mut self, index: usize) {
        let target = self.clients[index].app.synced + 1;
        let qh = self.clients[index].qh();
        self.clients[index].connection.display().sync(&qh, ());
        for _ in 0..100 {
            let client = &mut self.clients[index];
            client.connection.flush().unwrap();
            self.display.dispatch_clients(&mut self.state).unwrap();
            self.display.flush_clients().unwrap();
            let client = &mut self.clients[index];
            if let Some(guard) = client.connection.prepare_read() {
                let _ = guard.read();
            }
            client.queue.dispatch_pending(&mut client.app).unwrap();
            if client.app.synced >= target {
                return;
            }
        }
        panic!("client {index} did not sync");
    }

    fn roundtrip_all(&mut self) {
        for index in 0..self.clients.len() {
            self.roundtrip(index);
        }
    }

    /// Client `owner` sets the clipboard through data control, offering `mime_types`.
    fn set_selection(&mut self, owner: usize, mime_types: &[&str]) {
        let client = &self.clients[owner];
        let qh = client.qh();
        let manager = client.app.data_control_manager.clone().unwrap();
        let device = manager.get_data_device(client.app.seat.as_ref().unwrap(), &qh, ());
        let source = manager.create_data_source(&qh, ());
        for mime_type in mime_types {
            source.offer(mime_type.to_string());
        }
        device.set_selection(Some(&source));
        self.roundtrip(owner);
        self.roundtrip_all();
    }

    /// Client `index` gets a `wl_data_device`, and the clipboard focus.
    fn focus_data_device(&mut self, index: usize) {
        let client = &self.clients[index];
        let qh = client.qh();
        let manager = client.app.data_device_manager.clone().unwrap();
        manager.get_data_device(client.app.seat.as_ref().unwrap(), &qh, ());
        self.roundtrip(index);
        let focus = self.server_client(index);
        set_data_device_focus(&self.dh(), &self.seat, Some(focus));
        self.roundtrip(index);
    }

    /// Client `index` gets a data-control device (it gets the selection without focus).
    fn bind_data_control(&mut self, index: usize) {
        let client = &self.clients[index];
        let qh = client.qh();
        let manager = client.app.data_control_manager.clone().unwrap();
        manager.get_data_device(client.app.seat.as_ref().unwrap(), &qh, ());
        self.roundtrip(index);
    }

    /// Client `reader` makes a read with `receive`; client `owner`, if any, answers the requests
    /// of its sources with `answer`; other requests are closed unanswered. Returns what the reader
    /// got.
    fn read(
        &mut self,
        reader: usize,
        receive: impl FnOnce(&App, OwnedFd),
        owner: Option<(usize, &str)>,
    ) -> String {
        let (from, to) = std::io::pipe().unwrap();
        receive(&self.clients[reader].app, to.into());
        self.roundtrip(reader);
        self.roundtrip_all();
        for (index, client) in self.clients.iter_mut().enumerate() {
            for (_, fd) in client.app.sends.drain(..) {
                match owner {
                    Some((owner, answer)) if owner == index => {
                        File::from(fd).write_all(answer.as_bytes()).unwrap()
                    }
                    _ => drop(fd),
                }
            }
        }
        read_to_end(from)
    }
}

/// Everything until EOF; a test fails instead of hanging when the write end stays open.
fn read_to_end(from: std::io::PipeReader) -> String {
    use rustix::event::{PollFd, PollFlags, Timespec, poll};

    let mut file = File::from(OwnedFd::from(from));
    let mut got = Vec::new();
    let mut chunk = [0; 4096];
    loop {
        let mut fds = [PollFd::new(&file, PollFlags::IN)];
        let timeout = Timespec {
            tv_sec: 5,
            tv_nsec: 0,
        };
        assert!(
            poll(&mut fds, Some(&timeout)).unwrap() > 0,
            "the read did not end within 5 s"
        );
        match file.read(&mut chunk).unwrap() {
            0 => return String::from_utf8(got).unwrap(),
            n => got.extend_from_slice(&chunk[..n]),
        }
    }
}

fn receive_clipboard(mime_type: &'static str) -> impl FnOnce(&App, OwnedFd) {
    move |app, fd| {
        app.selection
            .as_ref()
            .unwrap()
            .receive(mime_type.into(), fd.as_fd())
    }
}

fn receive_control(mime_type: &'static str) -> impl FnOnce(&App, OwnedFd) {
    move |app, fd| {
        app.control_selection
            .as_ref()
            .unwrap()
            .receive(mime_type.into(), fd.as_fd())
    }
}

#[test]
fn a_client_read_reaches_the_handler_and_is_forwarded_by_default() {
    let mut f = Fixture::new();
    let owner = f.add_client();
    let reader = f.add_client();
    f.focus_data_device(reader);
    f.set_selection(owner, &[TEXT]);

    let got = f.read(reader, receive_clipboard(TEXT), Some((owner, "hello")));

    assert_eq!(got, "hello");
    assert_eq!(
        f.state.seen,
        [Seen {
            target: SelectionTarget::Clipboard,
            device_kind: DataDeviceKind::Core,
            reader: f.clients[reader].id.clone(),
            source_client: Some(f.clients[owner].id.clone()),
            compositor_data: None,
            mime_type: TEXT.into(),
            current: true,
        }]
    );
}

#[test]
fn the_handler_can_serve_a_read_itself() {
    let mut f = Fixture::new();
    let owner = f.add_client();
    let reader = f.add_client();
    f.focus_data_device(reader);
    f.set_selection(owner, &[TEXT]);
    f.state.serve = Serve::Write("from the compositor");

    let got = f.read(reader, receive_clipboard(TEXT), None);
    f.roundtrip(owner);

    assert_eq!(got, "from the compositor");
    assert!(f.clients[owner].app.sends.is_empty(), "the source was asked");
}

#[test]
fn a_read_of_an_older_selection_is_not_current() {
    let mut f = Fixture::new();
    let owner = f.add_client();
    let reader = f.add_client();
    f.focus_data_device(reader);
    f.set_selection(owner, &[TEXT]);
    f.state.serve = Serve::Write("data");
    let old = f.clients[reader].app.selection.clone().unwrap();
    f.set_selection(owner, &[TEXT]);

    f.read(reader, move |_, fd| old.receive(TEXT.into(), fd.as_fd()), None);
    f.read(reader, receive_clipboard(TEXT), None);

    let current: Vec<bool> = f.state.seen.iter().map(|seen| seen.current).collect();
    assert_eq!(current, [false, true]);
}

#[test]
fn a_read_of_the_compositor_selection_reaches_send_selection_by_default() {
    let mut f = Fixture::new();
    let reader = f.add_client();
    f.focus_data_device(reader);
    set_data_device_selection(&f.dh(), &f.seat, vec![TEXT.into()], "compositor data");
    f.roundtrip(reader);

    let got = f.read(reader, receive_clipboard(TEXT), None);

    assert_eq!(got, "compositor data");
    assert_eq!(f.state.seen.len(), 1);
    assert_eq!(f.state.seen[0].source_client, None);
    assert_eq!(f.state.seen[0].compositor_data, Some("compositor data"));
    assert!(f.state.seen[0].current);
}

#[test]
fn a_data_control_read_names_its_device() {
    let mut f = Fixture::new();
    let owner = f.add_client();
    let manager = f.add_client();
    f.bind_data_control(manager);
    f.set_selection(owner, &[TEXT]);

    let got = f.read(manager, receive_control(TEXT), Some((owner, "hello")));

    assert_eq!(got, "hello");
    assert_eq!(f.state.seen.len(), 1);
    assert_eq!(f.state.seen[0].device_kind, DataDeviceKind::ExtDataControl);
    assert_eq!(f.state.seen[0].reader, f.clients[manager].id);
}

#[test]
fn a_mime_type_that_was_not_offered_is_refused_before_the_handler() {
    let mut f = Fixture::new();
    let owner = f.add_client();
    let reader = f.add_client();
    f.focus_data_device(reader);
    f.set_selection(owner, &[TEXT]);

    let got = f.read(reader, receive_clipboard("image/png"), Some((owner, "never")));

    assert_eq!(got, "");
    assert!(f.state.seen.is_empty());
}

// Client side.

impl Dispatch<wl_registry::WlRegistry, ()> for App {
    fn event(
        app: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global { name, interface, .. } = event {
            match interface.as_str() {
                "wl_seat" => app.seat = Some(registry.bind(name, 1, qh, ())),
                "wl_data_device_manager" => app.data_device_manager = Some(registry.bind(name, 3, qh, ())),
                "ext_data_control_manager_v1" => {
                    app.data_control_manager = Some(registry.bind(name, 1, qh, ()))
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<wl_callback::WlCallback, ()> for App {
    fn event(
        app: &mut Self,
        _: &wl_callback::WlCallback,
        _: wl_callback::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        app.synced += 1;
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for App {
    fn event(
        _: &mut Self,
        _: &wl_seat::WlSeat,
        _: wl_seat::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_data_device_manager::WlDataDeviceManager, ()> for App {
    fn event(
        _: &mut Self,
        _: &wl_data_device_manager::WlDataDeviceManager,
        _: wl_data_device_manager::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_data_device::WlDataDevice, ()> for App {
    fn event(
        app: &mut Self,
        _: &wl_data_device::WlDataDevice,
        event: wl_data_device::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_data_device::Event::Selection { id } = event {
            app.selection = id;
        }
    }

    event_created_child!(App, wl_data_device::WlDataDevice, [
        wl_data_device::EVT_DATA_OFFER_OPCODE => (wl_data_offer::WlDataOffer, ()),
    ]);
}

impl Dispatch<wl_data_offer::WlDataOffer, ()> for App {
    fn event(
        _: &mut Self,
        _: &wl_data_offer::WlDataOffer,
        _: wl_data_offer::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ext_data_control_manager_v1::ExtDataControlManagerV1, ()> for App {
    fn event(
        _: &mut Self,
        _: &ext_data_control_manager_v1::ExtDataControlManagerV1,
        _: ext_data_control_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ext_data_control_device_v1::ExtDataControlDeviceV1, ()> for App {
    fn event(
        app: &mut Self,
        _: &ext_data_control_device_v1::ExtDataControlDeviceV1,
        event: ext_data_control_device_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let ext_data_control_device_v1::Event::Selection { id } = event {
            app.control_selection = id;
        }
    }

    event_created_child!(App, ext_data_control_device_v1::ExtDataControlDeviceV1, [
        ext_data_control_device_v1::EVT_DATA_OFFER_OPCODE => (ext_data_control_offer_v1::ExtDataControlOfferV1, ()),
    ]);
}

impl Dispatch<ext_data_control_offer_v1::ExtDataControlOfferV1, ()> for App {
    fn event(
        _: &mut Self,
        _: &ext_data_control_offer_v1::ExtDataControlOfferV1,
        _: ext_data_control_offer_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ext_data_control_source_v1::ExtDataControlSourceV1, ()> for App {
    fn event(
        app: &mut Self,
        _: &ext_data_control_source_v1::ExtDataControlSourceV1,
        event: ext_data_control_source_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let ext_data_control_source_v1::Event::Send { mime_type, fd } = event {
            app.sends.push((mime_type, fd));
        }
    }
}
