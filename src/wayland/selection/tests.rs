//! Reads through [`SelectionHandler::receive_selection`] and [`DataDeviceHandler::receive_dnd`],
//! with clients in the same process. A data-control client owns the selection (it needs no
//! keyboard focus to set it); a `wl_data_device` or primary selection client, focused by the
//! compositor, or another data-control client reads it. Drags come from a compositor-side
//! [`Source`] that names an owner client. All transfers are small (below a pipe's capacity), so
//! writing before reading cannot block.

use std::fs::File;
use std::io::{ErrorKind, Read, Write};
use std::os::unix::io::{AsFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::sync::Arc;

use smallvec::smallvec;
use wayland_client::backend::WaylandError;
use wayland_client::protocol::{
    wl_callback, wl_compositor, wl_data_device, wl_data_device_manager, wl_data_offer, wl_registry, wl_seat,
    wl_surface,
};
use wayland_client::{Connection, Dispatch, EventQueue, QueueHandle, event_created_child};
use wayland_protocols::ext::data_control::v1::client::{
    ext_data_control_device_v1, ext_data_control_manager_v1, ext_data_control_offer_v1,
    ext_data_control_source_v1,
};
use wayland_protocols::wp::primary_selection::zv1::client::{
    zwp_primary_selection_device_manager_v1, zwp_primary_selection_device_v1, zwp_primary_selection_offer_v1,
};
use wayland_server::backend::{ClientData, ClientId, DisconnectReason};
use wayland_server::protocol::wl_surface::WlSurface;
use wayland_server::{Client, Display, DisplayHandle};

use super::data_device::{
    DataDeviceHandler, DataDeviceState, DndRead, WaylandDndGrabHandler, set_data_device_focus,
    set_data_device_selection,
};
use super::ext_data_control::{DataControlHandler, DataControlState};
use super::primary_selection::{PrimarySelectionHandler, PrimarySelectionState, set_primary_focus};
use super::{SelectionDeviceKind, SelectionHandler, SelectionRead, SelectionReadSource, SelectionTarget};
use crate::backend::input::{ButtonState, InputTime};
use crate::input::dnd::{DnDGrab, DndAction, DndGrabHandler, Source, SourceMetadata};
use crate::input::pointer::{ButtonEvent, Focus, GrabStartData, MotionEvent};
use crate::input::{Seat, SeatHandler, SeatState};
use crate::utils::{IsAlive, SERIAL_COUNTER};
use crate::wayland::compositor::{CompositorClientState, CompositorHandler, CompositorState};
use crate::wayland::pointer_constraints::PointerConstraintsHandler;

const TEXT: &str = "text/plain;charset=utf-8";
const BTN_LEFT: u32 = 0x110;

/// What the compositor saw of one selection read.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Seen {
    target: SelectionTarget,
    device_kind: SelectionDeviceKind,
    reader: ClientId,
    /// The owner of a client source; `None` for the compositor's selection.
    source_client: Option<ClientId>,
    compositor_data: Option<&'static str>,
    mime_type: String,
    generation: u64,
    current: bool,
}

/// What the compositor saw of one drag-and-drop read.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SeenDnd {
    reader: ClientId,
    source_client: Option<ClientId>,
    mime_type: String,
    dropped: bool,
}

/// How the compositor serves reads.
enum Serve {
    Forward,
    Write(&'static str),
    Refuse,
    /// Set a compositor selection from inside the handler, then forward.
    SetSelectionAndForward,
}

struct State {
    dh: DisplayHandle,
    compositor: CompositorState,
    seat_state: SeatState<State>,
    data_device: DataDeviceState,
    primary_selection: PrimarySelectionState,
    data_control: DataControlState,
    surfaces: Vec<WlSurface>,
    seen: Vec<Seen>,
    seen_dnd: Vec<SeenDnd>,
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
            generation: read.generation,
            current: read.current,
        });
        match self.serve {
            Serve::Forward => read.forward(self, fd, seat),
            Serve::Write(data) => File::from(fd).write_all(data.as_bytes()).unwrap(),
            Serve::Refuse => drop(fd),
            Serve::SetSelectionAndForward => {
                set_data_device_selection(&self.dh.clone(), &seat, vec![TEXT.into()], "compositor data");
                read.forward(self, fd, seat);
            }
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

    fn new_surface(&mut self, surface: &WlSurface) {
        self.surfaces.push(surface.clone());
    }

    fn commit(&mut self, _surface: &WlSurface) {}
}

impl PointerConstraintsHandler for State {}

impl WaylandDndGrabHandler for State {}

impl DndGrabHandler for State {}

impl DataDeviceHandler for State {
    fn data_device_state(&mut self) -> &mut DataDeviceState {
        &mut self.data_device
    }

    fn receive_dnd(&mut self, read: DndRead, fd: OwnedFd, _seat: Seat<Self>) {
        self.seen_dnd.push(SeenDnd {
            reader: read.reader.id(),
            source_client: read.source_client.as_ref().map(|c| c.id()),
            mime_type: read.mime_type.clone(),
            dropped: read.dropped,
        });
        read.forward(fd);
    }
}

impl PrimarySelectionHandler for State {
    fn primary_selection_state(&mut self) -> &mut PrimarySelectionState {
        &mut self.primary_selection
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

/// A compositor-side drag source whose data `owner` provides.
struct DragSource {
    owner: Client,
}

impl IsAlive for DragSource {
    fn alive(&self) -> bool {
        true
    }
}

impl Source for DragSource {
    fn metadata(&self) -> Option<SourceMetadata> {
        Some(SourceMetadata {
            mime_types: vec![TEXT.into()],
            dnd_actions: smallvec![DndAction::Copy],
        })
    }

    fn choose_action(&self, _action: DndAction) {}

    fn send(&self, _mime_type: &str, fd: OwnedFd) {
        File::from(fd).write_all(b"dragged").unwrap();
    }

    fn drop_performed(&self) {}

    fn cancel(&self) {}

    fn finished(&self) {}

    fn owner_client(&self) -> Option<Client> {
        Some(self.owner.clone())
    }
}

/// The client side of a test client.
#[derive(Default)]
struct App {
    seat: Option<wl_seat::WlSeat>,
    compositor: Option<wl_compositor::WlCompositor>,
    data_device_manager: Option<wl_data_device_manager::WlDataDeviceManager>,
    primary_selection_manager:
        Option<zwp_primary_selection_device_manager_v1::ZwpPrimarySelectionDeviceManagerV1>,
    data_control_manager: Option<ext_data_control_manager_v1::ExtDataControlManagerV1>,
    data_control_device: Option<ext_data_control_device_v1::ExtDataControlDeviceV1>,
    /// The last `wl_data_device.selection`.
    selection: Option<wl_data_offer::WlDataOffer>,
    /// The last `zwp_primary_selection_device_v1.selection`.
    primary_selection: Option<zwp_primary_selection_offer_v1::ZwpPrimarySelectionOfferV1>,
    /// The last `ext_data_control_device_v1.selection`.
    control_selection: Option<ext_data_control_offer_v1::ExtDataControlOfferV1>,
    /// The offer of the last `wl_data_device.enter`, with its serial.
    drag: Option<(wl_data_offer::WlDataOffer, u32)>,
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
        let primary_selection = PrimarySelectionState::new::<State>(&dh);
        let data_control = DataControlState::new::<State, _>(&dh, Some(&primary_selection), |_| true);
        let state = State {
            dh: dh.clone(),
            compositor: CompositorState::new::<State>(&dh),
            seat_state,
            data_device: DataDeviceState::new::<State>(&dh),
            primary_selection,
            data_control,
            surfaces: Vec::new(),
            seen: Vec::new(),
            seen_dnd: Vec::new(),
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

    /// A connected client with the seat and the managers bound.
    fn add_client(&mut self) -> usize {
        let (server, client) = UnixStream::pair().unwrap();
        client.set_nonblocking(true).unwrap();
        let server_client = self
            .display
            .handle()
            .insert_client(server, Arc::new(TestClientData::default()))
            .unwrap();
        let connection = Connection::from_socket(client).unwrap();
        let queue = connection.new_event_queue();
        connection.display().get_registry(&queue.handle(), ());
        self.clients.push(TestClient {
            id: server_client.id(),
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

    /// Exchange messages until the server has handled everything client `index` sent.
    fn roundtrip(&mut self, index: usize) {
        let target = self.clients[index].app.synced + 1;
        let qh = self.clients[index].qh();
        self.clients[index].connection.display().sync(&qh, ());
        for _ in 0..100 {
            self.clients[index].connection.flush().unwrap();
            self.display.dispatch_clients(&mut self.state).unwrap();
            self.display.flush_clients().unwrap();
            let client = &mut self.clients[index];
            if let Some(guard) = client.connection.prepare_read() {
                match guard.read() {
                    Ok(_) => {}
                    Err(WaylandError::Io(err)) if err.kind() == ErrorKind::WouldBlock => {}
                    Err(err) => panic!("client {index}: {err}"),
                }
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

    /// The data-control device of client `index`; it gets selections without focus.
    fn data_control_device(&mut self, index: usize) -> ext_data_control_device_v1::ExtDataControlDeviceV1 {
        if self.clients[index].app.data_control_device.is_none() {
            let client = &self.clients[index];
            let qh = client.qh();
            let manager = client.app.data_control_manager.clone().unwrap();
            let device = manager.get_data_device(client.app.seat.as_ref().unwrap(), &qh, ());
            self.clients[index].app.data_control_device = Some(device);
            self.roundtrip(index);
        }
        self.clients[index].app.data_control_device.clone().unwrap()
    }

    /// Client `owner` sets the selection of `target` through data control, offering `mime_types`.
    fn set_selection(
        &mut self,
        owner: usize,
        target: SelectionTarget,
        mime_types: &[&str],
    ) -> ext_data_control_source_v1::ExtDataControlSourceV1 {
        let device = self.data_control_device(owner);
        let client = &self.clients[owner];
        let source = client
            .app
            .data_control_manager
            .as_ref()
            .unwrap()
            .create_data_source(&client.qh(), ());
        for mime_type in mime_types {
            source.offer(mime_type.to_string());
        }
        match target {
            SelectionTarget::Clipboard => device.set_selection(Some(&source)),
            SelectionTarget::Primary => device.set_primary_selection(Some(&source)),
        }
        self.roundtrip(owner);
        self.roundtrip_all();
        source
    }

    /// Client `index` gets a `wl_data_device` and the clipboard focus.
    fn focus_data_device(&mut self, index: usize) {
        let client = &self.clients[index];
        let manager = client.app.data_device_manager.clone().unwrap();
        manager.get_data_device(client.app.seat.as_ref().unwrap(), &client.qh(), ());
        self.roundtrip(index);
        set_data_device_focus(&self.dh(), &self.seat, Some(self.clients[index].client.clone()));
        self.roundtrip(index);
    }

    /// Client `index` gets a primary selection device and the primary selection focus.
    fn focus_primary_device(&mut self, index: usize) {
        let client = &self.clients[index];
        let manager = client.app.primary_selection_manager.clone().unwrap();
        manager.get_device(client.app.seat.as_ref().unwrap(), &client.qh(), ());
        self.roundtrip(index);
        set_primary_focus(&self.dh(), &self.seat, Some(self.clients[index].client.clone()));
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

fn receive_primary(mime_type: &'static str) -> impl FnOnce(&App, OwnedFd) {
    move |app, fd| {
        app.primary_selection
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

fn receive_drag(mime_type: &'static str) -> impl FnOnce(&App, OwnedFd) {
    move |app, fd| app.drag.as_ref().unwrap().0.receive(mime_type.into(), fd.as_fd())
}

/// A clipboard owner and a reader with the clipboard focus, after the owner offered `TEXT`.
fn clipboard() -> (Fixture, usize, usize) {
    let mut f = Fixture::new();
    let owner = f.add_client();
    let reader = f.add_client();
    f.focus_data_device(reader);
    f.set_selection(owner, SelectionTarget::Clipboard, &[TEXT]);
    (f, owner, reader)
}

#[test]
fn a_client_read_reaches_the_handler_and_is_forwarded_by_default() {
    let (mut f, owner, reader) = clipboard();

    let got = f.read(reader, receive_clipboard(TEXT), Some((owner, "hello")));

    assert_eq!(got, "hello");
    assert_eq!(
        f.state.seen,
        [Seen {
            target: SelectionTarget::Clipboard,
            device_kind: SelectionDeviceKind::DataDevice,
            reader: f.clients[reader].id.clone(),
            source_client: Some(f.clients[owner].id.clone()),
            compositor_data: None,
            mime_type: TEXT.into(),
            generation: 1,
            current: true,
        }]
    );
}

#[test]
fn the_handler_can_serve_a_read_itself() {
    let (mut f, owner, reader) = clipboard();
    f.state.serve = Serve::Write("from the compositor");

    let got = f.read(reader, receive_clipboard(TEXT), Some((owner, "never")));

    assert_eq!(got, "from the compositor");
}

#[test]
fn the_handler_can_refuse_a_read() {
    let (mut f, owner, reader) = clipboard();
    f.state.serve = Serve::Refuse;

    let got = f.read(reader, receive_clipboard(TEXT), Some((owner, "never")));

    assert_eq!(got, "");
    assert_eq!(f.state.seen.len(), 1);
}

#[test]
fn the_handler_can_change_the_selection_while_it_runs() {
    let (mut f, owner, reader) = clipboard();
    f.state.serve = Serve::SetSelectionAndForward;

    let got = f.read(reader, receive_clipboard(TEXT), Some((owner, "hello")));

    assert_eq!(got, "hello");
}

#[test]
fn a_read_of_an_older_selection_is_not_current() {
    let (mut f, owner, reader) = clipboard();
    f.state.serve = Serve::Write("data");
    let old = f.clients[reader].app.selection.clone().unwrap();
    f.set_selection(owner, SelectionTarget::Clipboard, &[TEXT]);

    f.read(reader, move |_, fd| old.receive(TEXT.into(), fd.as_fd()), None);
    f.read(reader, receive_clipboard(TEXT), None);

    let seen: Vec<(u64, bool)> = f.state.seen.iter().map(|s| (s.generation, s.current)).collect();
    assert_eq!(seen, [(1, false), (2, true)]);
}

#[test]
fn a_read_after_the_source_is_gone_is_not_current() {
    let (mut f, owner, reader) = clipboard();
    let source = f.set_selection(owner, SelectionTarget::Clipboard, &[TEXT]);
    let offer = f.clients[reader].app.selection.clone().unwrap();
    source.destroy();
    f.roundtrip(owner);

    let got = f.read(reader, move |_, fd| offer.receive(TEXT.into(), fd.as_fd()), None);

    assert_eq!(got, "");
    assert_eq!(f.state.seen.len(), 1);
    assert!(!f.state.seen[0].current);
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
fn a_primary_selection_read_counts_its_own_changes() {
    let mut f = Fixture::new();
    let owner = f.add_client();
    let reader = f.add_client();
    f.focus_primary_device(reader);
    f.set_selection(owner, SelectionTarget::Primary, &[TEXT]);
    // The clipboard changes, the primary selection does not.
    f.set_selection(owner, SelectionTarget::Clipboard, &[TEXT]);
    f.set_selection(owner, SelectionTarget::Clipboard, &[TEXT]);

    let got = f.read(reader, receive_primary(TEXT), Some((owner, "primary")));

    assert_eq!(got, "primary");
    assert_eq!(f.state.seen.len(), 1);
    let seen = &f.state.seen[0];
    assert_eq!(seen.target, SelectionTarget::Primary);
    assert_eq!(seen.device_kind, SelectionDeviceKind::PrimarySelection);
    assert_eq!((seen.generation, seen.current), (1, true));
}

#[test]
fn a_data_control_read_names_its_device() {
    let mut f = Fixture::new();
    let owner = f.add_client();
    let manager = f.add_client();
    f.data_control_device(manager);
    f.set_selection(owner, SelectionTarget::Clipboard, &[TEXT]);

    let got = f.read(manager, receive_control(TEXT), Some((owner, "hello")));

    assert_eq!(got, "hello");
    assert_eq!(f.state.seen.len(), 1);
    assert_eq!(f.state.seen[0].device_kind, SelectionDeviceKind::ExtDataControl);
    assert_eq!(f.state.seen[0].reader, f.clients[manager].id);
}

#[test]
fn a_mime_type_that_was_not_offered_is_refused_before_the_handler() {
    let (mut f, owner, reader) = clipboard();

    let got = f.read(reader, receive_clipboard("image/png"), Some((owner, "never")));

    assert_eq!(got, "");
    assert!(f.state.seen.is_empty());
}

#[test]
fn a_mime_type_added_after_the_offer_is_refused_before_the_handler() {
    let (mut f, owner, reader) = clipboard();
    let source = f.set_selection(owner, SelectionTarget::Clipboard, &[TEXT]);
    source.offer("text/uri-list".into());
    f.roundtrip(owner);

    let got = f.read(reader, receive_clipboard("text/uri-list"), Some((owner, "never")));

    assert_eq!(got, "");
    assert!(f.state.seen.is_empty());
}

#[test]
fn drag_reads_reach_the_handler_before_and_after_the_drop() {
    let mut f = Fixture::new();
    let owner = f.add_client();
    let reader = f.add_client();
    f.focus_data_device(reader);
    let client = &f.clients[reader];
    client
        .app
        .compositor
        .as_ref()
        .unwrap()
        .create_surface(&client.qh(), ());
    f.roundtrip(reader);
    let surface = f.state.surfaces.last().unwrap().clone();

    // The compositor drags data that `owner` provides onto the reader's surface.
    let pointer = f.seat.add_pointer();
    let start = GrabStartData {
        focus: None,
        button: BTN_LEFT,
        location: (0.0, 0.0).into(),
    };
    let source = DragSource {
        owner: f.clients[owner].client.clone(),
    };
    let grab: DnDGrab<State, DragSource, WlSurface> =
        DnDGrab::new_pointer(&f.dh(), start, source, f.seat.clone());
    pointer.set_grab(&mut f.state, grab, SERIAL_COUNTER.next_serial(), Focus::Keep);
    let motion = MotionEvent {
        location: (10.0, 10.0).into(),
        serial: SERIAL_COUNTER.next_serial(),
        time: InputTime::from_millis(0),
    };
    pointer.motion(&mut f.state, Some((surface, (0.0, 0.0).into())), &motion);
    pointer.frame(&mut f.state);
    f.roundtrip(reader);
    let (offer, serial) = f.clients[reader].app.drag.clone().expect("no drag offer");
    offer.accept(serial, Some(TEXT.into()));
    offer.set_actions(
        wl_data_device_manager::DndAction::Copy,
        wl_data_device_manager::DndAction::Copy,
    );
    f.roundtrip(reader);

    let before = f.read(reader, receive_drag(TEXT), None);
    let release = ButtonEvent {
        serial: SERIAL_COUNTER.next_serial(),
        time: InputTime::from_millis(0),
        button: BTN_LEFT,
        state: ButtonState::Released,
    };
    pointer.button(&mut f.state, &release);
    pointer.frame(&mut f.state);
    f.roundtrip(reader);
    let after = f.read(reader, receive_drag(TEXT), None);

    assert_eq!((before.as_str(), after.as_str()), ("dragged", "dragged"));
    let seen = |dropped| SeenDnd {
        reader: f.clients[reader].id.clone(),
        source_client: Some(f.clients[owner].id.clone()),
        mime_type: TEXT.into(),
        dropped,
    };
    assert_eq!(f.state.seen_dnd, [seen(false), seen(true)]);
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
                "wl_compositor" => app.compositor = Some(registry.bind(name, 1, qh, ())),
                "wl_data_device_manager" => app.data_device_manager = Some(registry.bind(name, 3, qh, ())),
                "zwp_primary_selection_device_manager_v1" => {
                    app.primary_selection_manager = Some(registry.bind(name, 1, qh, ()))
                }
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

impl Dispatch<wl_data_device::WlDataDevice, ()> for App {
    fn event(
        app: &mut Self,
        _: &wl_data_device::WlDataDevice,
        event: wl_data_device::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wl_data_device::Event::Selection { id } => app.selection = id,
            wl_data_device::Event::Enter { serial, id, .. } => app.drag = id.map(|offer| (offer, serial)),
            _ => {}
        }
    }

    event_created_child!(App, wl_data_device::WlDataDevice, [
        wl_data_device::EVT_DATA_OFFER_OPCODE => (wl_data_offer::WlDataOffer, ()),
    ]);
}

impl Dispatch<zwp_primary_selection_device_v1::ZwpPrimarySelectionDeviceV1, ()> for App {
    fn event(
        app: &mut Self,
        _: &zwp_primary_selection_device_v1::ZwpPrimarySelectionDeviceV1,
        event: zwp_primary_selection_device_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let zwp_primary_selection_device_v1::Event::Selection { id } = event {
            app.primary_selection = id;
        }
    }

    event_created_child!(App, zwp_primary_selection_device_v1::ZwpPrimarySelectionDeviceV1, [
        zwp_primary_selection_device_v1::EVT_DATA_OFFER_OPCODE => (zwp_primary_selection_offer_v1::ZwpPrimarySelectionOfferV1, ()),
    ]);
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

/// Objects whose events these tests ignore.
macro_rules! ignore_events {
    ($($proxy:ty),* $(,)?) => {
        $(
            impl Dispatch<$proxy, ()> for App {
                fn event(
                    _: &mut Self,
                    _: &$proxy,
                    _: <$proxy as wayland_client::Proxy>::Event,
                    _: &(),
                    _: &Connection,
                    _: &QueueHandle<Self>,
                ) {
                }
            }
        )*
    };
}

ignore_events!(
    wl_seat::WlSeat,
    wl_compositor::WlCompositor,
    wl_surface::WlSurface,
    wl_data_device_manager::WlDataDeviceManager,
    wl_data_offer::WlDataOffer,
    zwp_primary_selection_device_manager_v1::ZwpPrimarySelectionDeviceManagerV1,
    zwp_primary_selection_offer_v1::ZwpPrimarySelectionOfferV1,
    ext_data_control_manager_v1::ExtDataControlManagerV1,
    ext_data_control_offer_v1::ExtDataControlOfferV1,
);
