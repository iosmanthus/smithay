use std::mem;
use std::sync::{Arc, Mutex};

use tracing::debug;
use wayland_protocols::wp::text_input::zv3::server::zwp_text_input_v3::{
    self, ChangeCause, ContentHint, ContentPurpose, ZwpTextInputV3,
};
use wayland_server::backend::{ClientId, ObjectId};
use wayland_server::{Resource, protocol::wl_surface::WlSurface};

use crate::input::SeatHandler;
use crate::utils::{Logical, Rectangle};
use crate::wayland::{Dispatch2, input_method::InputMethodHandle};

#[derive(Default, Debug)]
pub(crate) struct TextInput {
    instances: Vec<Instance>,
    focus: Option<WlSurface>,
    active_text_input_id: Option<ObjectId>,
    compositor_input_method: bool,
    /// Events from the input method, held until it commits them (see [`TextInputHandle::done`]).
    pending_input_method_events: InputMethodEvents,
}

/// What the input method asked the active text-input to do since its last commit.
#[derive(Default, Debug)]
struct InputMethodEvents {
    preedit: Option<(String, i32, i32)>,
    commit: Option<String>,
    delete_surrounding_text: Option<(u32, u32)>,
}

impl TextInput {
    fn active_instance(&mut self) -> Option<&mut Instance> {
        let active_id = self.active_text_input_id.as_ref()?;
        let surface_id = self.focus.as_ref().filter(|surface| surface.is_alive())?.id();
        self.instances
            .iter_mut()
            .filter(|instance| instance.instance.id().same_client_as(&surface_id))
            .find(|instance| &instance.instance.id() == active_id)
    }

    fn with_focused_client_all_text_inputs<F>(&mut self, mut f: F)
    where
        F: FnMut(&ZwpTextInputV3, &WlSurface, u32),
    {
        if let Some(surface) = self.focus.as_ref().filter(|surface| surface.is_alive()) {
            for text_input in self.instances.iter() {
                let instance_id = text_input.instance.id();
                if instance_id.same_client_as(&surface.id()) {
                    f(&text_input.instance, surface, text_input.serial);
                }
            }
        };
    }

    fn with_active_text_input<F>(&mut self, mut f: F)
    where
        F: FnMut(&ZwpTextInputV3, &WlSurface, u32),
    {
        let active_id = match &self.active_text_input_id {
            Some(active_text_input_id) => active_text_input_id,
            None => return,
        };

        let surface = match self.focus.as_ref().filter(|surface| surface.is_alive()) {
            Some(surface) => surface,
            None => return,
        };

        let surface_id = surface.id();
        if let Some(text_input) = self
            .instances
            .iter()
            .filter(|instance| instance.instance.id().same_client_as(&surface_id))
            .find(|instance| &instance.instance.id() == active_id)
        {
            f(&text_input.instance, surface, text_input.serial);
        }
    }
}

/// Handle to text input instances
#[derive(Default, Debug, Clone)]
pub struct TextInputHandle {
    pub(crate) inner: Arc<Mutex<TextInput>>,
}

impl TextInputHandle {
    pub(super) fn add_instance(&self, instance: &ZwpTextInputV3) {
        let mut inner = self.inner.lock().unwrap();
        inner.instances.push(Instance {
            instance: instance.clone(),
            serial: 0,
            pending_state: Default::default(),
            preedit: None,
        });
    }

    fn increment_serial(&self, text_input: &ZwpTextInputV3) {
        if let Some(instance) = self
            .inner
            .lock()
            .unwrap()
            .instances
            .iter_mut()
            .find(|instance| instance.instance == *text_input)
        {
            instance.serial += 1
        }
    }

    /// Return the currently focused surface.
    pub fn focus(&self) -> Option<WlSurface> {
        self.inner.lock().unwrap().focus.clone()
    }

    /// Advance the focus for the client to `surface`.
    ///
    /// This doesn't send any 'enter' or 'leave' events.
    pub fn set_focus(&self, surface: Option<WlSurface>) {
        self.inner.lock().unwrap().focus = surface;
    }

    /// Send `leave` on the text-input instance for the currently focused
    /// surface.
    pub fn leave(&self) {
        let mut inner = self.inner.lock().unwrap();
        // Leaving clears the active text input.
        inner.active_text_input_id = None;
        // NOTE: we implement it in a symmetrical way with `enter`.
        inner.with_focused_client_all_text_inputs(|text_input, focus, _| {
            text_input.leave(focus);
        });
    }

    /// Send `enter` on the text-input instance for the currently focused
    /// surface.
    pub fn enter(&self) {
        let mut inner = self.inner.lock().unwrap();
        // NOTE: protocol states that if we have multiple text inputs enabled, `enter` must
        // be send for each of them.
        inner.with_focused_client_all_text_inputs(|text_input, focus, _| {
            text_input.enter(focus);
        });
    }

    /// Have the compositor act as the input method for this seat.
    ///
    /// While enabled, `enter` is delivered to the focused text-input even when no real
    /// `zwp_input_method_v2` is bound, so the compositor can `commit_string` into it (e.g. for
    /// remote-desktop text injection). Toggling this sends `enter`/`leave` for the current
    /// focus immediately; subsequent focus changes are handled by the keyboard focus logic.
    pub fn set_compositor_input_method(&self, active: bool) {
        {
            let mut inner = self.inner.lock().unwrap();
            if inner.compositor_input_method == active {
                return;
            }
            inner.compositor_input_method = active;
        }
        if active {
            self.enter();
        } else {
            self.leave();
        }
    }

    /// Whether the compositor is currently acting as the input method for this seat
    /// (see [`set_compositor_input_method`](Self::set_compositor_input_method)).
    pub fn compositor_input_method(&self) -> bool {
        self.inner.lock().unwrap().compositor_input_method
    }

    /// Hold a preedit from the input method until it commits.
    pub(crate) fn queue_preedit_string(&self, text: String, cursor_begin: i32, cursor_end: i32) {
        self.inner.lock().unwrap().pending_input_method_events.preedit =
            Some((text, cursor_begin, cursor_end));
    }

    /// Hold text the input method commits until it commits its state.
    pub(crate) fn queue_commit_string(&self, text: String) {
        self.inner.lock().unwrap().pending_input_method_events.commit = Some(text);
    }

    /// Hold a deletion from the input method until it commits.
    pub(crate) fn queue_delete_surrounding_text(&self, before_length: u32, after_length: u32) {
        self.inner
            .lock()
            .unwrap()
            .pending_input_method_events
            .delete_surrounding_text = Some((before_length, after_length));
    }

    /// Send the input method's held events to the active text-input, followed by `done`.
    ///
    /// The events are held rather than forwarded as they arrive because `done` is also sent after
    /// every commit from the text-input (see [`Self::acknowledge_commit`]): forwarded early, an
    /// input method preedit would be applied, and then cleared, by that acknowledgement.
    ///
    /// The `discard_state` is used when the input-method signaled that
    /// the state should be discarded and wrong serial sent.
    pub fn done(&self, discard_state: bool) {
        let mut inner = self.inner.lock().unwrap();
        let events = mem::take(&mut inner.pending_input_method_events);
        let Some(instance) = inner.active_instance() else {
            return;
        };
        if let Some((text, cursor_begin, cursor_end)) = &events.preedit {
            instance
                .instance
                .preedit_string(Some(text.clone()), *cursor_begin, *cursor_end);
        }
        if let Some(text) = events.commit {
            instance.instance.commit_string(Some(text));
        }
        if let Some((before_length, after_length)) = events.delete_surrounding_text {
            instance
                .instance
                .delete_surrounding_text(before_length, after_length);
        }
        // `done` clears a preedit that was not sent before it.
        instance.preedit = events.preedit;
        if discard_state {
            debug!("discarding text-input state due to serial");
            // Discarding is done by sending non-matching serial.
            instance.instance.done(0);
        } else {
            instance.instance.done(instance.serial);
        }
    }

    /// Answer a commit from `text_input` with `done`, re-sending the preedit it shows so the
    /// `done` does not clear it.
    ///
    /// Clients count their commits and expect a `done` carrying that count: Chromium holds back
    /// its cursor rectangle until the last commit is acknowledged, so without this the input
    /// method's popup stayed where the previous text field was until the first key produced a
    /// `done`. KWin acknowledges every commit the same way, which GTK relies on too.
    fn acknowledge_commit(&self, text_input: &ZwpTextInputV3) {
        let inner = self.inner.lock().unwrap();
        let Some(instance) = inner
            .instances
            .iter()
            .find(|instance| instance.instance == *text_input)
        else {
            return;
        };
        if let Some((text, cursor_begin, cursor_end)) = &instance.preedit {
            text_input.preedit_string(Some(text.clone()), *cursor_begin, *cursor_end);
        }
        text_input.done(instance.serial);
    }

    /// Access the text-input instances for the currently focused surface.
    pub fn with_focused_text_input<F>(&self, mut f: F)
    where
        F: FnMut(&ZwpTextInputV3, &WlSurface),
    {
        let mut inner = self.inner.lock().unwrap();
        inner.with_focused_client_all_text_inputs(|ti, surface, _| {
            f(ti, surface);
        });
    }

    /// Access the active text-input instance for the currently focused surface.
    pub fn with_active_text_input<F>(&self, mut f: F)
    where
        F: FnMut(&ZwpTextInputV3, &WlSurface),
    {
        let mut inner = self.inner.lock().unwrap();
        inner.with_active_text_input(|ti, surface, _| {
            f(ti, surface);
        });
    }

    /// Call the callback with the serial of the active text_input or with the passed
    /// `default` one when empty.
    pub(crate) fn active_text_input_serial_or_default<F>(&self, default: u32, mut callback: F)
    where
        F: FnMut(u32),
    {
        let mut inner = self.inner.lock().unwrap();
        let mut should_default = true;
        inner.with_active_text_input(|_, _, serial| {
            should_default = false;
            callback(serial);
        });
        if should_default {
            callback(default)
        }
    }
}

/// User data of ZwpTextInputV3 object
#[derive(Debug)]
pub struct TextInputUserData {
    pub(super) handle: TextInputHandle,
    pub(crate) input_method_handle: InputMethodHandle,
}

impl<D> Dispatch2<ZwpTextInputV3, D> for TextInputUserData
where
    D: SeatHandler,
    D: 'static,
{
    fn request(
        &self,
        state: &mut D,
        _client: &wayland_server::Client,
        resource: &ZwpTextInputV3,
        request: zwp_text_input_v3::Request,
        _dhandle: &wayland_server::DisplayHandle,
        _data_init: &mut wayland_server::DataInit<'_, D>,
    ) {
        let commit = matches!(request, zwp_text_input_v3::Request::Commit);
        // Always increment serial to not desync with clients.
        if commit {
            self.handle.increment_serial(resource);
        }
        self.process_request(state, resource, request);
        if commit {
            self.handle.acknowledge_commit(resource);
        }
    }

    fn destroyed(&self, state: &mut D, _client: ClientId, text_input: &ZwpTextInputV3) {
        let destroyed_id = text_input.id();
        let deactivate_im = {
            let mut inner = self.handle.inner.lock().unwrap();
            inner.instances.retain(|inst| inst.instance.id() != destroyed_id);
            let destroyed_focused = inner
                .focus
                .as_ref()
                .map(|focus| focus.id().same_client_as(&destroyed_id))
                .unwrap_or(true);

            // Deactivate IM when we either lost focus entirely or destroyed text-input for the
            // currently focused client.
            destroyed_focused
                && !inner
                    .instances
                    .iter()
                    .any(|inst| inst.instance.id().same_client_as(&destroyed_id))
        };

        if deactivate_im {
            self.input_method_handle.deactivate_input_method(state);
        }
    }
}

impl TextInputUserData {
    fn process_request<D: SeatHandler + 'static>(
        &self,
        state: &mut D,
        resource: &ZwpTextInputV3,
        request: zwp_text_input_v3::Request,
    ) {
        // Discard requests without any active input method instance, unless the compositor
        // itself is acting as the input method for this seat (the text-injection escape hatch).
        // In that case we still process enable/commit so the focused client becomes the active
        // text input and the compositor can `commit_string` into it.
        if !self.input_method_handle.has_instance() && !self.handle.compositor_input_method() {
            debug!("discarding text-input request without IME running");
            return;
        }

        let focus = match self.handle.focus() {
            Some(focus) if focus.id().same_client_as(&resource.id()) => focus,
            _ => {
                debug!("discarding text-input request for unfocused client");
                return;
            }
        };

        let mut guard = self.handle.inner.lock().unwrap();
        let pending_state = match guard.instances.iter_mut().find_map(|instance| {
            if instance.instance == *resource {
                Some(&mut instance.pending_state)
            } else {
                None
            }
        }) {
            Some(pending_state) => pending_state,
            None => {
                debug!("got request for untracked text-input");
                return;
            }
        };

        match request {
            zwp_text_input_v3::Request::Enable => {
                pending_state.enable = Some(true);
            }
            zwp_text_input_v3::Request::Disable => {
                pending_state.enable = Some(false);
            }
            zwp_text_input_v3::Request::SetSurroundingText { text, cursor, anchor } => {
                pending_state.surrounding_text = Some((text, cursor as u32, anchor as u32));
            }
            zwp_text_input_v3::Request::SetTextChangeCause { cause } => {
                // Guard against clients sending us unknown values from future versions.
                let cause = cause.into_result().unwrap_or(ChangeCause::Other);
                pending_state.text_change_cause = Some(cause);
            }
            zwp_text_input_v3::Request::SetContentType { hint, purpose } => {
                // Guard against clients sending us unknown values from future versions.
                let hint = ContentHint::from_bits_truncate(u32::from(hint));
                let purpose = purpose.into_result().unwrap_or(ContentPurpose::Normal);
                pending_state.content_type = Some((hint, purpose));
            }
            zwp_text_input_v3::Request::SetCursorRectangle { x, y, width, height } => {
                pending_state.cursor_rectangle = Some(Rectangle::new((x, y).into(), (width, height).into()));
            }
            zwp_text_input_v3::Request::Commit => {
                let mut new_state = mem::take(pending_state);
                let _ = pending_state;
                let active_text_input_id = &mut guard.active_text_input_id;

                if active_text_input_id.is_some() && *active_text_input_id != Some(resource.id()) {
                    debug!("discarding text_input request since we already have an active one");
                    return;
                }

                match new_state.enable {
                    Some(true) => {
                        *active_text_input_id = Some(resource.id());
                        // Enabling resets the client's state, preedit included.
                        if let Some(instance) = guard.instances.iter_mut().find(|i| i.instance == *resource) {
                            instance.preedit = None;
                        }
                        // Drop the guard before calling to other subsystem.
                        drop(guard);
                        self.input_method_handle.activate_input_method(state, &focus);
                    }
                    Some(false) => {
                        *active_text_input_id = None;
                        if let Some(instance) = guard.instances.iter_mut().find(|i| i.instance == *resource) {
                            instance.preedit = None;
                        }
                        // Drop the guard before calling to other subsystem.
                        drop(guard);
                        self.input_method_handle.deactivate_input_method(state);
                        return;
                    }
                    None => {
                        if *active_text_input_id != Some(resource.id()) {
                            debug!("discarding text_input requests before enabling it");
                            return;
                        }

                        // Drop the guard before calling to other subsystems later on.
                        drop(guard);
                    }
                }

                if let Some((text, cursor, anchor)) = new_state.surrounding_text.take() {
                    self.input_method_handle.with_instance(move |input_method| {
                        input_method.object.surrounding_text(text, cursor, anchor)
                    });
                }

                if let Some(cause) = new_state.text_change_cause.take() {
                    self.input_method_handle.with_instance(move |input_method| {
                        input_method.object.text_change_cause(cause);
                    });
                }

                if let Some((hint, purpose)) = new_state.content_type.take() {
                    self.input_method_handle.with_instance(move |input_method| {
                        input_method.object.content_type(hint, purpose);
                    });
                }

                if let Some(rect) = new_state.cursor_rectangle.take() {
                    self.input_method_handle
                        .set_text_input_rectangle::<D>(state, rect);
                }

                self.input_method_handle.with_instance(|input_method| {
                    input_method.done();
                });
            }
            zwp_text_input_v3::Request::Destroy => {
                // Nothing to do
            }
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
struct Instance {
    instance: ZwpTextInputV3,
    serial: u32,
    pending_state: TextInputState,
    /// The preedit the client shows: the last one sent before a `done`.
    preedit: Option<(String, i32, i32)>,
}

#[derive(Debug, Default)]
struct TextInputState {
    enable: Option<bool>,
    surrounding_text: Option<(String, u32, u32)>,
    content_type: Option<(ContentHint, ContentPurpose)>,
    cursor_rectangle: Option<Rectangle<i32, Logical>>,
    text_change_cause: Option<ChangeCause>,
}
