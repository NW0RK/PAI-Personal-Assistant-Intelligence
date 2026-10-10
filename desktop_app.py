"""Desktop display. Tk stays on the main thread; supervision runs off-thread."""
import copy
import ctypes
import os
from pathlib import Path
import queue
import shutil
import threading
import time
import tkinter as tk
import uuid
from tkinter import ttk, filedialog

from app_settings import (load_preferences, save_preferences, validate_preferences,
                          discover_microphones, load_appearance, save_appearance, POSITIONS)
from launch_gemma import Supervisor, memory_status
from ui_theme import COLORS, SPACE, install_theme
from desktop_helpers import Calibration, clamp_geometry, work_areas, markdown_conversation


def apply_preferences(args, preferences):
    args.device = preferences['device']
    args.silence = preferences['silence']
    args.threshold = preferences['threshold']
    args.no_tts = not preferences['voice']
    args.no_overlay = not preferences['captions']


def initial_preferences(args):
    values, warning = load_preferences(args.settings_file)
    overrides = dict(device=args.device, silence=args.silence, threshold=args.threshold,
                     voice=not args.no_tts, captions=not args.no_overlay)
    for key in args.preference_overrides:
        values[key] = overrides[key]
    return validate_preferences(values), warning


class Conversation:
    """Display history; the assistant alone owns model context and media."""
    def __init__(self):
        self.turns = {}
        self.revision = 0

    def handle(self, event):
        kind = event.get('event')
        if kind == 'cleared':
            self.turns.clear()
        elif kind == 'assistant_stopped':
            for turn in self.turns.values():
                turn['ended'] = True
        elif kind == 'turn_started':
            self.turns[event['turn_id']] = dict(text='', complete=False, ended=False,
                                              screen=event.get('screen_sharing', False),
                                              input_kind=event.get('input_kind', 'voice'), input_text=event.get('input_text', ''))
            if len(self.turns) > 100:
                del self.turns[next(iter(self.turns))]
        elif kind in ('partial', 'reply', 'turn_finished'):
            turn = self.turns.get(event.get('turn_id'))
            if turn is None or (turn['ended'] and kind != 'turn_finished'):
                return
            turn['text'] = event.get('text', '')
            if kind != 'partial':
                turn['complete'] = event.get('complete', False)
            if kind == 'turn_finished':
                turn['ended'] = True
        else:
            return
        self.revision += 1


class SupervisorWorker:
    def __init__(self, args, preferences, *, supervisor_factory=Supervisor):
        self.args = copy.copy(args)
        apply_preferences(self.args, preferences)
        self.args.screen_sharing = False
        self.supervisor = supervisor_factory(self.args)
        self.commands = queue.Queue()
        self.output = queue.Queue()
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self.run, name='Gemma supervisor', daemon=False)
        self.pending_preferences = None
        self.preferences_request = None
        self.calibration = None
        self.calibration_request = None
        self.calibration_events = queue.Queue()
        self.calibration_terminal = None
        self.restore_listening = False

    def start(self):
        self.thread.start()

    def send(self, action, **fields):
        self.commands.put(dict(action=action, **fields))

    def handle(self, command):
        action = command['action']
        if (self.calibration or self.calibration_request) and action not in ('cancel_calibration', 'appearance', 'screen_sharing'):
            raise ValueError('Finish or cancel microphone calibration first.')
        if action == 'start':
            self.args.text_only = command.get('mode', 'voice') == 'text'
            self.supervisor.start()
        elif action == 'restart':
            self.supervisor.restart()
            self.output.put(dict(event='cleared'))
        elif action == 'preferences':
            if self.pending_preferences is not None:
                raise ValueError('Settings are still being applied.')
            values = validate_preferences(command['preferences'])
            self.preferences_request = command.get('request_id') or uuid.uuid4().hex
            if self.supervisor.phase == 'Running':
                self.pending_preferences = values
                self.preferences_deadline = time.monotonic() + 12
                if self.supervisor.send(dict(action='configure', preferences=values, request_id=self.preferences_request)) is False:
                    self.pending_preferences = None
                    raise ValueError('The assistant disconnected. Your edits are kept; reconnect before applying changes.')
            else:
                self.commit_preferences(values)
        elif action == 'appearance':
            appearance = command['appearance']
            self.args.caption_position = appearance['caption_position']
            self.supervisor.send(dict(action='caption_position', position=self.args.caption_position))
            try:
                save_appearance(appearance, self.args.settings_file)
            except OSError as exc:
                self.output.put(dict(event='failure', kind='appearance', code='settings_save_failed', detail=f'Window preferences could not be saved: {exc}'))
        elif action == 'send_text':
            if self.supervisor.phase != 'Running' or self.supervisor.server_state != 'Ready' or self.supervisor.state not in ('Listening', 'Paused', 'Error'):
                self.output.put(dict(event='failure', kind='input', code='input_unavailable', request_id=command.get('request_id'), detail='The assistant is busy or disconnected. Your message has not been sent.'))
            else:
                if self.supervisor.send(command) is False:
                    self.output.put(dict(event='failure', kind='input', code='input_unavailable', request_id=command.get('request_id'), detail='The assistant disconnected. Your message has not been sent.'))
        elif action == 'calibrate':
            if self.supervisor.phase == 'Loading' or self.supervisor.state not in ('Stopped', 'Listening', 'Paused', 'Error'):
                raise ValueError('Stop the current reply before testing your microphone.')
            self.calibration_request = command['device']
            self.restore_listening = self.supervisor.phase == 'Running' and self.supervisor.state == 'Listening'
            self.calibration_deadline = time.monotonic() + 3
            if self.supervisor.phase == 'Running' and self.supervisor.state != 'Paused':
                self.supervisor.send('pause')
        elif action == 'cancel_calibration':
            if self.calibration:
                self.calibration.cancelled.set()
            elif self.calibration_request:
                self.calibration_request = None
                self.output.put(dict(event='calibration', stage='cancelled'))
                if self.restore_listening:
                    self.supervisor.send('resume')
        elif action == 'advanced':
            if self.supervisor.phase not in ('Stopped', 'Error') or (
                    self.supervisor.assistant and self.supervisor.assistant.poll() is None):
                raise ValueError('Server settings can be changed when the assistant is stopped.')
            if self.supervisor.server and self.supervisor.server.poll() is None:
                raise ValueError('Quit and reopen the app with the new server settings to reload the owned server.')
            values = command['values']
            if not 1 <= values['port'] <= 65535 or values['ctx'] <= 0 or values['gpu_layers'] < -1:
                raise ValueError('Enter a valid port, positive context size, and GPU layer count of -1 or greater.')
            for key in ('server', 'model', 'mmproj', 'mtp', 'port', 'ctx', 'gpu_layers', 'no_mtp'):
                setattr(self.args, key, values[key])
            self.supervisor.url = f'http://127.0.0.1:{self.args.port}'
            self.output.put(dict(event='advanced_applied', values={key: str(value) for key, value in values.items()}, detail='Server settings applied for this session.'))
        elif action == 'screen_sharing':
            self.args.screen_sharing = command['enabled']
            self.supervisor.send(command)
        elif action == 'clear':
            if self.supervisor.phase == 'Running':
                self.supervisor.send('clear')
            else:
                self.output.put(dict(event='cleared'))
        elif action in ('pause', 'resume', 'cancel'):
            self.supervisor.send(action)

    def commit_preferences(self, values):
        apply_preferences(self.args, values)
        warning = ''
        try:
            save_preferences(values, self.args.settings_file)
        except OSError as exc:
            warning = f'Changes apply for this session but could not be saved: {exc}'
        self.pending_preferences = None
        self.output.put(dict(event='preferences_applied', preferences=values, request_id=self.preferences_request, warning=warning))

    def tick_calibration(self):
        if self.calibration_request:
            if self.supervisor.phase != 'Running' or self.supervisor.state == 'Paused':
                self.calibration = Calibration(self.calibration_request, self.calibration_events)
                self.calibration_request = None
                self.calibration.start()
            elif time.monotonic() >= self.calibration_deadline:
                self.calibration_request = None
                self.output.put(dict(event='calibration', stage='error', detail='The assistant did not pause. Retry after pausing listening.'))
                if self.restore_listening:
                    self.supervisor.send('resume')
        if self.calibration:
            while not self.calibration_events.empty():
                event = self.calibration_events.get_nowait()
                if event['stage'] in ('result', 'error', 'cancelled'):
                    self.calibration_terminal = event
                else:
                    self.output.put(event)
            if self.calibration.done.is_set():
                self.calibration.thread.join()
                self.calibration = None
                if self.restore_listening:
                    self.supervisor.send('resume')
                self.output.put(self.calibration_terminal or dict(event='calibration', stage='error', detail='Microphone test stopped unexpectedly.'))
                self.calibration_terminal = None

    def run(self):
        last_snapshot = 0
        resources = ''
        next_resources = 0
        try:
            while True:
                closing = False
                while True:
                    try:
                        command = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if command['action'] == 'quit':
                        closing = True
                        break
                    try:
                        self.handle(command)
                    except Exception as exc:
                        self.output.put(dict(event='failure', kind='settings' if command['action'] == 'preferences' else 'setup',
                                             code='server_settings' if command['action'] == 'advanced' else 'settings_invalid', action=command['action'], detail=str(exc)))
                if closing:
                    break
                self.supervisor.tick()
                self.tick_calibration()
                while not self.supervisor.updates.empty():
                    event = self.supervisor.updates.get_nowait()
                    if event.get('event') == 'configured' and (
                            event.get('request_id') == self.preferences_request and self.preferences_request is not None
                            or event.get('request_id') is None and self.pending_preferences is not None):
                        self.commit_preferences(validate_preferences(event.get('preferences') or self.pending_preferences))
                    elif event.get('event') == 'failure' and event.get('kind') == 'settings':
                        self.pending_preferences = None
                    self.output.put(event)
                if self.pending_preferences is not None and (self.supervisor.phase != 'Running' or time.monotonic() >= self.preferences_deadline):
                    self.pending_preferences = None
                    self.output.put(dict(event='failure', kind='settings', code='diagnostics', detail='Settings were not acknowledged by the assistant. Your edits are kept; retry when services are ready.'))
                now = time.monotonic()
                if now >= next_resources:
                    resources = f"RAM: {memory_status()}   ·   Disk: {shutil.disk_usage(Path(__file__).parent).free / 2**30:.1f} GB free"
                    next_resources = now + 2
                if now - last_snapshot >= .1:
                    self.output.put(dict(event='snapshot', state=self.supervisor.state,
                        phase=self.supervisor.phase, server=self.supervisor.server_state,
                        recovery=('server_settings' if any(str(value).startswith('MISSING:') for key, value in self.supervisor.checks.items() if key in ('Model', 'Server executable', 'Vision/audio model'))
                                  else 'missing_dependencies' if any(str(value).startswith('MISSING') for value in self.supervisor.checks.values())
                                  else 'server_unavailable' if self.supervisor.server_state == 'Unavailable' else 'diagnostics'),
                        detail=self.supervisor.notice, checks=dict(self.supervisor.checks),
                        logs=list(self.supervisor.logs), resources=resources,
                        running=bool(self.supervisor.assistant and self.supervisor.assistant.poll() is None)))
                    last_snapshot = now
                self.closed.wait(.03)
        except Exception as exc:
            self.output.put(dict(event='failure', kind='supervisor', detail=f'Service supervisor stopped: {exc}'))
        finally:
            if self.calibration:
                self.calibration.cancelled.set()
                self.calibration.thread.join(timeout=6)
            try:
                self.supervisor.close()
            except Exception as exc:
                self.output.put(dict(event='failure', kind='shutdown', detail=f'Shutdown failed: {exc}'))
            self.closed.set()
            self.output.put(dict(event='closed'))


class DesktopApp:
    def __init__(self, root, args, *, worker_factory=SupervisorWorker, discover=discover_microphones):
        self.root, self.args, self.discover = root, args, discover
        self.preferences, warning = initial_preferences(args)
        self.appearance = load_appearance(args.settings_file)
        args.caption_position = self.appearance['caption_position']
        self.applying = False
        self.pending_preferences = None
        self.advanced_pending = None
        self.calibrating = False
        self.pending_text = None
        self.geometry_timer = None
        self.rendered_turns = {}
        self.geometry_ready = False
        self.display_scale = float(root.tk.call('tk', 'scaling')) / (96 / 72)
        self.worker = worker_factory(args, self.preferences)
        self.devices = []
        self.discovery_events = queue.Queue()
        self.refreshing = False
        self.closing = False
        self.state = 'Stopped'
        self.phase = 'Stopped'
        self.running = False
        self.server_state = 'Not checked'
        self.last_notice_snapshot = None
        self.auto_start = not args.wait
        self.conversation = Conversation()
        self.rendered_revision = -1
        self.copy_buttons = []
        self.build()
        self.banner.set(warning)
        self.message.set(warning or 'Finding microphones…')
        self.worker.start()
        self.refresh_devices()
        self.geometry_ready = True
        self.root.after(30, self.poll)

    def build(self):
        root = self.root
        root.title('Gemma Live')
        scale = float(root.tk.call('tk', 'scaling')) / (96 / 72)
        width = min(int(900 * scale), root.winfo_screenwidth() - 80)
        height = min(int(800 * scale), root.winfo_screenheight() - 120)
        root.geometry(f'{width}x{height}')
        root.minsize(min(int(720 * scale), width), min(int(640 * scale), height))
        if self.appearance['geometry']:
            saved = list(self.appearance['geometry'])
            minimum_width, minimum_height = root.minsize()
            saved[2], saved[3] = max(saved[2], minimum_width), max(saved[3], minimum_height)
            x, y, width, height = clamp_geometry(saved, work_areas(root))
            root.minsize(min(minimum_width, width), min(minimum_height, height))
            root.geometry(f'{width}x{height}{x:+d}{y:+d}')
        if self.appearance['maximized']:
            root.state('zoomed')
        c = COLORS
        self.theme, self.theme_images = install_theme(root)
        self.settings_open = False
        self.sections = {}
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        body = ttk.Frame(root, padding=SPACE['page'])
        body.grid(sticky='nsew')
        body.columnconfigure(0, weight=1)
        body.rowconfigure(4, weight=1)
        header = ttk.Frame(body)
        header.grid(row=0, sticky='ew', pady=(0, 12))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text='Gemma Live', style='Title.TLabel').grid(row=0, column=0, sticky='w')
        self.subtitle = ttk.Label(header, text='Your local voice and screen assistant', style='Muted.TLabel')
        self.subtitle.grid(row=1, column=0, sticky='w')
        self.compact_header = height / scale < 600
        if self.compact_header:
            self.subtitle.grid_remove()
        self.settings_button = ttk.Button(header, text='Settings', command=self.toggle_settings)
        self.settings_button.grid(row=0, column=1, rowspan=2, padx=(8, 8))
        self.quit_button = ttk.Button(header, text='Quit', style='Quiet.TButton', command=self.close)
        self.quit_button.grid(row=0, column=2, rowspan=2)
        self.status = tk.StringVar(master=self.root, value='Stopped')
        self.server = tk.StringVar(master=self.root, value='Server: checking…')
        self.message = tk.StringVar(master=self.root)
        self.mic_status = tk.StringVar(master=self.root, value='Microphone off')
        self.resources = tk.StringVar(master=self.root)
        self.banner = tk.StringVar(master=self.root)
        session = ttk.Frame(body, padding=SPACE['section'], style='Panel.TFrame')
        session.grid(row=1, sticky='ew')
        session.columnconfigure(0, weight=1)
        self.status_label = ttk.Label(session, textvariable=self.status, style='State.TLabel', background=c['panel'])
        self.status_label.grid(row=0, column=0, sticky='w')
        ttk.Label(session, textvariable=self.server, style='PanelMuted.TLabel').grid(row=0, column=1, sticky='e')
        message_label = ttk.Label(session, textvariable=self.message, style='PanelMuted.TLabel', wraplength=700)
        message_label.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(4, 12))
        controls = ttk.Frame(session, style='Panel.TFrame')
        controls.grid(row=2, column=0, columnspan=2, sticky='ew')
        self.start_button = ttk.Button(controls, text='Start assistant', style='Primary.TButton', command=self.start_services)
        self.start_button.grid(row=0, column=0, padx=(0, 8))
        self.text_start_button = ttk.Button(controls, text='Start text session', command=lambda: self.start_services('text'))
        self.text_start_button.grid(row=0, column=1, padx=(0, 8))
        self.pause_button = ttk.Button(controls, text='Pause listening', style='Primary.TButton', command=self.pause_resume, state='disabled')
        self.cancel_button = ttk.Button(controls, text='Stop reply', command=lambda: self.worker.send('cancel'), state='disabled')
        self.cancel_button.grid(row=0, column=2)
        mic = ttk.Frame(session, style='Panel.TFrame')
        mic.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(12, 0))
        mic.columnconfigure(0, weight=1)
        ttk.Label(mic, textvariable=self.mic_status, style='PanelMuted.TLabel').grid(row=0, column=0, sticky='w')
        self.meter = ttk.Progressbar(mic, maximum=100, length=100)
        self.meter.grid(row=0, column=1, sticky='e')
        self.recovery_button = ttk.Button(session, text='Open diagnostics', style='Quiet.TButton', command=self.recover)
        self.recovery_code = None
        self.recovery_button.grid(row=4, column=0, columnspan=2, sticky='w', pady=(8, 0))
        self.recovery_button.grid_remove()
        sharing_row = ttk.Frame(body)
        sharing_row.grid(row=2, sticky='ew', pady=(10, 0))
        self.sharing = tk.BooleanVar(master=self.root, value=False)
        self.sharing_text = tk.StringVar(master=self.root, value='Share screen · Off')
        self.share_button = ttk.Checkbutton(sharing_row, textvariable=self.sharing_text, variable=self.sharing, command=self.share_screen)
        self.share_button.pack(side='left')
        sharing_note = ttk.Label(body, text='Primary screen on the next message. Earlier screenshots stay in context until cleared.',
                                 style='Muted.TLabel', wraplength=750)
        sharing_note.grid(row=3, sticky='ew', pady=(0, 10))
        self.chat_frame = ttk.Frame(body, style='Panel.TFrame', padding=12)
        self.chat_frame.grid(row=4, sticky='nsew')
        self.chat_frame.columnconfigure(0, weight=1)
        self.chat_frame.rowconfigure(1, weight=1)
        chat_header = ttk.Frame(self.chat_frame, style='Panel.TFrame')
        chat_header.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 8))
        ttk.Label(chat_header, text='Conversation', style='Section.TLabel', background=c['panel']).pack(side='left')
        self.clear_button = ttk.Button(chat_header, text='Clear conversation', style='Quiet.TButton', command=lambda: self.worker.send('clear'))
        self.clear_button.pack(side='right')
        self.export_button = ttk.Button(chat_header, text='Export', style='Quiet.TButton', command=self.export_conversation)
        self.export_button.pack(side='right', padx=(0, 8))
        self.chat = tk.Text(self.chat_frame, wrap='word', state='disabled', font=('Segoe UI', 11),
                            bg=c['panel'], fg=c['text'], selectbackground=c['border'], selectforeground=c['text'],
                            relief='flat', padx=8, pady=8, height=8, insertbackground=c['text'],
                            highlightthickness=1, highlightbackground=c['panel'], highlightcolor=c['accent'], spacing3=8)
        self.chat.grid(row=1, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(self.chat_frame, command=self.chat.yview)
        scrollbar.grid(row=1, column=1, sticky='ns')
        self.chat_scrollbar = scrollbar
        self.chat.configure(yscrollcommand=lambda first, last: self.chat_scrolled(scrollbar, first, last))
        self.chat.tag_configure('voice', foreground=c['muted'], spacing1=18, spacing3=8)
        self.chat.tag_configure('assistant', foreground=c['accent'], font=('Segoe UI', 11, 'bold'), spacing3=6)
        self.chat.tag_configure('muted', foreground=c['muted'])
        self.jump_button = ttk.Button(chat_header, text='Jump to latest', style='Quiet.TButton', command=self.jump_latest)
        composer_row = ttk.Frame(self.chat_frame, style='Panel.TFrame')
        composer_row.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(12, 0))
        composer_row.columnconfigure(0, weight=1)
        self.composer = tk.Text(composer_row, height=2, wrap='word', font=('Segoe UI', 10),
                                bg=c['background'], fg=c['text'], insertbackground=c['text'],
                                selectbackground=c['border'], relief='flat', padx=8, pady=6,
                                highlightthickness=1, highlightbackground=c['border'], highlightcolor=c['accent'])
        self.composer.grid(row=0, column=0, sticky='ew')
        self.send_button = ttk.Button(composer_row, text='Send', style='Primary.TButton', command=self.send_text, state='disabled')
        self.send_button.grid(row=0, column=1, padx=(8, 0))
        self.composer.bind('<Return>', self.composer_return)
        self.composer.bind('<Shift-Return>', self.composer_newline)
        self.composer.bind('<Control-Return>', self.composer_start)
        self.composer.bind('<<Modified>>', self.composer_changed)
        self.composer_hint = tk.StringVar(master=self.root, value='Start a voice or text session to send a message. Shift+Enter adds a line.')
        hint = ttk.Label(self.chat_frame, textvariable=self.composer_hint, style='PanelMuted.TLabel', wraplength=700)
        hint.grid(row=4, column=0, columnspan=2, sticky='ew', pady=(4, 0))
        banner_label = ttk.Label(body, textvariable=self.banner, foreground=c['warning'], wraplength=760)
        banner_label.grid(row=5, sticky='ew', pady=(6, 0))
        self.advanced_container = ttk.Frame(body)
        self.advanced_container.columnconfigure(0, weight=1)
        self.advanced_container.rowconfigure(1, weight=1)
        settings_header = ttk.Frame(self.advanced_container)
        settings_header.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        ttk.Label(settings_header, text='Settings', style='Section.TLabel').pack(side='left')
        self.back_button = ttk.Button(settings_header, text='Back to conversation', style='Quiet.TButton', command=self.toggle_settings)
        self.back_button.pack(side='right')
        self.advanced_canvas = tk.Canvas(self.advanced_container, bg=c['background'], highlightthickness=0)
        self.advanced_canvas.grid(row=1, column=0, sticky='nsew')
        advanced_scroll = ttk.Scrollbar(self.advanced_container, command=self.advanced_canvas.yview)
        advanced_scroll.grid(row=1, column=1, sticky='ns')
        self.advanced_canvas.configure(yscrollcommand=advanced_scroll.set)
        self.advanced = ttk.Frame(self.advanced_canvas)
        advanced_window = self.advanced_canvas.create_window(0, 0, window=self.advanced, anchor='nw')
        self.advanced.bind('<Configure>', lambda _: self.advanced_canvas.configure(scrollregion=self.advanced_canvas.bbox('all')))
        self.advanced_canvas.bind('<Configure>', lambda e: self.advanced_canvas.itemconfigure(advanced_window, width=e.width))
        self.advanced.columnconfigure(0, weight=1)
        setup = ttk.LabelFrame(self.advanced, text='Microphone and replies', padding=12)
        setup.grid(row=0, sticky='ew', pady=(0, 12))
        setup.columnconfigure(0, weight=1)
        self.device = tk.StringVar(master=self.root, value=self.preferences['device'])
        ttk.Label(setup, text='Microphone').grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 4))
        self.device_select = ttk.Combobox(setup, textvariable=self.device, state='readonly')
        self.device_select.grid(row=1, column=0, sticky='ew')
        self.refresh_button = ttk.Button(setup, text='Rescan microphones', command=self.refresh_devices)
        self.refresh_button.grid(row=1, column=1, padx=(8, 0))
        calibration_row = ttk.Frame(setup)
        calibration_row.grid(row=4, column=0, columnspan=2, sticky='ew', pady=(8, 0))
        self.calibration_button = ttk.Button(calibration_row, text='Test microphone', command=self.test_microphone)
        self.calibration_button.pack(side='left')
        self.calibration_cancel = ttk.Button(calibration_row, text='Cancel test', style='Quiet.TButton', command=self.cancel_calibration, state='disabled')
        self.calibration_cancel.pack(side='left', padx=(8, 0))
        self.calibration_meter = ttk.Progressbar(calibration_row, maximum=100, length=90)
        self.calibration_meter.pack(side='right')
        self.calibration_message = tk.StringVar(master=self.root, value='Measure background noise, then speak for five seconds.')
        self.calibration_label = ttk.Label(setup, textvariable=self.calibration_message, style='Muted.TLabel', wraplength=650)
        self.calibration_label.grid(row=5, column=0, columnspan=2, sticky='ew', pady=(6, 0))
        ttk.Label(setup, text='Caption position').grid(row=6, column=0, sticky='w', pady=(8, 4))
        self.caption_position = tk.StringVar(master=self.root, value=self.appearance['caption_position'])
        position = ttk.Combobox(setup, values=POSITIONS, textvariable=self.caption_position, state='readonly', width=16)
        position.grid(row=7, column=0, sticky='w')
        position.bind('<<ComboboxSelected>>', self.change_caption_position)
        self.voice = tk.BooleanVar(master=self.root, value=self.preferences['voice'])
        self.captions = tk.BooleanVar(master=self.root, value=self.preferences['captions'])
        ttk.Checkbutton(setup, text='Speak replies', variable=self.voice).grid(row=2, column=0, columnspan=2, sticky='w', pady=(8, 0))
        ttk.Checkbutton(setup, text='Floating captions', variable=self.captions).grid(row=3, column=0, columnspan=2, sticky='w')
        self.speech_section = self.make_section('Speech detection', 1)
        self.server_section = self.make_section('Server configuration', 3)
        self.speech_section.columnconfigure(1, weight=1)
        self.server_section.columnconfigure(1, weight=1)
        self.silence = tk.StringVar(master=self.root, value=str(self.preferences['silence']))
        self.threshold = tk.StringVar(master=self.root, value=str(self.preferences['threshold']))
        ttk.Label(self.speech_section, text='End message after silence (seconds)').grid(row=0, column=0, sticky='w')
        ttk.Entry(self.speech_section, textvariable=self.silence, width=10).grid(row=0, column=1, sticky='w')
        ttk.Label(self.speech_section, text='Speech threshold (lower = more sensitive)').grid(row=1, column=0, sticky='w')
        ttk.Entry(self.speech_section, textvariable=self.threshold, width=10).grid(row=1, column=1, sticky='w')
        presets = ttk.Frame(self.speech_section)
        presets.grid(row=2, column=0, columnspan=2, sticky='w', pady=(8, 0))
        for label, seconds in (('Quick · 0.5 s', .5), ('Natural · 1 s', 1), ('Unhurried · 1.5 s', 1.5)):
            ttk.Button(presets, text=label, style='Quiet.TButton', command=lambda value=seconds: self.silence.set(str(value))).pack(side='left', padx=(0, 6))
        self.server_vars = {}
        for row, (key, label) in enumerate((('server', 'Server executable'), ('model', 'Model'),
                ('mmproj', 'Vision/audio model'), ('mtp', 'Draft model'), ('port', 'Port'),
                ('ctx', 'Context size'), ('gpu_layers', 'GPU layers')), start=0):
            self.server_vars[key] = tk.StringVar(master=self.root, value=str(getattr(self.args, key)))
            ttk.Label(self.server_section, text=label).grid(row=row, column=0, sticky='w')
            ttk.Entry(self.server_section, textvariable=self.server_vars[key]).grid(row=row, column=1, sticky='ew', pady=2)
            if key in ('server', 'model', 'mmproj', 'mtp'):
                ttk.Button(self.server_section, text='Browse', command=lambda k=key: self.browse(k)).grid(row=row, column=2, padx=(8, 0))
        self.no_mtp = tk.BooleanVar(master=self.root, value=self.args.no_mtp)
        ttk.Checkbutton(self.server_section, text='Disable draft model', variable=self.no_mtp).grid(row=7, column=0, sticky='w')
        self.server_apply_button = ttk.Button(self.server_section, text='Apply server settings', command=self.apply_advanced)
        self.server_apply_button.grid(row=8, column=0, sticky='w', pady=(8, 0))
        self.server_discard_button = ttk.Button(self.server_section, text='Discard server edits', style='Quiet.TButton', command=self.discard_advanced)
        self.server_discard_button.grid(row=8, column=1, sticky='e')
        self.server_settings_state = tk.StringVar(master=self.root)
        ttk.Label(self.server_section, textvariable=self.server_settings_state, style='Muted.TLabel').grid(row=9, column=0, columnspan=3, sticky='w')
        self.applied_server_values = self.server_form_values()

        apply_row = ttk.Frame(self.advanced_container)
        apply_row.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(12, 0))
        apply_row.columnconfigure(2, weight=1)
        self.apply_button = ttk.Button(apply_row, text='Apply changes', style='Primary.TButton', command=self.apply_setup)
        self.apply_button.grid(row=0, column=0, sticky='w', padx=(0, 12))
        apply_note = ttk.Label(apply_row, text='Applying microphone, reply, or speech detection changes stops the current reply.', style='Muted.TLabel', wraplength=500)
        self.discard_button = ttk.Button(apply_row, text='Discard changes', style='Quiet.TButton', command=self.discard_setup)
        self.discard_button.grid(row=0, column=1, sticky='w', padx=(0, 12))
        self.settings_state = tk.StringVar(master=self.root)
        ttk.Label(apply_row, textvariable=self.settings_state, style='Muted.TLabel').grid(row=0, column=2, sticky='w')
        apply_note.grid(row=1, column=0, columnspan=3, sticky='ew', pady=(6, 0))
        apply_row.bind('<Configure>', lambda e: apply_note.configure(
            wraplength=max(120, e.width - 20)))
        self.diagnostics_section = self.make_section('Resources and diagnostics', 7)
        self.diagnostics_section.columnconfigure(0, weight=1)
        self.diagnostics = tk.StringVar(master=self.root)
        ttk.Label(self.diagnostics_section, textvariable=self.resources, style='Muted.TLabel').grid(row=0, sticky='w', pady=(0, 8))
        diagnostics_label = ttk.Label(self.diagnostics_section, textvariable=self.diagnostics, style='Muted.TLabel', wraplength=700)
        diagnostics_label.grid(row=1, sticky='ew')
        body.bind('<Configure>', lambda e: [label.configure(wraplength=max(200, e.width - 56))
                                          for label in (message_label, sharing_note, banner_label, diagnostics_label, hint, self.calibration_label)])
        for variable in (self.device, self.voice, self.captions, self.silence, self.threshold, self.no_mtp, *self.server_vars.values()):
            variable.trace_add('write', self.settings_changed)
        self.settings_changed()
        root.bind('<Configure>', self.window_changed, add='+')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.bind('<Control-Return>', lambda _: self.start_services())
        root.bind('<Control-space>', lambda _: self.pause_resume())
        root.bind('<Escape>', lambda _: self.worker.send('cancel'))
        root.bind('<MouseWheel>', lambda e: self.advanced_canvas.yview_scroll(-int(e.delta / 120), 'units')
                  if self.advanced_container.winfo_ismapped() else None)
        if os.name == 'nt':
            root.update_idletasks()
            user32 = ctypes.windll.user32
            user32.GetParent.argtypes = [ctypes.c_void_p]
            user32.GetParent.restype = ctypes.c_void_p
            user32.SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            user32.SetWindowDisplayAffinity(user32.GetParent(root.winfo_id()), 0x11)

    def refresh_devices(self):
        if self.refreshing or self.closing:
            return
        self.refreshing = True
        self.refresh_button.configure(state='disabled')
        def scan():
            try:
                self.discovery_events.put(self.discover())
            except Exception as exc:
                self.discovery_events.put(([], f'Microphone discovery failed: {exc}'))
        threading.Thread(target=scan, name='Microphone discovery', daemon=True).start()

    def read_setup(self):
        if self.device.get() and self.device.get() not in self.devices:
            raise ValueError('Select an available microphone, or refresh the device list.')
        return validate_preferences(dict(device=self.device.get(), silence=float(self.silence.get()),
            threshold=float(self.threshold.get()), voice=self.voice.get(), captions=self.captions.get()))

    def settings_changed(self, *_):
        try:
            draft = self.read_setup()
            dirty = draft != self.preferences
        except ValueError:
            dirty = True
        self.settings_state.set('Applying…' if self.applying else 'Unsaved changes' if dirty else 'Applied settings')
        self.apply_button.configure(state='normal' if dirty and not self.applying and not self.calibrating and not self.closing else 'disabled')
        self.discard_button.configure(state='normal' if dirty and not self.applying and not self.calibrating and not self.closing else 'disabled')
        advanced_dirty = self.server_form_values() != self.applied_server_values
        self.server_settings_state.set('Applying…' if self.advanced_pending else 'Unsaved server edits' if advanced_dirty else 'Server settings unchanged')
        can_apply = advanced_dirty and not self.advanced_pending and self.phase in ('Stopped', 'Error') and not self.running and not self.calibrating and not self.closing
        self.server_apply_button.configure(state='normal' if can_apply else 'disabled')
        self.server_discard_button.configure(state='normal' if advanced_dirty and not self.advanced_pending and not self.closing else 'disabled')

    def discard_setup(self):
        if self.applying or self.calibrating:
            return
        for variable, key in ((self.device, 'device'), (self.voice, 'voice'), (self.captions, 'captions'),
                              (self.silence, 'silence'), (self.threshold, 'threshold')):
            variable.set(self.preferences[key])
        self.settings_changed()

    def server_form_values(self):
        return dict({key: var.get() for key, var in self.server_vars.items()}, no_mtp=self.no_mtp.get())

    def discard_advanced(self):
        if self.advanced_pending:
            return
        for key, variable in self.server_vars.items():
            variable.set(self.applied_server_values[key])
        self.no_mtp.set(self.applied_server_values['no_mtp'])
        self.settings_changed()

    def apply_setup(self):
        if self.closing or self.applying or self.calibrating:
            return False
        try:
            values = self.read_setup()
        except ValueError as exc:
            self.message.set(str(exc))
            self.show_recovery('settings_invalid')
            return False
        if values == self.preferences:
            return True
        self.pending_preferences = values
        self.applying = True
        self.settings_changed()
        self.worker.send('preferences', preferences=values, request_id=uuid.uuid4().hex)
        return True

    def start_services(self, mode='voice'):
        if self.closing or self.calibrating or self.applying:
            return
        self.recovery_button.grid_remove()
        if self.running:
            if self.server_state == 'Unavailable':
                self.message.set('Reconnecting local services…')
                self.worker.send('restart')
            elif self.state in ('Error', 'Paused'):
                self.pause_resume()
        elif mode == 'voice' and (not self.preferences['device'] or self.preferences['device'] not in self.devices):
            self.message.set('Apply an available microphone in Settings, or start a text session.')
            self.show_recovery('microphone_unavailable')
            if not self.settings_open:
                self.toggle_settings()
        else:
            self.message.set('Starting local services…')
            self.worker.send('start', mode=mode)

    def pause_resume(self):
        if self.closing or self.calibrating or self.applying or not self.running:
            return
        if self.state in ('Paused', 'Error'):
            if not self.preferences['device'] or self.preferences['device'] not in self.devices:
                self.message.set('Apply an available microphone to resume listening. Typed messages remain available.')
                self.show_recovery('microphone_unavailable')
                return
            self.worker.send('resume')
        else:
            self.worker.send('pause')

    def composer_return(self, event):
        if not event.state & 5:
            self.send_text()
            return 'break'

    def composer_newline(self, _):
        self.composer.insert('insert', '\n')
        return 'break'

    def composer_start(self, _):
        self.start_services()
        return 'break'

    def composer_changed(self, _=None):
        if self.composer.edit_modified():
            self.composer.edit_modified(False)
            self.update_input_controls()

    def update_input_controls(self):
        idle = self.running and self.phase == 'Running' and self.server_state == 'Ready' and self.state in ('Listening', 'Paused', 'Error')
        enabled = idle and not self.calibrating and not self.pending_text and not self.applying and not self.closing
        self.send_button.configure(state='normal' if enabled and self.composer.get('1.0', 'end-1c').strip() else 'disabled')
        self.composer_hint.set('Sending…' if self.pending_text else
            'Finish or cancel the microphone test to send.' if self.calibrating else
            'Start a voice or text session to send a message.' if not self.running else
            'Reconnect the local server to send. Your draft is kept.' if self.server_state != 'Ready' else
            'Finish or stop the current reply to send another message.' if not idle else
            'Enter sends · Shift+Enter adds a line · Uses your speech and screen-sharing settings.')
        self.calibration_button.configure(state='normal' if self.device.get() in self.devices and not self.calibrating and not self.applying and not self.closing and self.phase != 'Loading' and self.state in ('Stopped', 'Listening', 'Paused', 'Error') else 'disabled')
        self.calibration_cancel.configure(state='normal' if self.calibrating and not self.closing else 'disabled')
        self.clear_button.configure(state='normal' if not self.calibrating and not self.closing else 'disabled')
        self.recovery_button.configure(state='normal' if not self.calibrating and not self.closing else 'disabled')

    def send_text(self):
        self.update_input_controls()
        if self.send_button.instate(['disabled']):
            return
        text = self.composer.get('1.0', 'end-1c')
        self.pending_text = dict(text=text, request_id=uuid.uuid4().hex)
        self.worker.send('send_text', **self.pending_text)
        self.update_input_controls()

    def test_microphone(self):
        self.update_input_controls()
        if self.calibration_button.instate(['disabled']):
            return
        self.calibrating = True
        self.calibration_message.set('Pausing listening before the test…')
        self.worker.send('calibrate', device=self.device.get())
        self.settings_changed()
        self.update_input_controls()
        self.start_button.configure(state='disabled')
        self.text_start_button.configure(state='disabled')
        self.pause_button.configure(state='disabled')

    def cancel_calibration(self):
        if self.calibrating:
            self.worker.send('cancel_calibration')
            self.calibration_message.set('Stopping microphone test…')

    def change_caption_position(self, _=None):
        self.appearance['caption_position'] = self.caption_position.get()
        self.worker.send('appearance', appearance=dict(self.appearance))

    def window_changed(self, event):
        if event.widget is not self.root or not self.geometry_ready or self.closing:
            return
        compact = event.height / self.display_scale < 600
        if compact != self.compact_header:
            self.compact_header = compact
            if compact:
                self.subtitle.grid_remove()
            else:
                self.subtitle.grid()
        if self.geometry_timer:
            self.root.after_cancel(self.geometry_timer)
        self.geometry_timer = self.root.after(500, self.remember_window)

    def remember_window(self):
        self.geometry_timer = None
        state = self.root.state()
        if state in ('iconic', 'withdrawn'):
            return
        appearance = dict(self.appearance)
        appearance['maximized'] = state == 'zoomed'
        if state == 'normal':
            appearance['geometry'] = [self.root.winfo_x(), self.root.winfo_y(), self.root.winfo_width(), self.root.winfo_height()]
        if appearance != self.appearance:
            self.appearance = appearance
            self.worker.send('appearance', appearance=dict(appearance))

    def show_recovery(self, code):
        actions = dict(microphone_unavailable='Open microphone settings', settings_invalid='Open settings',
                       microphone_scan_failed='Rescan microphones',
                       server_settings='Open server settings', server_unavailable='Reconnect',
                       reply_failed='Open diagnostics', missing_dependencies='Open diagnostics',
                       voice_unavailable='Open diagnostics', captions_unavailable='Open settings')
        self.recovery_code = code
        self.recovery_button.configure(text=actions.get(code, 'Open diagnostics'))
        self.recovery_button.grid()

    def recover(self):
        code = self.recovery_code
        if code == 'microphone_scan_failed':
            self.refresh_devices()
            if not self.settings_open:
                self.toggle_settings()
            return
        if code == 'server_unavailable':
            self.start_services('text' if not self.preferences['device'] else 'voice')
            return
        if not self.settings_open:
            self.toggle_settings()
        section = 'Server configuration' if code == 'server_settings' else 'Resources and diagnostics' if code not in ('microphone_unavailable', 'settings_invalid', 'captions_unavailable') else None
        if section and not self.sections[section][0].winfo_manager():
            self.toggle_section(section)
        self.root.update_idletasks()
        if section:
            total = max(1, self.advanced.winfo_reqheight())
            self.advanced_canvas.yview_moveto(self.sections[section][1].winfo_y() / total)
        else:
            self.advanced_canvas.yview_moveto(0)
            self.device_select.focus_set()

    def export_conversation(self):
        if not self.conversation.turns:
            self.message.set('There are no messages to export yet.')
            return
        path = filedialog.asksaveasfilename(parent=self.root, title='Export conversation',
            defaultextension='.md', initialfile='Gemma Live conversation.md', filetypes=[('Markdown', '*.md')])
        if path:
            try:
                Path(path).write_text(markdown_conversation(self.conversation.turns), encoding='utf-8')
                self.message.set('Conversation exported.')
            except OSError as exc:
                self.message.set(f'Could not export conversation: {exc}')

    def chat_scrolled(self, scrollbar, first, last):
        scrollbar.set(first, last)
        if hasattr(self, 'jump_button') and not getattr(self, 'rendering_chat', False):
            if self.conversation.turns and float(last) < .98:
                self.jump_button.pack(side='left', padx=(12, 0))
            else:
                self.jump_button.pack_forget()

    def jump_latest(self):
        self.rendering_chat = True
        self.jump_button.pack_forget()
        self.chat.update_idletasks()
        # Resolve Tk's deferred wrapping before choosing the actual bottom.
        self.chat.tk.call(self.chat._w, 'count', '-update', '-ypixels', '1.0', 'end')
        self.chat.yview_moveto(1)
        self.chat.update_idletasks()
        self.rendering_chat = False
        self.chat_scrolled(self.chat_scrollbar, *self.chat.yview())

    def share_screen(self):
        self.sharing_text.set('Share screen · ' + ('On' if self.sharing.get() else 'Off'))
        self.worker.send('screen_sharing', enabled=self.sharing.get())

    def toggle_settings(self):
        if self.closing:
            return
        if self.settings_open:
            self.advanced_container.grid_remove()
            self.chat_frame.grid()
            self.settings_open = False
            self.settings_button.configure(text='Settings')
            self.settings_button.focus_set()
        else:
            self.chat_frame.grid_remove()
            self.advanced_container.grid(row=4, sticky='nsew')
            self.settings_open = True
            self.settings_button.configure(text='Conversation')
            self.back_button.focus_set()

    def make_section(self, title, row):
        section = ttk.Frame(self.advanced, padding=(0, 8))
        button = ttk.Button(self.advanced, text=title + '  +', style='Quiet.TButton',
                            command=lambda: self.toggle_section(title))
        button.grid(row=row, sticky='ew', pady=(4, 0))
        self.sections[title] = (section, button, row + 1)
        return section

    def toggle_section(self, title):
        section, button, row = self.sections[title]
        if section.winfo_manager():
            section.grid_remove()
            button.configure(text=title + '  +')
        else:
            section.grid(row=row, sticky='ew')
            button.configure(text=title + '  −')

    def toggle_advanced(self):
        # Retain the old navigation entry point for callers.
        self.toggle_settings()

    def browse(self, key):
        chosen = filedialog.askopenfilename(parent=self.root, title=f'Select {key}')
        if chosen:
            self.server_vars[key].set(chosen)

    def apply_advanced(self):
        try:
            values = {key: (int(var.get()) if key in ('port', 'ctx', 'gpu_layers') else Path(var.get()))
                      for key, var in self.server_vars.items()}
            values['no_mtp'] = self.no_mtp.get()
            self.advanced_pending = self.server_form_values()
            self.settings_changed()
            self.worker.send('advanced', values=values)
        except ValueError:
            self.message.set('Port, context size, and GPU layers must be whole numbers.')

    def render_conversation(self):
        if self.rendered_revision == self.conversation.revision:
            return
        follow = not self.rendered_turns or self.chat.yview()[1] >= .98
        self.chat.mark_set('reading_position', self.chat.index('@0,0'))
        self.chat.mark_gravity('reading_position', 'left')
        self.rendering_chat = True
        self.chat.configure(state='normal')
        for key in list(self.rendered_turns):
            if key not in self.conversation.turns:
                record = self.rendered_turns.pop(key)
                self.chat.delete(record['start'], record['end'])
                if record.get('button'):
                    record['button'].destroy()
                for mark in ('start', 'reply_start', 'reply_end', 'footer', 'end'):
                    self.chat.mark_unset(record[mark])
        if not self.conversation.turns:
            self.chat.delete('1.0', 'end')
            self.chat.insert('end', 'Speak when the microphone says Listening, or type a message below.\nEnable screen sharing when you want Gemma to see your screen.', 'muted')
        for key, turn in self.conversation.turns.items():
            record = self.rendered_turns.get(key)
            signature = (turn['text'], turn['complete'], turn['ended'])
            if record and record['signature'] == signature:
                continue
            if record is None:
                if not self.rendered_turns:
                    self.chat.delete('1.0', 'end')
                prefix = 'turn_' + uuid.uuid4().hex
                record = {mark: prefix + '_' + mark for mark in ('start', 'reply_start', 'reply_end', 'footer', 'end')}
                self.chat.mark_set(record['start'], 'end-1c')
                self.chat.mark_gravity(record['start'], 'left')
                label = 'You · Typed message' if turn['input_kind'] == 'text' else 'You · Voice message'
                self.chat.insert('end', label + (' · screen shared' if turn['screen'] else '') + '\n', 'voice')
                if turn['input_kind'] == 'text':
                    self.chat.insert('end', turn['input_text'] + '\n')
                self.chat.insert('end', 'Gemma\n', 'assistant')
                for mark in ('reply_start', 'reply_end', 'footer', 'end'):
                    self.chat.mark_set(record[mark], 'end-1c')
                    self.chat.mark_gravity(record[mark], 'left')
                record['button'] = None
                record['footer_signature'] = None
                self.rendered_turns[key] = record
            text = turn['text'] or ('No completed reply.' if turn['ended'] else 'Waiting for Gemma…')
            self.chat.delete(record['reply_start'], record['reply_end'])
            for mark in ('reply_end', 'footer', 'end'):
                self.chat.mark_gravity(record[mark], 'right')
            self.chat.insert(record['reply_start'], text + '\n')
            for mark in ('reply_end', 'footer', 'end'):
                self.chat.mark_gravity(record[mark], 'left')
            footer_signature = (turn['ended'] and not turn['complete'], bool(turn['text'] and (turn['ended'] or turn['complete'])))
            if footer_signature != record['footer_signature']:
                self.chat.delete(record['footer'], record['end'])
                if record['button']:
                    record['button'].destroy()
                    record['button'] = None
                self.chat.mark_gravity(record['end'], 'right')
                if turn['ended'] and not turn['complete']:
                    self.chat.insert(record['footer'], 'Incomplete reply\n', 'muted')
                if turn['text'] and (turn['ended'] or turn['complete']):
                    button = ttk.Button(self.chat, text='Copy reply', style='Copy.TButton')
                    button.configure(command=lambda k=key, b=button: self.copy_reply(self.conversation.turns[k]['text'], b))
                    self.chat.window_create(record['end'], window=button)
                    self.chat.insert(record['end'], '\n')
                    record['button'] = button
                self.chat.mark_gravity(record['end'], 'left')
                record['footer_signature'] = footer_signature
            record['signature'] = signature
        self.copy_buttons = [record['button'] for record in self.rendered_turns.values() if record['button']]
        # Tk always keeps a final newline. Keep its empty display line compact,
        # so scrolling to the end leaves room for the full reply action.
        self.chat.tag_configure('tail', font=('Segoe UI', 1), spacing1=0, spacing3=0)
        self.chat.tag_remove('tail', '1.0', 'end')
        self.chat.tag_add('tail', 'end-1c', 'end')
        self.chat.configure(state='disabled')
        self.chat.update_idletasks()
        if follow:
            self.chat.yview_moveto(1)
        else:
            self.chat.yview('reading_position')
        self.rendering_chat = False
        self.chat_scrolled(self.chat_scrollbar, *self.chat.yview())
        self.rendered_revision = self.conversation.revision

    def copy_reply(self, text, button=None):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        if button is not None:
            button.configure(text='Copied')
            def restore():
                if button.winfo_exists():
                    button.configure(text='Copy reply')
            self.root.after(1500, restore)

    def handle_event(self, event):
        kind = event.get('event')
        if self.closing and kind not in ('closed', 'failure'):
            return
        self.conversation.handle(event)
        if kind == 'snapshot':
            self.state, self.phase, self.running = event['state'], event['phase'], event['running']
            self.status.set(self.state)
            self.server.set('Server: ' + event['server'])
            self.server_state = event['server']
            self.resources.set(event['resources'])
            self.diagnostics.set('\n'.join(f'{key}: {value}' for key, value in event['checks'].items())
                                 + '\nRecent activity:\n' + '\n'.join(event['logs']))
            notice_snapshot = (self.phase, self.state, event['detail'])
            changed = notice_snapshot != self.last_notice_snapshot
            self.last_notice_snapshot = notice_snapshot
            if changed and (self.phase in ('Error', 'Loading') or self.state == 'Error'):
                self.message.set(event['detail'].replace('press Enter', 'click Retry').replace('Enter retries, Q quits', 'Retry reconnects; Quit closes the app'))
            elif changed and self.phase == 'Stopped' and event['detail'].startswith('Assistant exited'):
                self.message.set('Assistant stopped. Click Start assistant to begin again.')
            if changed and self.running and self.server_state == 'Unavailable':
                self.message.set('Local server unavailable. Click Reconnect to reconnect; this resets conversation context.')
            busy = self.state in ('Thinking', 'Preparing voice', 'Speaking')
            self.cancel_button.configure(state='normal' if busy and not self.calibrating else 'disabled')
            self.pause_button.configure(text='Resume listening' if self.state in ('Paused', 'Error') else 'Pause listening',
                                        state='normal' if self.running and not self.calibrating and not self.applying else 'disabled')
            needs_start = (not self.running or self.phase in ('Stopped', 'Error') or self.state == 'Error'
                           or self.server_state == 'Unavailable')
            label = ('Reconnect' if self.running and self.server_state == 'Unavailable' else
                     'Retry' if self.phase == 'Error' or self.state == 'Error' else 'Start assistant')
            self.start_button.configure(text=label, state='normal' if needs_start and self.phase != 'Loading' and not self.calibrating and not self.applying else 'disabled')
            if not self.running:
                self.text_start_button.grid(row=0, column=1, padx=(0, 8))
                self.text_start_button.configure(state='normal' if self.phase in ('Stopped', 'Error') and not self.calibrating and not self.applying else 'disabled')
            else:
                self.text_start_button.grid_remove()
            if needs_start:
                self.start_button.grid(row=0, column=0, padx=(0, 8))
                self.pause_button.grid_remove()
            else:
                self.start_button.grid_remove()
                self.pause_button.grid(row=0, column=1, padx=(0, 8))
            if self.phase == 'Error' or (self.running and self.server_state == 'Unavailable'):
                self.show_recovery(event.get('recovery', 'diagnostics'))
            self.settings_changed()
            self.update_input_controls()
            self.status_label.configure(foreground=COLORS['error'] if self.state == 'Error' or self.server_state == 'Unavailable'
                                        else COLORS['accent'] if self.state == 'Listening' else COLORS['muted'])
            if self.state != 'Listening':
                self.meter['value'] = 0
                self.mic_status.set('Microphone off while replying' if busy else 'Microphone off')
            elif not self.mic_status.get().startswith(('Listening', 'Speech detected')):
                self.mic_status.set('Listening')
        elif kind == 'microphone' and self.state == 'Listening':
            self.meter['value'] = min(100, event.get('level', 0) * 1000)
            self.mic_status.set('Speech detected' if event.get('speech') else 'Listening')
        elif kind == 'status':
            if event.get('state') == 'Listening' and not event.get('detail'):
                self.message.set('Ready. Speak naturally; Gemma pauses the microphone while replying.')
        elif kind == 'failure':
            self.message.set(event.get('detail', 'An operation failed.'))
            self.show_recovery(event.get('code', 'diagnostics'))
            if event.get('kind') == 'settings' or event.get('action') == 'preferences':
                self.applying = False
                self.pending_preferences = None
            if event.get('action') == 'advanced':
                self.advanced_pending = None
            if event.get('action') == 'calibrate':
                self.calibrating = False
                self.calibration_message.set(event['detail'])
            if self.pending_text and (event.get('kind') == 'input' or event.get('request_id') == self.pending_text['request_id'] or event.get('action') == 'send_text'):
                self.pending_text = None
            self.settings_changed()
            self.update_input_controls()
        elif kind in ('preferences_applied', 'configured'):
            if kind == 'preferences_applied':
                self.preferences = dict(event.get('preferences') or self.pending_preferences or self.preferences)
                self.applying = False
                self.pending_preferences = None
                self.settings_changed()
                self.update_input_controls()
            if event.get('warning'):
                self.banner.set(event['warning'])
            elif kind == 'preferences_applied':
                self.banner.set('')
                self.message.set('Changes saved.')
            elif kind == 'configured':
                self.banner.set('' if event.get('voice_available') else 'Voice output is off. Replies appear in the conversation.')
        elif kind == 'turn_started':
            if self.pending_text and event.get('request_id') == self.pending_text['request_id']:
                if self.composer.get('1.0', 'end-1c') == self.pending_text['text']:
                    self.composer.delete('1.0', 'end')
                self.pending_text = None
            self.state = 'Thinking'
            self.recovery_button.grid_remove()
            self.update_input_controls()
        elif kind == 'assistant_stopped':
            self.pending_text = None
            if self.applying:
                self.applying = False
                self.pending_preferences = None
                self.message.set('Assistant stopped before settings were acknowledged. Your edits are kept.')
            self.update_input_controls()
            self.settings_changed()
        elif kind == 'advanced_applied':
            self.applied_server_values = dict(self.advanced_pending or self.server_form_values())
            self.advanced_pending = None
            self.message.set(event['detail'])
            self.settings_changed()
        elif kind == 'calibration':
            stage = event['stage']
            if stage == 'noise':
                self.calibration_message.set('Stay quiet for three seconds while background noise is measured.')
            elif stage == 'speech':
                self.calibration_message.set('Speak naturally now for five seconds.')
            elif stage == 'level':
                self.calibration_meter['value'] = min(100, event['level'] * 1000)
            elif stage in ('result', 'error', 'cancelled'):
                self.calibrating = False
                self.calibration_meter['value'] = 0
                if stage == 'result':
                    self.threshold.set(str(event['threshold']))
                    self.calibration_message.set('Suggested sensitivity added to your draft. Choose a silence preset and apply changes.')
                    if not self.speech_section.winfo_manager():
                        self.toggle_section('Speech detection')
                else:
                    self.calibration_message.set(event.get('detail', 'Microphone test cancelled. Settings were not changed.'))
                self.settings_changed()
                self.update_input_controls()
        elif kind == 'information':
            self.message.set(event['detail'])
        elif kind == 'closed' and not self.closing:
            self.message.set('Services stopped. Close and reopen the app to restart.')
            self.running = False

    def poll(self):
        try:
            while True:
                devices, error = self.discovery_events.get_nowait()
                self.refreshing = False
                self.refresh_button.configure(state='normal')
                self.devices = devices
                self.device_select.configure(values=devices)
                selected = self.device.get()
                if selected and selected not in devices:
                    self.device.set('')
                    self.worker.send('pause')
                    self.message.set(error or 'Saved microphone is unavailable. Open Settings, select a microphone, and apply changes.')
                    if not getattr(self.args, 'text_only', False):
                        self.auto_start = False
                    self.show_recovery('microphone_unavailable')
                    if not self.settings_open:
                        self.toggle_settings()
                elif error:
                    self.message.set(error)
                    self.show_recovery('microphone_scan_failed')
                elif not selected:
                    if devices:
                        self.device.set(self.args.device if self.args.device in devices else devices[0])
                    self.message.set('Choose your microphone in Settings, then click Start assistant.')
                    if not self.settings_open:
                        self.toggle_settings()
                elif self.auto_start:
                    self.start_services('text' if getattr(self.args, 'text_only', False) else 'voice')
                if self.auto_start and getattr(self.args, 'text_only', False) and not self.running:
                    self.start_services('text')
                self.auto_start = False
                self.settings_changed()
                self.update_input_controls()
        except queue.Empty:
            pass
        try:
            while True:
                self.handle_event(self.worker.output.get_nowait())
        except queue.Empty:
            pass
        self.render_conversation()
        if self.closing and self.worker.closed.is_set():
            self.root.destroy()
            return
        self.root.after(30, self.poll)

    def close(self):
        if self.closing:
            return
        if self.geometry_timer:
            self.root.after_cancel(self.geometry_timer)
            self.geometry_timer = None
        self.remember_window()
        self.closing = True
        self.message.set('Closing the assistant and services started by this app…')
        for button in (self.start_button, self.pause_button, self.cancel_button, self.apply_button, self.refresh_button, self.clear_button, self.share_button, self.settings_button, self.back_button, self.text_start_button, self.send_button, self.export_button, self.discard_button, self.server_apply_button, self.server_discard_button, self.calibration_button, self.calibration_cancel, self.recovery_button):
            button.configure(state='disabled')
        self.worker.send('quit')


def run_desktop(args):
    if os.name == 'nt':
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    app = DesktopApp(root, args)
    try:
        root.mainloop()
    finally:
        if not app.closing:
            app.close()
        app.worker.thread.join(timeout=12)
    return 0
