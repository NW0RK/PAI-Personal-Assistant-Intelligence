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
from tkinter import ttk, filedialog

from app_settings import (load_preferences, save_preferences, validate_preferences,
                          discover_microphones)
from launch_gemma import Supervisor, memory_status


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
                                              screen=event.get('screen_sharing', False))
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

    def start(self):
        self.thread.start()

    def send(self, action, **fields):
        self.commands.put(dict(action=action, **fields))

    def handle(self, command):
        action = command['action']
        if action == 'start':
            self.supervisor.start()
        elif action == 'restart':
            self.supervisor.restart()
            self.output.put(dict(event='cleared'))
        elif action == 'preferences':
            values = validate_preferences(command['preferences'])
            if not values['device']:
                raise ValueError('Select a microphone before applying settings.')
            apply_preferences(self.args, values)
            warning = ''
            try:
                save_preferences(values, self.args.settings_file)
            except OSError as exc:
                warning = f'Settings apply for this session but could not be saved: {exc}'
            if self.supervisor.phase == 'Running':
                self.supervisor.send(dict(action='configure', preferences=values))
            self.output.put(dict(event='preferences_applied', warning=warning))
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
            self.output.put(dict(event='information', detail='Server settings applied for this session.'))
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
                        self.output.put(dict(event='failure', kind='setup', detail=str(exc)))
                if closing:
                    break
                self.supervisor.tick()
                while not self.supervisor.updates.empty():
                    self.output.put(self.supervisor.updates.get_nowait())
                now = time.monotonic()
                if now >= next_resources:
                    resources = f"RAM: {memory_status()}   ·   Disk: {shutil.disk_usage(Path(__file__).parent).free / 2**30:.1f} GB free"
                    next_resources = now + 2
                if now - last_snapshot >= .1:
                    self.output.put(dict(event='snapshot', state=self.supervisor.state,
                        phase=self.supervisor.phase, server=self.supervisor.server_state,
                        detail=self.supervisor.notice, checks=dict(self.supervisor.checks),
                        logs=list(self.supervisor.logs), resources=resources,
                        running=bool(self.supervisor.assistant and self.supervisor.assistant.poll() is None)))
                    last_snapshot = now
                self.closed.wait(.03)
        except Exception as exc:
            self.output.put(dict(event='failure', kind='supervisor', detail=f'Service supervisor stopped: {exc}'))
        finally:
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
        self.root.after(30, self.poll)

    def build(self):
        root = self.root
        root.title('Gemma Live')
        scale = float(root.tk.call('tk', 'scaling')) / (96 / 72)
        width = min(int(900 * scale), root.winfo_screenwidth() - 80)
        height = min(int(800 * scale), root.winfo_screenheight() - 120)
        root.geometry(f'{width}x{height}')
        root.minsize(min(int(720 * scale), width), min(int(640 * scale), height))
        root.configure(bg='#111827')
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('.', font=('Segoe UI', 10), background='#111827', foreground='#e5e7eb')
        style.configure('TFrame', background='#111827')
        style.configure('TLabel', background='#111827', foreground='#e5e7eb')
        style.configure('Title.TLabel', font=('Segoe UI', 23, 'bold'))
        style.configure('State.TLabel', font=('Segoe UI', 15, 'bold'), foreground='#93c5fd')
        style.configure('Muted.TLabel', foreground='#aebbd0')
        style.configure('TButton', background='#26364e', foreground='#f9fafb', padding=(10, 7))
        style.map('TButton', background=[('active', '#345078'), ('disabled', '#1a2434')],
                  foreground=[('disabled', '#77869c')])
        style.configure('TCheckbutton', background='#111827', foreground='#e5e7eb')
        style.map('TCheckbutton', background=[('active', '#111827')])
        style.configure('TEntry', fieldbackground='#e5e7eb', foreground='#111827')
        style.configure('TCombobox', fieldbackground='#e5e7eb', foreground='#111827')
        style.map('TCombobox', fieldbackground=[('readonly', '#e5e7eb')], foreground=[('readonly', '#111827')])
        style.configure('TLabelframe', background='#111827', bordercolor='#334155')
        style.configure('TLabelframe.Label', background='#111827', foreground='#aebbd0')
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        body = ttk.Frame(root, padding=22)
        body.grid(sticky='nsew')
        body.columnconfigure(0, weight=1)
        body.rowconfigure(7, weight=1)
        ttk.Label(body, text='Gemma Live', style='Title.TLabel').grid(row=0, sticky='w')
        ttk.Label(body, text='Your local voice and screen assistant', style='Muted.TLabel').grid(row=1, sticky='w', pady=(0, 12))
        self.status = tk.StringVar(value='Stopped')
        self.server = tk.StringVar(value='Server: checking…')
        self.message = tk.StringVar()
        self.mic_status = tk.StringVar(value='Microphone off')
        self.resources = tk.StringVar()
        self.banner = tk.StringVar()
        status = ttk.Frame(body)
        status.grid(row=2, sticky='ew')
        ttk.Label(status, textvariable=self.status, style='State.TLabel').pack(side='left')
        ttk.Label(status, textvariable=self.server, style='Muted.TLabel').pack(side='right')
        message_label = ttk.Label(body, textvariable=self.message, wraplength=810)
        message_label.grid(row=3, sticky='ew', pady=(6, 10))
        controls = ttk.Frame(body)
        controls.grid(row=4, sticky='ew', pady=(0, 14))
        self.start_button = ttk.Button(controls, text='Start / Retry', command=self.start_services)
        self.start_button.pack(side='left', padx=(0, 8))
        self.pause_button = ttk.Button(controls, text='Pause', command=self.pause_resume)
        self.pause_button.pack(side='left', padx=(0, 8))
        self.cancel_button = ttk.Button(controls, text='Cancel reply', command=lambda: self.worker.send('cancel'))
        self.cancel_button.pack(side='left', padx=(0, 8))
        ttk.Button(controls, text='Clear conversation', command=lambda: self.worker.send('clear')).pack(side='left')
        ttk.Button(controls, text='Quit', command=self.close).pack(side='right')
        setup = ttk.LabelFrame(body, text='Microphone and sharing', padding=12)
        setup.grid(row=5, sticky='ew', pady=(0, 12))
        setup.columnconfigure(1, weight=1)
        self.device = tk.StringVar(value=self.preferences['device'])
        ttk.Label(setup, text='Microphone').grid(row=0, column=0, sticky='w', padx=(0, 12))
        self.device_select = ttk.Combobox(setup, textvariable=self.device, state='readonly')
        self.device_select.grid(row=0, column=1, sticky='ew')
        self.refresh_button = ttk.Button(setup, text='Refresh', command=self.refresh_devices)
        self.refresh_button.grid(row=0, column=2, padx=(8, 0))
        ttk.Label(setup, textvariable=self.mic_status, style='Muted.TLabel').grid(row=1, column=0, columnspan=2, sticky='w', pady=6)
        self.meter = ttk.Progressbar(setup, maximum=100)
        self.meter.grid(row=1, column=2, sticky='ew')
        self.sharing = tk.BooleanVar(value=False)
        ttk.Checkbutton(setup, text='Share primary screen with Gemma', variable=self.sharing,
                        command=self.share_screen).grid(row=2, column=0, columnspan=3, sticky='w')
        sharing_note = ttk.Label(setup, text='Applies to the next voice message. Earlier screenshots remain in context until you clear the conversation.',
                                 style='Muted.TLabel', wraplength=750)
        sharing_note.grid(row=3, column=0, columnspan=3, sticky='w', pady=(2, 6))
        options = ttk.Frame(setup)
        options.grid(row=4, column=0, columnspan=3, sticky='ew')
        self.voice = tk.BooleanVar(value=self.preferences['voice'])
        self.captions = tk.BooleanVar(value=self.preferences['captions'])
        ttk.Checkbutton(options, text='Speak replies', variable=self.voice).pack(side='left')
        ttk.Checkbutton(options, text='Floating captions', variable=self.captions).pack(side='left', padx=12)
        self.apply_button = ttk.Button(options, text='Apply setup', command=self.apply_setup)
        self.apply_button.pack(side='right')
        banner_label = ttk.Label(setup, textvariable=self.banner, foreground='#fbbf24', wraplength=760)
        banner_label.grid(row=5, column=0, columnspan=3, sticky='w', pady=(4, 0))
        body.bind('<Configure>', lambda e: [label.configure(wraplength=max(300, e.width - 80))
                                          for label in (message_label, sharing_note, banner_label)])
        ttk.Button(body, text='Advanced settings and diagnostics', command=self.toggle_advanced).grid(row=6, sticky='w', pady=(0, 8))
        self.advanced_container = ttk.Frame(body)
        self.advanced_container.columnconfigure(0, weight=1)
        self.advanced_container.rowconfigure(0, weight=1)
        self.advanced_canvas = tk.Canvas(self.advanced_container, bg='#111827', highlightthickness=0)
        self.advanced_canvas.grid(row=0, column=0, sticky='nsew')
        advanced_scroll = ttk.Scrollbar(self.advanced_container, command=self.advanced_canvas.yview)
        advanced_scroll.grid(row=0, column=1, sticky='ns')
        self.advanced_canvas.configure(yscrollcommand=advanced_scroll.set)
        self.advanced = ttk.Frame(self.advanced_canvas)
        advanced_window = self.advanced_canvas.create_window(0, 0, window=self.advanced, anchor='nw')
        self.advanced.bind('<Configure>', lambda _: self.advanced_canvas.configure(scrollregion=self.advanced_canvas.bbox('all')))
        self.advanced_canvas.bind('<Configure>', lambda e: self.advanced_canvas.itemconfigure(advanced_window, width=e.width))
        self.advanced.columnconfigure(1, weight=1)
        self.silence = tk.StringVar(value=str(self.preferences['silence']))
        self.threshold = tk.StringVar(value=str(self.preferences['threshold']))
        ttk.Label(self.advanced, text='Silence to end a turn (seconds)').grid(row=0, column=0, sticky='w')
        ttk.Entry(self.advanced, textvariable=self.silence, width=10).grid(row=0, column=1, sticky='w')
        ttk.Label(self.advanced, text='Speech energy threshold (lower = more sensitive)').grid(row=1, column=0, sticky='w')
        ttk.Entry(self.advanced, textvariable=self.threshold, width=10).grid(row=1, column=1, sticky='w')
        self.server_vars = {}
        for row, (key, label) in enumerate((('server', 'Server executable'), ('model', 'Model'),
                ('mmproj', 'Vision/audio model'), ('mtp', 'Draft model'), ('port', 'Port'),
                ('ctx', 'Context size'), ('gpu_layers', 'GPU layers')), start=2):
            self.server_vars[key] = tk.StringVar(value=str(getattr(self.args, key)))
            ttk.Label(self.advanced, text=label).grid(row=row, column=0, sticky='w')
            ttk.Entry(self.advanced, textvariable=self.server_vars[key]).grid(row=row, column=1, sticky='ew', pady=2)
            if key in ('server', 'model', 'mmproj', 'mtp'):
                ttk.Button(self.advanced, text='Browse', command=lambda k=key: self.browse(k)).grid(row=row, column=2, padx=(8, 0))
        self.no_mtp = tk.BooleanVar(value=self.args.no_mtp)
        ttk.Checkbutton(self.advanced, text='Disable draft model', variable=self.no_mtp).grid(row=9, column=0, sticky='w')
        ttk.Button(self.advanced, text='Apply server settings', command=self.apply_advanced).grid(row=9, column=1, sticky='e')
        self.diagnostics = tk.StringVar()
        ttk.Label(self.advanced, textvariable=self.diagnostics, style='Muted.TLabel', wraplength=780).grid(row=10, column=0, columnspan=3, sticky='ew')
        # Advanced controls replace the conversation while open to fit small displays.
        self.chat_frame = ttk.LabelFrame(body, text='Conversation', padding=8)
        self.chat_frame.grid(row=7, sticky='nsew')
        self.chat_frame.columnconfigure(0, weight=1)
        self.chat_frame.rowconfigure(0, weight=1)
        self.chat = tk.Text(self.chat_frame, wrap='word', state='disabled', font=('Segoe UI', 11),
                            bg='#182235', fg='#e5e7eb', selectbackground='#345078', relief='flat',
                            padx=12, pady=12, height=8, insertbackground='white')
        self.chat.grid(row=0, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(self.chat_frame, command=self.chat.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.chat.configure(yscrollcommand=scrollbar.set)
        self.chat.tag_configure('voice', foreground='#93c5fd', spacing1=12, spacing3=5)
        self.chat.tag_configure('muted', foreground='#aebbd0')
        ttk.Label(body, textvariable=self.resources, style='Muted.TLabel').grid(row=8, sticky='w', pady=(8, 0))
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
        if self.device.get() not in self.devices:
            raise ValueError('Select an available microphone, or refresh the device list.')
        return validate_preferences(dict(device=self.device.get(), silence=float(self.silence.get()),
            threshold=float(self.threshold.get()), voice=self.voice.get(), captions=self.captions.get()))

    def apply_setup(self):
        try:
            self.preferences = self.read_setup()
        except ValueError as exc:
            self.message.set(str(exc))
            return False
        self.worker.send('preferences', preferences=self.preferences)
        return True

    def start_services(self):
        if self.closing:
            return
        if self.running:
            if self.server_state == 'Unavailable' and self.apply_setup():
                self.message.set('Reconnecting local services…')
                self.worker.send('restart')
            elif self.state in ('Error', 'Paused') and self.apply_setup():
                self.worker.send('resume')
        elif self.apply_setup():
            self.message.set('Starting local services…')
            self.worker.send('start')

    def pause_resume(self):
        if self.state in ('Paused', 'Error'):
            if self.apply_setup():
                self.worker.send('resume')
        else:
            self.worker.send('pause')

    def share_screen(self):
        self.worker.send('screen_sharing', enabled=self.sharing.get())

    def toggle_advanced(self):
        if self.advanced_container.winfo_manager():
            self.advanced_container.grid_remove()
            self.chat_frame.grid()
        else:
            self.chat_frame.grid_remove()
            self.advanced_container.grid(row=7, sticky='nsew')

    def browse(self, key):
        chosen = filedialog.askopenfilename(parent=self.root, title=f'Select {key}')
        if chosen:
            self.server_vars[key].set(chosen)

    def apply_advanced(self):
        try:
            values = {key: (int(var.get()) if key in ('port', 'ctx', 'gpu_layers') else Path(var.get()))
                      for key, var in self.server_vars.items()}
            values['no_mtp'] = self.no_mtp.get()
            self.worker.send('advanced', values=values)
        except ValueError:
            self.message.set('Port, context size, and GPU layers must be whole numbers.')

    def render_conversation(self):
        if self.rendered_revision == self.conversation.revision:
            return
        scroll = self.chat.yview()
        follow = not self.conversation.turns or scroll[1] >= .98
        for button in self.copy_buttons:
            button.destroy()
        self.copy_buttons.clear()
        self.chat.configure(state='normal')
        self.chat.delete('1.0', 'end')
        if not self.conversation.turns:
            self.chat.insert('end', 'Speak when the microphone says Listening.\nEnable screen sharing when you want Gemma to see your screen.', 'muted')
        for turn in self.conversation.turns.values():
            self.chat.insert('end', 'Voice message' + (' · screen shared' if turn['screen'] else '') + '\n', 'voice')
            text = turn['text'] or ('No completed reply.' if turn['ended'] else 'Waiting for Gemma…')
            self.chat.insert('end', text + '\n')
            if turn['ended'] and not turn['complete']:
                self.chat.insert('end', 'Incomplete reply\n', 'muted')
            if turn['text'] and (turn['ended'] or turn['complete']):
                button = ttk.Button(self.chat, text='Copy reply', command=lambda t=turn['text']: self.copy_reply(t))
                self.copy_buttons.append(button)
                self.chat.window_create('end', window=button)
                self.chat.insert('end', '\n')
        self.chat.configure(state='disabled')
        if follow:
            self.chat.see('end')
        else:
            self.chat.yview_moveto(scroll[0])
        self.rendered_revision = self.conversation.revision

    def copy_reply(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

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
                self.message.set(event['detail'].replace('press Enter', 'click Start / Retry').replace('Enter retries, Q quits', 'Start / Retry retries; Quit closes the app'))
            elif changed and self.phase == 'Stopped' and event['detail'].startswith('Assistant exited'):
                self.message.set('Assistant stopped. Click Start / Retry to begin again.')
            if changed and self.running and self.server_state == 'Unavailable':
                self.message.set('Local server unavailable. Click Start / Retry to reconnect; this resets conversation context.')
            busy = self.state in ('Thinking', 'Preparing voice', 'Speaking')
            self.cancel_button.configure(state='normal' if busy else 'disabled')
            self.pause_button.configure(text='Resume' if self.state in ('Paused', 'Error') else 'Pause',
                                        state='normal' if self.running else 'disabled')
            self.start_button.configure(state='normal' if self.phase in ('Stopped', 'Error') or self.state == 'Error'
                                        or (self.running and self.server_state == 'Unavailable') else 'disabled')
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
        elif kind in ('preferences_applied', 'configured'):
            if event.get('warning'):
                self.banner.set(event['warning'])
            elif kind == 'preferences_applied':
                self.banner.set('')
                self.message.set('Setup saved.')
            elif kind == 'configured':
                self.banner.set('' if event.get('voice_available') else 'Voice output is off. Replies appear in the conversation.')
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
                    self.message.set(error or 'Saved microphone is unavailable. Select a microphone and apply setup.')
                    self.auto_start = False
                elif error:
                    self.message.set(error)
                elif not selected:
                    if devices:
                        self.device.set(self.args.device if self.args.device in devices else devices[0])
                    self.message.set('Select your microphone, then click Start / Retry to begin.')
                elif self.auto_start:
                    self.start_services()
                self.auto_start = False
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
        self.closing = True
        self.message.set('Closing the assistant and services started by this app…')
        for button in (self.start_button, self.pause_button, self.cancel_button, self.apply_button, self.refresh_button):
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
