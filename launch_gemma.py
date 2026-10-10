"""Gemma Live service supervisor with desktop and terminal displays."""
import argparse
from collections import deque
import ctypes
import json
import math
import os
from pathlib import Path
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = Path.home() / '.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF'
STATES = ('Starting', 'Listening', 'Thinking', 'Preparing voice', 'Speaking', 'Paused', 'Error')
CONTROL_BAR = '[ Enter: Start all ]  [ P: Pause/Resume ]  [ C: Cancel reply ]  [ Q: Quit ]'


def clicked_control(x, y):
    if y != 2:
        return None
    for label, key in (('[ Enter: Start all ]', '\r'), ('[ P: Pause/Resume ]', 'p'),
                       ('[ C: Cancel reply ]', 'c'), ('[ Q: Quit ]', 'q')):
        start = CONTROL_BAR.index(label)
        if start <= x < start + len(label):
            return key
    return None


class TerminalInput:
    """Windows console keyboard and mouse buttons, including Windows Terminal."""
    def __init__(self):
        from ctypes import wintypes as w
        class Coord(ctypes.Structure):
            _fields_ = [('x', w.SHORT), ('y', w.SHORT)]
        class Key(ctypes.Structure):
            _fields_ = [('down', w.BOOL), ('repeat', w.WORD), ('key', w.WORD),
                        ('scan', w.WORD), ('char', w.WCHAR), ('controls', w.DWORD)]
        class Mouse(ctypes.Structure):
            _fields_ = [('position', Coord), ('buttons', w.DWORD),
                        ('controls', w.DWORD), ('flags', w.DWORD)]
        class Event(ctypes.Union):
            _fields_ = [('key', Key), ('mouse', Mouse), ('padding', ctypes.c_byte * 16)]
        class Record(ctypes.Structure):
            _fields_ = [('kind', w.WORD), ('event', Event)]
        self.Record = Record
        self.kernel = ctypes.windll.kernel32
        self.kernel.GetStdHandle.restype = w.HANDLE
        self.kernel.GetConsoleMode.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
        self.kernel.SetConsoleMode.argtypes = [w.HANDLE, w.DWORD]
        self.kernel.GetNumberOfConsoleInputEvents.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
        self.kernel.ReadConsoleInputW.argtypes = [w.HANDLE, ctypes.POINTER(Record), w.DWORD, ctypes.POINTER(w.DWORD)]
        self.handle = self.kernel.GetStdHandle(-10)
        self.mode = w.DWORD()
        self.kernel.GetConsoleMode(self.handle, ctypes.byref(self.mode))
        self.kernel.SetConsoleMode(self.handle, (self.mode.value | 0x98) & ~0x46)

    def read(self):
        count = ctypes.c_ulong()
        while self.kernel.GetNumberOfConsoleInputEvents(self.handle, ctypes.byref(count)) and count.value:
            record = self.Record()
            if not self.kernel.ReadConsoleInputW(self.handle, ctypes.byref(record), 1, ctypes.byref(count)):
                break
            if record.kind == 1 and record.event.key.down:
                return record.event.key.char.lower()
            if record.kind == 2:
                mouse = record.event.mouse
                if mouse.flags == 0 and mouse.buttons & 1:
                    return clicked_control(mouse.position.x, mouse.position.y)
        return None

    def close(self):
        self.kernel.SetConsoleMode(self.handle, self.mode.value)


def server_command(args):
    command = [str(args.server), '-m', str(args.model), '-mm', str(args.mmproj),
               '--media-path', str(ROOT / 'media'), '-c', str(args.ctx), '-ngl', str(args.gpu_layers),
               '--load-mode', 'mmap',
               '-fa', 'auto', '--reasoning-effort', 'none', '--port', str(args.port), '--host', '127.0.0.1']
    if not args.no_mtp and args.mtp.is_file():
        command += ['--spec-type', 'draft-mtp', '--spec-draft-model', str(args.mtp), '--threads-draft', '8']
    return command


def health(url):
    try:
        with urllib.request.urlopen(url + '/health', timeout=0.3) as response:
            return 'Ready' if json.load(response).get('status') == 'ok' else 'Loading'
    except urllib.error.HTTPError as exc:
        result = 'Loading' if exc.code == 503 else 'Unavailable'
        exc.close()
        return result
    except (OSError, ValueError):
        return 'Unavailable'


def port_in_use(port):
    with socket.socket() as connection:
        connection.settimeout(0.1)
        return connection.connect_ex(('127.0.0.1', port)) == 0


def memory_status():
    if os.name != 'nt':
        return 'Not available'
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total', 'page_available',
                                                  'virtual_total', 'virtual_available', 'extended')]
    info = Memory()
    info.length = ctypes.sizeof(info)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(info)):
        return f'{info.available / 2**30:.1f} GB free / {info.total / 2**30:.1f} GB'
    return 'Not available'


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.url = f'http://127.0.0.1:{args.port}'
        self.server = None
        self.assistant = None
        self.server_log = None
        self.events = queue.Queue()
        self.updates = queue.Queue()
        self.logs = deque(maxlen=6)
        self.checks = {}
        self.state = 'Stopped'
        self.server_state = 'Not checked'
        self.phase = 'Stopped'
        self.deadline = 0
        self.next_health = 0
        self.last_draw = 0
        self.quitting = False
        self.notice = 'Press Enter to start all services.'
        self.reader = None
        self.session = 0

    def preflight(self):
        # Import here so a missing dependency is reported inside the dashboard.
        try:
            import live_gemma as live
        except ImportError as exc:
            self.checks = {'Python dependencies': f'MISSING: {exc}. Install requirements.txt.'}
            return False
        self.checks = {'Python': sys.version.split()[0],
                       'Microphone': self.args.device,
                       'FFmpeg': 'OK' if shutil.which('ffmpeg') else 'MISSING',
                       'Piper': 'OK' if live.resolve_piper(self.args.piper_path) else 'Unavailable (captions only)',
                       'Playback': 'OK' if shutil.which('ffplay') else 'Unavailable (captions only)',
                       'Voice model': 'OK' if all((ROOT / 'voices' / (live.PIPER_MODEL + ext)).is_file()
                                                 for ext in ('.onnx', '.onnx.json')) else 'Unavailable (captions only)'}
        return getattr(self.args, 'text_only', False) or self.checks['FFmpeg'] == 'OK'

    def start(self):
        if self.quitting:
            return
        if self.phase not in ('Stopped', 'Error'):
            return
        if self.assistant and self.assistant.poll() is None:
            return
        self.state = 'Starting'
        if not self.preflight():
            self.phase = self.state = 'Error'
            self.notice = 'Fix the missing dependencies above, then press Enter to retry.'
            return
        self.server_state = health(self.url)
        if self.server_state == 'Ready':
            self.launch_assistant()
            return
        if self.server_state == 'Loading':
            self.phase = 'Loading'
            self.deadline = time.monotonic() + self.args.startup_timeout
            self.notice = 'Waiting for the existing server to finish loading.'
            return
        if self.server and self.server.poll() is None:
            self.phase = 'Loading'
            self.deadline = time.monotonic() + self.args.startup_timeout
            return
        if port_in_use(self.args.port):
            self.phase = self.state = 'Error'
            self.notice = f'Port {self.args.port} is occupied but is not a ready server. Choose another --port.'
            return
        for label, path in (('Server executable', self.args.server), ('Model', self.args.model), ('Vision/audio model', self.args.mmproj)):
            self.checks[label] = 'OK' if path.is_file() else f'MISSING: {path}'
        if any(value.startswith('MISSING') for value in self.checks.values()):
            self.phase = self.state = 'Error'
            self.notice = 'Fix missing model/server paths, then press Enter to retry.'
            return
        try:
            (ROOT / 'media').mkdir(exist_ok=True)
            if self.server_log:
                self.server_log.close()
            self.server_log = open(ROOT / 'launcher-server.log', 'ab', buffering=0)
            self.server = subprocess.Popen(server_command(self.args), cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=self.server_log, stderr=self.server_log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except OSError as exc:
            self.phase = self.state = 'Error'
            self.notice = f'Server could not start: {exc}'
            return
        self.phase = 'Loading'
        self.deadline = time.monotonic() + self.args.startup_timeout
        self.notice = 'Loading the local model. Controls remain available.'

    def launch_assistant(self):
        command = [sys.executable, '-u', str(ROOT / 'live_gemma.py'), '--terminal-controls',
                   '--server-url', self.url, '--device', self.args.device,
                   '--speech-mode', self.args.speech_mode,
                   '--silence', str(self.args.silence), '--threshold', str(self.args.threshold)]
        if getattr(self.args, 'text_only', False):
            command.append('--text-only')
        command += ['--caption-position', getattr(self.args, 'caption_position', 'Top left')]
        if getattr(self.args, 'desktop', False):
            command.append('--captions-only')
        if getattr(self.args, 'screen_sharing', False):
            command.append('--screen-sharing')
        if self.args.no_overlay:
            command.append('--no-overlay')
        if self.args.no_tts:
            command.append('--no-tts')
        if self.args.piper_path:
            command += ['--piper-path', self.args.piper_path]
        try:
            self.assistant = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except OSError as exc:
            self.phase = self.state = 'Error'
            self.notice = f'Assistant could not start: {exc}'
            return
        self.phase = 'Running'
        self.session += 1
        session, process = self.session, self.assistant
        def read_output():
            with process.stdout:
                for line in process.stdout:
                    self.events.put((session, line.rstrip()))
        self.reader = threading.Thread(target=read_output, daemon=True)
        self.reader.start()
        self.notice = 'Ready. Speak when the assistant says Listening.'

    def send(self, action):
        if self.assistant and self.assistant.poll() is None:
            try:
                self.assistant.stdin.write((json.dumps(action) if isinstance(action, dict) else action) + '\n')
                self.assistant.stdin.flush()
                return True
            except (OSError, ValueError):
                pass
        return False

    def tick(self):
        while True:
            try:
                session, line = self.events.get_nowait()
            except queue.Empty:
                break
            if session != self.session:
                continue
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                event = None
            if isinstance(event, dict) and event.get('event') == 'status':
                if event.get('state') in STATES and isinstance(event.get('detail'), str):
                    self.state = event['state']
                    self.notice = event['detail']
            if isinstance(event, dict) and event.get('event') in (
                    'status', 'turn_started', 'partial', 'reply', 'turn_finished', 'failure',
                    'microphone', 'screen_sharing', 'cleared', 'configured'):
                self.updates.put(event)
            elif not isinstance(event, dict) or event.get('event') != 'timing':
                self.logs.append(line)
        now = time.monotonic()
        if now >= self.next_health:
            self.server_state = health(self.url)
            self.next_health = now + 2
        if self.phase == 'Loading':
            if self.server_state == 'Ready':
                self.launch_assistant()
            elif self.server and self.server.poll() is not None:
                self.phase = self.state = 'Error'
                self.notice = 'Server exited. Read launcher-server.log, then press Enter to retry.'
            elif now > self.deadline:
                self.phase = self.state = 'Error'
                self.notice = 'Server startup timed out. Check launcher-server.log; Enter retries, Q quits.'
        elif self.phase == 'Running' and self.assistant.poll() is not None:
            code = self.assistant.returncode
            self.assistant.stdin.close()
            self.phase = 'Stopped' if code == 0 else 'Error'
            self.state = self.phase
            self.notice = f'Assistant exited ({code}). Enter starts again; Q closes owned services.'
            self.updates.put(dict(event='assistant_stopped', code=code))
        if now - self.last_draw >= 0.25:
            self.draw()
            self.last_draw = now

    def draw(self):
        pass

    def stop_assistant(self):
        self.send('quit')
        if self.assistant:
            try:
                self.assistant.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # This PID is exclusively owned by this launcher; include its stuck children.
                subprocess.run(['taskkill', '/PID', str(self.assistant.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                self.assistant.wait(timeout=3)
            if self.assistant.stdin and not self.assistant.stdin.closed:
                self.assistant.stdin.close()
            self.assistant = None
            self.session += 1
            self.updates.put(dict(event='assistant_stopped', code=0))

    def restart(self):
        self.stop_assistant()
        self.phase = self.state = 'Stopped'
        self.start()

    def close(self):
        self.quitting = True
        self.stop_assistant()
        if self.server and self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait(timeout=3)
        if self.server_log:
            self.server_log.close()


class Dashboard(Supervisor):
    """Optional terminal display using the same service ownership rules."""

    def draw(self):
        while not self.updates.empty():
            self.updates.get_nowait()
        width = max(40, min(100, shutil.get_terminal_size((90, 30)).columns - 1))
        owned = 'started here' if self.server and self.server.poll() is None else 'external / not running'
        lines = ['GEMMA LIVE | Local voice + screen assistant', '=' * width,
                 CONTROL_BAR, '',
                 f'Assistant: {self.state}', f'Server: {self.server_state} | {self.url} | {owned}',
                 f'Memory: {memory_status()}', f'Disk: {shutil.disk_usage(ROOT).free / 2**30:.1f} GB free', '']
        lines += [f'{key}: {value}' for key, value in self.checks.items()]
        lines += ['', self.notice, '', 'Recent activity:'] + list(self.logs)
        clean = [''.join(c for c in line if c.isprintable())[:width] for line in lines]
        clean = clean[:max(1, shutil.get_terminal_size((90, 30)).lines - 1)]
        if sys.stdout.isatty():
            print('\x1b[H' + '\n'.join(line + '\x1b[K' for line in clean) + '\x1b[J', end='', flush=True)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--ctx', type=int, default=32768)
    parser.add_argument('--gpu-layers', type=int, default=99)
    server_candidates = [ROOT / 'llama-server.exe', ROOT.parent / 'llama-server.exe', ROOT.parent.parent / 'llama-server.exe']
    parser.add_argument('--server', type=Path, default=next((path for path in server_candidates if path.is_file()), server_candidates[1]))
    parser.add_argument('--model', type=Path, default=DEFAULT_MODELS / 'gemma-4-12B-it-QAT-Q4_0.gguf')
    parser.add_argument('--mmproj', type=Path, default=DEFAULT_MODELS / 'mmproj-gemma-4-12B-it-QAT-BF16.gguf')
    parser.add_argument('--mtp', type=Path, default=DEFAULT_MODELS / 'gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf')
    parser.add_argument('--no-mtp', action='store_true')
    parser.add_argument('--no-overlay', action='store_true')
    parser.add_argument('--overlay', dest='no_overlay', action='store_false')
    parser.add_argument('--speech-mode', choices=('streamed', 'full'), default='streamed')
    parser.add_argument('--no-tts', action='store_true')
    parser.add_argument('--tts', dest='no_tts', action='store_false')
    parser.add_argument('--piper-path')
    parser.add_argument('--text-only', action='store_true', help='Start a text session without opening a microphone')
    parser.add_argument('--device', default='Microphone (fifine Microphone)')
    parser.add_argument('--silence', type=float, default=1.0)
    parser.add_argument('--threshold', type=float, default=0.01)
    parser.add_argument('--terminal', action='store_true', help='Use the terminal dashboard')
    parser.add_argument('--settings-file', type=Path, help='Override the per-user settings file')
    parser.add_argument('--screen-sharing', action='store_true', help='Enable screen context in terminal mode')
    parser.add_argument('--startup-timeout', type=float, default=180)
    parser.add_argument('--check', action='store_true', help='Print system checks without starting services')
    parser.add_argument('--wait', action='store_true', help='Wait for Start/Enter instead of starting automatically')
    args = parser.parse_args(argv)
    passed = argv if argv is not None else sys.argv[1:]
    flags = {arg.split('=')[0] for arg in passed}
    args.preference_overrides = {key for key, options in {
        'device': ('--device',), 'silence': ('--silence',), 'threshold': ('--threshold',),
        'voice': ('--tts', '--no-tts'), 'captions': ('--overlay', '--no-overlay')
    }.items() if flags.intersection(options)}
    args.desktop = not args.terminal
    if (not 1 <= args.port <= 65535 or args.ctx <= 0
            or not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0):
        parser.error('Port, context size, and startup timeout must be positive and valid')
    if not math.isfinite(args.silence) or not 0.1 <= args.silence <= 10:
        parser.error('Silence must be between 0.1 and 10 seconds')
    if not math.isfinite(args.threshold) or not 0.0001 <= args.threshold <= 0.5:
        parser.error('Threshold must be between 0.0001 and 0.5')
    return args


def main():
    args = parse_args()
    dashboard = Dashboard(args)
    if args.check:
        ok = dashboard.preflight()
        print(json.dumps({'checks': dashboard.checks, 'server': health(dashboard.url),
                          'memory': memory_status(),
                          'model_files': {str(path): path.is_file() for path in (args.server, args.model, args.mmproj)}}, indent=2))
        return 0 if ok else 1
    if not args.terminal:
        from desktop_app import run_desktop
        return run_desktop(args)
    if os.name != 'nt' or not sys.stdin.isatty():
        print('Open Start Gemma.cmd in a Windows terminal, or use --check for diagnostics.')
        return 1
    kernel = ctypes.windll.kernel32
    kernel.GetStdHandle.restype = ctypes.c_void_p
    kernel.GetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.SetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    handle = kernel.GetStdHandle(-11)
    mode = ctypes.c_ulong()
    kernel.GetConsoleMode(handle, ctypes.byref(mode))
    kernel.SetConsoleMode(handle, mode.value | 4)
    print('\x1b[2J\x1b[H\x1b[?25l', end='')
    terminal_input = TerminalInput()
    try:
        dashboard.preflight()
        if not args.wait:
            dashboard.start()
        while not dashboard.quitting:
            key = terminal_input.read()
            if key:
                if key in ('q', '\x03'):
                    break
                if key == '\r':
                    dashboard.start()
                elif key == 'p':
                    dashboard.send('resume' if dashboard.state in ('Paused', 'Error') else 'pause')
                elif key == 'c':
                    dashboard.send('cancel')
            dashboard.tick()
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.close()
        terminal_input.close()
        print('\x1b[?25h\nGemma Live stopped. External servers were left running.')
        kernel.SetConsoleMode(handle, mode.value)
    return 0


if __name__ == '__main__':
    sys.exit(main())
