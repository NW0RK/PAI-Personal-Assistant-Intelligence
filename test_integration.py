"""Opt-in checks that open desktop controls and use the running local model.
Run with GEMMA_INTEGRATION=1. Ordinary unit-test discovery skips these checks.
"""
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
import wave
import live_gemma as live


@unittest.skipUnless(os.environ.get('GEMMA_INTEGRATION') == '1', 'requires desktop and running model')
class DesktopIntegrationTests(unittest.TestCase):
    def test_launcher_control_pipe_does_not_block_inference_start(self):
        # Keep stdin OPEN and idle: communicate() would close it and miss the
        # Windows buffered-reader/spawn deadlock present in the launcher path.
        process = subprocess.Popen([sys.executable, '-u', 'live_gemma.py',
            '--terminal-controls', '--test', '--test-audio', 'media/test.wav', '--no-tts'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', cwd=Path(__file__).parent,
            creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            process.wait(timeout=25)
            output = process.stdout.read()
            self.assertEqual(process.returncode, 0, output)
            self.assertIn('[Gemma]', output)
            self.assertIn('Waiting for the local model', output)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)
            process.stdin.close()
            process.stdout.close()

    def test_native_terminal_keyboard_and_mouse(self):
        code = r'''
import ctypes
import launch_gemma as launch
kernel = ctypes.windll.kernel32
kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p,
                              ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p]
kernel.CreateFileW.restype = ctypes.c_void_p
kernel.SetStdHandle.argtypes = [ctypes.c_ulong, ctypes.c_void_p]
console_input = kernel.CreateFileW('CONIN$', 0xC0000000, 3, None, 3, 0, None)
assert kernel.SetStdHandle(-10, console_input)
controls = launch.TerminalInput()
kernel.FlushConsoleInputBuffer.argtypes = [ctypes.c_void_p]
kernel.WriteConsoleInputW.argtypes = [ctypes.c_void_p, ctypes.POINTER(controls.Record), ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong)]
try:
    kernel.FlushConsoleInputBuffer(controls.handle)
    record = controls.Record()
    record.kind = 1
    record.event.key.down = True
    record.event.key.char = 'p'
    count = ctypes.c_ulong()
    assert kernel.WriteConsoleInputW(controls.handle, ctypes.byref(record), 1, ctypes.byref(count))
    assert controls.read() == 'p'
    record = controls.Record()
    record.kind = 2
    record.event.mouse.position.x = launch.CONTROL_BAR.index('Start all')
    record.event.mouse.position.y = 2
    record.event.mouse.buttons = 1
    assert kernel.WriteConsoleInputW(controls.handle, ctypes.byref(record), 1, ctypes.byref(count))
    assert controls.read() == '\r'
finally:
    controls.close()
'''
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                timeout=10, cwd=Path(__file__).parent,
                                creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_turn(self):
        args = SimpleNamespace(device=live.DEFAULT_DEVICE, silence=1, threshold=.01,
                               no_tts=True, test=True, server_url=live.SERVER_URL, piper_path=None)
        controller = live.LiveController(args)
        try:
            with wave.open(str(Path(live.MEDIA_DIR) / 'test.wav')) as audio:
                self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, 16000))
                controller.start_turn(audio.readframes(audio.getnframes()))
            deadline = time.monotonic() + 45
            while controller.active and time.monotonic() < deadline:
                controller.tick()
                time.sleep(.03)
            self.assertTrue(controller.last_success, controller.notice)
            self.assertIsNone(controller.active)
            self.assertEqual(len(controller.messages), 3)
        finally:
            controller.close()
        self.assertFalse(controller.owned)

    def test_tk_buttons_emit_control_protocol(self):
        code = r'''
import io, sys, threading
import desktop_overlay as overlay
original = overlay.tk.Tk
class Input:
    def __iter__(self):
        yield '{"action":"status","state":"Thinking","detail":"Testing controls"}\n'
        yield '{"action":"show","text":"Streamed caption"}\n'
        threading.Event().wait(5)
sys.stdin = Input()
def root_factory():
    root = original()
    def check():
        try:
            panel = next(w for w in root.winfo_children() if isinstance(w, overlay.tk.Toplevel))
            frame = next(w for w in panel.winfo_children() if isinstance(w, overlay.tk.Frame))
            buttons = frame.winfo_children()
            assert [w.cget('text') for w in buttons] == ['Pause', 'Cancel reply', 'Quit']
            assert str(buttons[1].cget('state')) == 'normal'
            for button in buttons:
                button.invoke()
        except Exception as exc:
            print('TEST_FAILURE:' + repr(exc), flush=True)
        finally:
            root.destroy()
    root.after(300, check)
    return root
overlay.tk.Tk = root_factory
overlay.run_overlay()
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                timeout=10, cwd=Path(__file__).parent)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('TEST_FAILURE', result.stdout)
        for action in ('pause', 'cancel', 'quit'):
            self.assertIn('"action": "' + action + '"', result.stdout)


if __name__ == '__main__':
    unittest.main()
