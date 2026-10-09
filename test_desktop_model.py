"""Opt-in local-model checks. Starts an owned server if necessary and releases it.

Uses the existing test recording; never opens the microphone. The screen-sharing
case captures the primary screen. GEMMA_MODEL_TESTS=1 enables these checks.
"""
import os
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
import wave

import launch_gemma as launch
import live_gemma as live
import desktop_app as desktop


@unittest.skipUnless(os.environ.get('GEMMA_MODEL_TESTS') == '1', 'loads the local model')
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recording = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        cls.addClassCleanup(cls.recording.cleanup)
        cls.audio_path = Path(cls.recording.name) / 'speech.wav'
        # test.wav is almost silent; test_speech.wav contains the existing voice
        # sample but needs resampling from 22050 Hz to the capture format.
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i',
                        str(Path(live.MEDIA_DIR) / 'test_speech.wav'), '-ac', '1', '-ar', '16000',
                        str(cls.audio_path)], check=True, capture_output=True, timeout=10,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        cls.supervisor = launch.Supervisor(launch.parse_args(['--no-overlay', '--no-tts']))
        cls.addClassCleanup(cls.supervisor.close)
        cls.supervisor.launch_assistant = lambda: setattr(cls.supervisor, 'phase', 'Verified ready')
        cls.supervisor.start()
        deadline = time.monotonic() + 180
        while cls.supervisor.phase == 'Loading' and time.monotonic() < deadline:
            cls.supervisor.tick()
            time.sleep(.1)
        if cls.supervisor.phase != 'Verified ready':
            raise RuntimeError(cls.supervisor.notice)

    def run_turn(self, sharing, voice=False):
        args = SimpleNamespace(device='unused', silence=1, threshold=.01, no_tts=not voice,
            test=True, server_url=self.supervisor.url, piper_path=None, screen_sharing=sharing,
            speech_mode='streamed')
        warning = live.configure_voice(args)
        if voice:
            self.assertFalse(args.no_tts, warning)
        controller = live.LiveController(args)
        captured = []
        original_emit = controller.emit
        def emit(event, **fields):
            captured.append(dict(event=event, **fields))
            original_emit(event, **fields)
        controller.emit = emit
        output = io.StringIO()
        redirect = redirect_stdout(output)
        redirect.__enter__()
        try:
            with wave.open(str(self.audio_path)) as source:
                self.assertEqual((source.getnchannels(), source.getsampwidth(), source.getframerate()), (1, 2, 16000))
                controller.start_turn(source.readframes(source.getnframes()))
            self.assertEqual(len(controller.active.paths), 2 if sharing else 1)
            deadline = time.monotonic() + 90
            while controller.active and time.monotonic() < deadline:
                controller.tick()
                time.sleep(.03)
            self.assertIsNone(controller.active, controller.notice)
            self.assertTrue(controller.last_success, controller.notice)
            self.assertTrue(any(e['event'] == 'reply' and e['text'].strip() for e in captured))
            self.assertFalse(any(e['event'] == 'failure' for e in captured), captured)
            controller.control('clear')
            self.assertEqual(len(controller.messages), 1)
            self.assertFalse(controller.owned)
        finally:
            controller.close()
            redirect.__exit__(None, None, None)

    def test_voice_only_reply_and_media_cleanup(self):
        self.run_turn(False)

    def test_screen_shared_reply_and_media_cleanup(self):
        self.run_turn(True)

    @unittest.skipUnless(os.environ.get('GEMMA_PLAYBACK_TESTS') == '1', 'plays a spoken reply')
    def test_streamed_spoken_reply(self):
        self.run_turn(False, voice=True)

    @unittest.skipUnless(os.environ.get('GEMMA_DESKTOP_SESSION_TESTS') == '1', 'opens the app and microphone')
    def test_desktop_start_pause_resume_and_owned_cleanup(self):
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(directory.cleanup)
        root = desktop.tk.Tk()
        app = desktop.DesktopApp(root, launch.parse_args(['--wait', '--threshold', '0.5',
            '--port', str(self.supervisor.args.port), '--settings-file', str(Path(directory.name) / 'settings.json')]))
        captured = []
        original = app.handle_event
        def handle(event):
            captured.append(event)
            original(event)
        app.handle_event = handle
        def wait_until(predicate, seconds=20):
            deadline = time.monotonic() + seconds
            while not predicate() and time.monotonic() < deadline:
                root.update()
                time.sleep(.03)
            self.assertTrue(predicate(), app.message.get())
        try:
            wait_until(lambda: not app.refreshing)
            self.assertTrue(app.devices, app.message.get())
            app.start_services()
            wait_until(lambda: app.state == 'Listening')
            wait_until(lambda: any(e['event'] == 'microphone' for e in captured))
            app.sharing.set(True)
            app.share_screen()
            wait_until(lambda: any(e.get('event') == 'screen_sharing' and e.get('enabled') for e in captured))
            app.pause_resume()
            wait_until(lambda: app.state == 'Paused')
            app.worker.send('clear')
            wait_until(lambda: any(e['event'] == 'cleared' for e in captured))
            self.assertEqual(app.state, 'Paused')
            app.pause_resume()
            wait_until(lambda: app.state == 'Listening')
            self.assertIsNone(app.worker.supervisor.server)  # external server is reused
            app.close()
            wait_until(app.worker.closed.is_set)
            self.assertEqual(launch.health(self.supervisor.url), 'Ready')
        finally:
            if not app.closing:
                app.close()
            app.worker.thread.join(12)
            try:
                root.destroy()
            except desktop.tk.TclError:
                pass


if __name__ == '__main__':
    unittest.main()
