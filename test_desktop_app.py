import json
import os
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import app_settings as settings
import desktop_app as desktop
import launch_gemma as launch


class PreferencesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'settings.json'

    def test_round_trip_does_not_persist_screen_sharing(self):
        values = dict(settings.DEFAULTS, device='USB microphone', threshold=.02, screen_sharing=True)
        settings.save_preferences(values, self.path)
        actual, warning = settings.load_preferences(self.path)
        self.assertEqual(actual['device'], 'USB microphone')
        self.assertEqual(actual['threshold'], .02)
        self.assertNotIn('screen_sharing', actual)
        self.assertEqual(warning, '')

    def test_invalid_saved_settings_restore_defaults_with_warning(self):
        for content in ('broken', '[]', '{"version":2}',
                        '{"version":1,"preferences":{"threshold":NaN}}',
                        '{"version":1,"preferences":{"voice":"yes"}}'):
            self.path.write_text(content)
            values, warning = settings.load_preferences(self.path)
            self.assertEqual(values, settings.DEFAULTS)
            self.assertIn('defaults restored', warning)

    def test_invalid_write_leaves_good_settings_intact(self):
        settings.save_preferences(settings.DEFAULTS, self.path)
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            settings.save_preferences(dict(settings.DEFAULTS, silence=-1), self.path)
        self.assertEqual(before, self.path.read_bytes())

    def test_cli_overrides_saved_preferences(self):
        settings.save_preferences(dict(settings.DEFAULTS, device='Saved', voice=False), self.path)
        args = launch.parse_args(['--settings-file', str(self.path), '--device=Chosen', '--tts', '--silence', '0.4'])
        values, _ = desktop.initial_preferences(args)
        self.assertEqual(values['device'], 'Chosen')
        self.assertTrue(values['voice'])
        self.assertEqual(values['silence'], .4)
        args = launch.parse_args(['--settings-file', str(self.path)])
        values, _ = desktop.initial_preferences(args)
        self.assertEqual(values['device'], 'Saved')
        self.assertFalse(values['voice'])

    def test_audio_discovery_ignores_video_and_alternative_names(self):
        output = '''[dshow] "Camera" (video)
[dshow] "USB Microphone" (audio)
[dshow] Alternative name "@device:other"
[dshow] "USB Microphone" (audio)
[dshow] "Second microphone" (audio)'''
        self.assertEqual(settings.parse_microphones(output), ['USB Microphone', 'Second microphone'])
        with patch.object(settings.subprocess, 'run', return_value=Mock(stderr=output, stdout='', returncode=1)):
            self.assertEqual(settings.discover_microphones()[0], ['USB Microphone', 'Second microphone'])

    def test_discovery_failure_is_actionable(self):
        with patch.object(settings.subprocess, 'run', side_effect=FileNotFoundError):
            devices, warning = settings.discover_microphones()
        self.assertEqual(devices, [])
        self.assertIn('FFmpeg', warning)


class ConversationTests(unittest.TestCase):
    def test_partial_completion_and_interruption(self):
        model = desktop.Conversation()
        model.handle(dict(event='turn_started', turn_id='a', screen_sharing=False))
        model.handle(dict(event='partial', turn_id='a', text='Hello'))
        self.assertFalse(model.turns['a']['complete'])
        model.handle(dict(event='reply', turn_id='a', text='Hello there', complete=True))
        model.handle(dict(event='turn_finished', turn_id='a', text='Hello there', complete=True))
        model.handle(dict(event='partial', turn_id='a', text='stale'))
        self.assertEqual(model.turns['a']['text'], 'Hello there')
        model.handle(dict(event='turn_started', turn_id='b'))
        model.handle(dict(event='partial', turn_id='b', text='Interrupted'))
        model.handle(dict(event='assistant_stopped'))
        self.assertTrue(model.turns['b']['ended'])
        self.assertFalse(model.turns['b']['complete'])

    def test_clear_rejects_late_events(self):
        model = desktop.Conversation()
        model.handle(dict(event='turn_started', turn_id='a'))
        model.handle(dict(event='cleared'))
        model.handle(dict(event='reply', turn_id='a', text='late', complete=True))
        self.assertEqual(model.turns, {})


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.directory.cleanup)
        self.args = launch.parse_args(['--settings-file', str(Path(self.directory.name) / 'settings.json')])
        self.preferences = dict(settings.DEFAULTS, device='Test microphone')
        self.worker = desktop.SupervisorWorker(self.args, self.preferences)

    def test_settings_and_sharing_follow_assistant_restart(self):
        self.worker.supervisor.send = Mock()
        self.worker.supervisor.phase = 'Running'
        self.worker.handle(dict(action='screen_sharing', enabled=True))
        values = dict(self.preferences, silence=.5, captions=False)
        self.worker.handle(dict(action='preferences', preferences=values))
        self.assertTrue(self.worker.args.screen_sharing)
        self.assertTrue(self.worker.args.no_overlay)
        self.worker.supervisor.send.assert_called_with(dict(action='configure', preferences=values))
        self.assertEqual(settings.load_preferences(self.args.settings_file)[0], values)
        restarted = desktop.SupervisorWorker(self.args, values)
        self.assertFalse(restarted.args.screen_sharing)

    def test_slow_health_does_not_block_desktop_command_submission(self):
        entered = threading.Event()
        release = threading.Event()
        def slow_tick():
            entered.set()
            release.wait(2)
        self.worker.supervisor.tick = slow_tick
        self.worker.supervisor.close = Mock()
        self.worker.start()
        try:
            self.assertTrue(entered.wait(1))
            started = time.monotonic()
            self.worker.send('quit')
            self.assertLess(time.monotonic() - started, .1)
        finally:
            release.set()
            self.worker.thread.join(3)
        self.assertTrue(self.worker.closed.is_set())
        self.worker.supervisor.close.assert_called_once()

    def test_invalid_advanced_settings_do_not_change_args(self):
        values = dict(port=0, ctx=100, gpu_layers=2)
        with self.assertRaises(ValueError):
            self.worker.handle(dict(action='advanced', values=values))
        self.assertEqual(self.worker.args.port, 8080)

    def test_clear_offline_clears_display(self):
        self.worker.handle(dict(action='clear'))
        self.assertEqual(self.worker.output.get_nowait(), dict(event='cleared'))


@unittest.skipUnless(os.environ.get('GEMMA_DESKTOP_TESTS') == '1', 'opens the desktop window')
class WindowTests(unittest.TestCase):
    def test_caption_process_exit_does_not_quit_desktop_assistant(self):
        from desktop_overlay import SpeechOverlay
        overlay = SpeechOverlay(captions_only=True)
        try:
            overlay.process.terminate()
            overlay.process.wait(timeout=3)
            overlay.reader.join(timeout=1)
            self.assertEqual(overlay.poll_controls(), [])
        finally:
            overlay.close()

    def test_window_controls_missing_microphone_captions_and_shutdown(self):
        class FakeWorker:
            def __init__(self, args, preferences):
                self.output = queue.Queue()
                self.closed = threading.Event()
                self.commands = []
            def start(self):
                pass
            def send(self, action, **fields):
                self.commands.append((action, fields))
                if action == 'quit':
                    self.closed.set()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / 'settings.json'
            settings.save_preferences(dict(settings.DEFAULTS, device='Missing'), path)
            args = launch.parse_args(['--settings-file', str(path), '--wait'])
            root = desktop.tk.Tk()
            try:
                app = desktop.DesktopApp(root, args, worker_factory=FakeWorker,
                                         discover=lambda: (['Available'], ''))
                def wait_until(predicate):
                    deadline = time.monotonic() + 2
                    while not predicate() and time.monotonic() < deadline:
                        root.update()
                        time.sleep(.01)
                    self.assertTrue(predicate())
                wait_until(lambda: not app.refreshing)
                self.assertEqual(app.device.get(), '')
                self.assertIn('unavailable', app.message.get())
                app.device.set('Available')
                app.start_button.invoke()
                self.assertEqual([a for a, _ in app.worker.commands][-2:], ['preferences', 'start'])
                app.sharing.set(True)
                app.share_screen()
                self.assertEqual(app.worker.commands[-1], ('screen_sharing', dict(enabled=True)))
                for event in (dict(event='turn_started', turn_id='a'),
                              dict(event='partial', turn_id='a', text='Some answer'),
                              dict(event='turn_finished', turn_id='a', text='Some answer', complete=False)):
                    app.handle_event(event)
                app.render_conversation()
                self.assertIn('Incomplete reply', app.chat.get('1.0', 'end'))
                app.copy_buttons[0].invoke()
                self.assertEqual(root.clipboard_get(), 'Some answer')
                app.toggle_advanced()
                root.update()
                self.assertTrue(app.advanced.winfo_ismapped())
                app.toggle_advanced()
                app.close()
                self.assertEqual(app.worker.commands[-1][0], 'quit')
            finally:
                try:
                    root.destroy()
                except desktop.tk.TclError:
                    pass


if __name__ == '__main__':
    unittest.main()

