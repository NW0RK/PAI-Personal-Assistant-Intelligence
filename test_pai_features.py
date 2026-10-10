"""Feature checks use fake audio, inference, and desktop sessions."""
import itertools
import json
import os
from pathlib import Path
import queue
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

import app_settings as settings
import desktop_app as desktop
import desktop_helpers as helpers
import launch_gemma as launch
import live_gemma as live
import test_live_gemma as controller_checks
import test_desktop_visuals as visuals


class AppearanceAndExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'settings.json'

    def test_appearance_and_audio_saves_preserve_each_other(self):
        preferences = dict(settings.DEFAULTS, device='Applied microphone', threshold=.02)
        settings.save_preferences(preferences, self.path)
        appearance = dict(settings.UI_DEFAULTS, geometry=[-600, 20, 900, 800], maximized=True, caption_position='Bottom right')
        settings.save_appearance(appearance, self.path)
        self.assertEqual(settings.load_preferences(self.path)[0], preferences)
        settings.save_preferences(dict(preferences, voice=False), self.path)
        self.assertEqual(settings.load_appearance(self.path), appearance)
        self.assertEqual(json.loads(self.path.read_text())['version'], 1)

    def test_bad_appearance_does_not_reset_audio_and_legacy_files_load(self):
        preferences = dict(settings.DEFAULTS, threshold=.04)
        self.path.write_text(json.dumps(dict(version=1, preferences=preferences)))
        self.assertEqual(settings.load_appearance(self.path), settings.UI_DEFAULTS)
        self.path.write_text(json.dumps(dict(version=1, preferences=preferences,
            appearance=dict(geometry=[True, 0, 800, 600], maximized='yes', caption_position='invalid'))))
        self.assertEqual(settings.load_appearance(self.path), settings.UI_DEFAULTS)
        self.assertEqual(settings.load_preferences(self.path), (preferences, ''))

    def test_restore_clamps_removed_monitor_and_caption_corners(self):
        areas = [(-1280, 0, 0, 1024), (0, 0, 1920, 1040)]
        self.assertEqual(helpers.clamp_geometry([-1200, 30, 900, 700], areas), [-1200, 30, 900, 700])
        self.assertEqual(helpers.clamp_geometry([5000, 2000, 3000, 2000], [(0, 0, 1920, 1040)]), [0, 0, 1920, 1040])
        for position in settings.POSITIONS:
            x, y, w, h = helpers.caption_geometry(position, 640, 140, areas[1])
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x + w, 1920)
            self.assertEqual(y, 876 if position.startswith('Bottom') else 24)

    def test_export_marks_voice_and_incomplete_without_media(self):
        text = helpers.markdown_conversation(dict(a=dict(input_kind='text', input_text='Explain this', text='A reply', complete=True, screen=True),
                                                 b=dict(input_kind='voice', text='Partial', complete=False, screen=False)))
        self.assertIn('Explain this', text)
        self.assertIn('Screen shared', text)
        self.assertIn('no transcript', text)
        self.assertIn('Incomplete reply', text)
        self.assertNotIn('file://', text)


class CalibrationTests(unittest.TestCase):
    def test_sensitivity_rejects_noise_silence_clipping_and_bad_samples(self):
        threshold = helpers.suggested_threshold([.002] * 30, [.04] * 30)
        self.assertGreater(threshold, .002)
        self.assertLess(threshold, .04)
        for noise, speech in (([], []), ([0] * 30, [0] * 30), ([.03] * 30, [.04] * 30),
                              ([.001] * 30, [.95] * 30), ([float('nan')] * 30, [.1] * 30)):
            with self.assertRaises(ValueError):
                helpers.suggested_threshold(noise, speech)

    def make_calibration(self):
        output = queue.Queue()
        phase = ['noise']
        original = output.put
        def put(event):
            if event['stage'] in ('noise', 'speech'):
                phase[0] = event['stage']
            original(event)
        output.put = put
        capture = Mock()
        capture.read_pending.side_effect = lambda: struct.pack('<h', 60 if phase[0] == 'noise' else 1500) * live.FRAME_SAMPLES
        calibration = helpers.Calibration('fake', output, capture_factory=lambda _: capture,
                                          clock=itertools.count(0, .25).__next__)
        calibration.cancelled.wait = Mock(return_value=False)
        return calibration, capture, output

    def test_guided_sampling_stops_capture_and_returns_only_numeric_result(self):
        calibration, capture, output = self.make_calibration()
        calibration.run()
        capture.start.assert_called_once()
        capture.stop.assert_called_once()
        events = list(output.queue)
        self.assertEqual([e['stage'] for e in events if e['stage'] != 'level'], ['noise', 'speech', 'result'])
        self.assertGreater(events[-1]['threshold'], .001)
        self.assertTrue(calibration.done.is_set())

    def test_cancel_and_capture_failure_always_release_microphone(self):
        for cancel in (True, False):
            calibration, capture, output = self.make_calibration()
            if cancel:
                calibration.cancelled.wait.return_value = True
            else:
                capture.check_alive.side_effect = RuntimeError('Unplugged')
            calibration.run()
            capture.stop.assert_called_once()
            self.assertEqual(list(output.queue)[-1]['stage'], 'cancelled' if cancel else 'error')


class TypedControllerTests(unittest.TestCase):
    setUp = controller_checks.ControllerTests.setUp
    complete = controller_checks.ControllerTests.complete

    def test_text_only_completion_keeps_microphone_paused_and_has_no_audio(self):
        self.controller.control('pause')
        self.controller.control(dict(action='screen_sharing', enabled=False))
        self.controller.mic.start.reset_mock()
        self.controller.control(dict(action='send_text', text='Hello from text', request_id='one'))
        turn = self.controller.active
        self.assertEqual(turn.paths, ())
        self.assertEqual(turn.user['content'], [{'type': 'text', 'text': 'Hello from text'}])
        self.assertIsNone(self.job.call_args.args[2])
        self.complete()
        self.assertEqual(self.controller.state, 'Paused')
        self.controller.mic.start.assert_not_called()
        self.assertEqual(len(self.controller.messages), 3)

    def test_shared_typed_turn_cancellation_cleans_image_and_retains_pause(self):
        self.controller.control('pause')
        self.controller.start_text_turn('What is on screen?')
        turn = self.controller.active
        self.assertEqual(len(turn.paths), 1)
        self.assertTrue(turn.paths[0].endswith('.jpg'))
        self.assertEqual([c['type'] for c in turn.user['content']], ['text', 'image_url'])
        Path(turn.paths[0]).write_bytes(b'image')
        self.controller.control('cancel')
        self.assertFalse(Path(turn.paths[0]).exists())
        self.assertEqual(self.controller.state, 'Paused')
        self.assertEqual(len(self.controller.messages), 1)

    def test_text_busy_rejection_and_clear_preserves_pause(self):
        self.controller.control('pause')
        self.controller.start_text_turn('First')
        turn = self.controller.active
        self.controller.start_text_turn('Second')
        self.assertIs(self.controller.active, turn)
        self.job.assert_called_once()
        self.controller.control('clear')
        self.assertEqual(self.controller.state, 'Paused')
        self.assertIsNone(self.controller.active)

    def test_typed_worker_never_writes_audio_and_captures_only_requested_image(self):
        for paths in ((), ('image.jpg',)):
            with patch.object(live, 'write_wav') as write, patch.object(live, 'capture_screenshot') as capture, patch.object(live, 'chat_completion', return_value=('Answer', None)):
                connection = Mock()
                live.inference_worker(connection, 'typed', [], None, paths, 'local')
                write.assert_not_called()
                self.assertEqual(capture.call_count, len(paths))
                connection.send.assert_any_call(('typed', 'complete', 'Answer'))

    def test_text_only_listen_never_opens_capture_but_explicit_resume_can(self):
        self.args.text_only = True
        self.controller.status('Starting')
        self.controller.mic.start.reset_mock()
        self.controller.listen()
        self.assertEqual(self.controller.state, 'Paused')
        self.controller.mic.start.assert_not_called()
        self.controller.control('resume')
        self.controller.mic.start.assert_called_once()


class WorkerFeatureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'settings.json'
        self.preferences = dict(settings.DEFAULTS, device='Applied')
        settings.save_preferences(self.preferences, self.path)
        self.worker = desktop.SupervisorWorker(launch.parse_args(['--wait', '--settings-file', str(self.path)]), self.preferences)
        self.supervisor = self.worker.supervisor
        self.supervisor.send = Mock()
        self.supervisor.phase = 'Running'
        self.supervisor.state = 'Paused'
        self.supervisor.server_state = 'Ready'

    def test_acknowledged_settings_commit_and_save_failure_reports_session_changes(self):
        values = dict(self.preferences, threshold=.05)
        self.worker.handle(dict(action='preferences', preferences=values, request_id='apply'))
        self.assertEqual(settings.load_preferences(self.path)[0], self.preferences)
        self.supervisor.updates.put(dict(event='configured', request_id='apply', preferences=values))
        self.supervisor.tick = lambda: self.worker.send('quit')
        self.supervisor.close = Mock()
        self.worker.run()
        self.assertEqual(settings.load_preferences(self.path)[0], values)
        self.assertIsNone(self.worker.pending_preferences)
        with patch.object(desktop, 'save_preferences', side_effect=OSError('Read only')):
            self.worker.commit_preferences(self.preferences)
        self.assertEqual(self.worker.args.threshold, self.preferences['threshold'])
        self.assertIn('could not be saved', list(self.worker.output.queue)[-1]['warning'])

    def test_send_rejects_busy_and_text_start_does_not_need_device(self):
        self.worker.handle(dict(action='send_text', text='Ready', request_id='one'))
        self.supervisor.send.assert_called_with(dict(action='send_text', text='Ready', request_id='one'))
        self.supervisor.state = 'Thinking'
        self.worker.handle(dict(action='send_text', text='Busy', request_id='two'))
        self.assertEqual(list(self.worker.output.queue)[-1]['code'], 'input_unavailable')
        self.supervisor.start = Mock()
        self.worker.handle(dict(action='start', mode='text'))
        self.assertTrue(self.worker.args.text_only)

    def test_closed_control_pipe_rejects_messages_and_keeps_applied_settings(self):
        self.supervisor.send.return_value = False
        self.worker.handle(dict(action='send_text', text='Keep this draft', request_id='offline'))
        event = self.worker.output.get_nowait()
        self.assertEqual((event['kind'], event['request_id']), ('input', 'offline'))
        with self.assertRaisesRegex(ValueError, 'disconnected'):
            self.worker.handle(dict(action='preferences', preferences=dict(self.preferences, threshold=.04)))
        self.assertIsNone(self.worker.pending_preferences)
        self.assertEqual(settings.load_preferences(self.path)[0], self.preferences)

    def test_calibration_waits_for_pause_and_restores_only_listening(self):
        self.supervisor.state = 'Listening'
        self.worker.handle(dict(action='calibrate', device='Draft microphone'))
        self.supervisor.send.assert_called_with('pause')
        with patch.object(desktop, 'Calibration') as factory:
            self.worker.tick_calibration()
            factory.assert_not_called()
            self.supervisor.state = 'Paused'
            session = factory.return_value
            session.done.is_set.return_value = False
            self.worker.tick_calibration()
            factory.assert_called_once_with('Draft microphone', self.worker.calibration_events)
            session.start.assert_called_once()
            self.worker.handle(dict(action='cancel_calibration'))
            session.cancelled.set.assert_called_once()
            self.worker.calibration_events.put(dict(event='calibration', stage='cancelled'))
            session.done.is_set.return_value = True
            self.worker.tick_calibration()
            self.supervisor.send.assert_called_with('resume')
            self.assertIsNone(self.worker.calibration)


@unittest.skipUnless(os.environ.get('GEMMA_DESKTOP_TESTS') == '1', 'opens a simulated UI')
class FeatureWindowTests(unittest.TestCase):
    setUp = visuals.VisualWindowTests.setUp

    def test_drafts_discard_ack_failure_and_resume_without_implicit_apply(self):
        app = self.app
        visuals.snapshot(app, 'Paused')
        app.threshold.set('.04')
        self.assertEqual(app.settings_state.get(), 'Unsaved changes')
        app.pause_button.invoke()
        self.assertEqual(app.worker.commands[-1][0], 'resume')
        self.assertNotIn('preferences', [action for action, fields in app.worker.commands])
        app.discard_setup()
        self.assertEqual(float(app.threshold.get()), app.preferences['threshold'])
        app.threshold.set('.05')
        app.apply_setup()
        self.assertTrue(app.applying)
        app.handle_event(dict(event='failure', kind='settings', code='settings_invalid', detail='Try again'))
        self.assertFalse(app.applying)
        self.assertEqual(float(app.threshold.get()), .05)
        app.apply_setup()
        acknowledged = dict(app.pending_preferences)
        app.threshold.set('.06')
        app.handle_event(dict(event='preferences_applied', preferences=acknowledged, warning='Session only'))
        self.assertEqual(app.preferences['threshold'], .05)
        self.assertEqual(float(app.threshold.get()), .06)
        self.assertEqual(app.settings_state.get(), 'Unsaved changes')
        self.assertEqual(app.banner.get(), 'Session only')

    def test_text_start_send_ack_disconnection_keeps_draft(self):
        app = self.app
        app.preferences['device'] = ''
        app.start_services('text')
        self.assertEqual(app.worker.commands[-1], ('start', {'mode': 'text'}))
        visuals.snapshot(app, 'Paused')
        app.composer.insert('1.0', 'Typed draft')
        app.send_text()
        request = dict(app.pending_text)
        self.assertEqual(app.worker.commands[-1][0], 'send_text')
        app.handle_event(dict(event='turn_started', turn_id='typed', input_kind='text', input_text=request['text'], request_id=request['request_id']))
        self.assertEqual(app.composer.get('1.0', 'end-1c'), '')
        app.handle_event(dict(event='turn_finished', turn_id='typed', text='Reply', complete=True))
        app.render_conversation()
        self.assertIn('Typed draft', app.chat.get('1.0', 'end'))
        visuals.snapshot(app, 'Paused', server='Unavailable')
        app.composer.insert('1.0', 'Keep me')
        app.send_text()
        self.assertIsNone(app.pending_text)
        self.assertEqual(app.composer.get('1.0', 'end-1c'), 'Keep me')

    def test_completed_widgets_selection_and_reading_position_survive_streaming(self):
        app = self.app
        app.handle_event(dict(event='turn_started', turn_id='old'))
        app.handle_event(dict(event='turn_finished', turn_id='old', text='Completed answer. ' * 80, complete=True))
        app.render_conversation()
        button = app.copy_buttons[0]
        record = app.rendered_turns['old']
        app.chat.tag_add('sel', record['reply_start'], record['reply_start'] + '+9c')
        selected = app.chat.tag_ranges('sel')
        app.handle_event(dict(event='turn_started', turn_id='new'))
        for index in range(12):
            app.handle_event(dict(event='partial', turn_id='new', text='Streaming ' * (index + 1)))
            app.render_conversation()
            self.assertIs(app.copy_buttons[0], button)
            self.assertEqual(app.chat.tag_ranges('sel'), selected)
        self.root.update()
        app.chat.yview_moveto(0)
        self.root.update()
        self.assertTrue(app.jump_button.winfo_ismapped())
        app.jump_latest()
        self.root.update()
        self.assertGreater(app.chat.yview()[1], .98)

    def test_export_recovery_geometry_and_calibration_draft(self):
        app = self.app
        app.handle_event(dict(event='turn_started', turn_id='voice'))
        app.handle_event(dict(event='turn_finished', turn_id='voice', text='Partial', complete=False))
        target = Path(self.directory.name) / 'conversation.md'
        with patch.object(desktop.filedialog, 'asksaveasfilename', return_value=str(target)):
            app.export_conversation()
        self.assertIn('Incomplete reply', target.read_text())
        app.show_recovery('microphone_unavailable')
        app.recover()
        self.assertTrue(app.settings_open)
        self.assertNotIn('preferences', [a for a, _ in app.worker.commands])
        app.handle_event(dict(event='calibration', stage='result', threshold=.025))
        self.assertEqual(float(app.threshold.get()), .025)
        self.assertEqual(app.preferences['threshold'], settings.DEFAULTS['threshold'])
        app.caption_position.set('Bottom right')
        app.change_caption_position()
        self.assertEqual(app.worker.commands[-1][1]['appearance']['caption_position'], 'Bottom right')
        app.remember_window()
        self.assertEqual(len(app.appearance['geometry']), 4)

    def test_history_eviction_removes_only_oldest_display_and_reuses_completed_buttons(self):
        app = self.app
        for index in range(100):
            app.handle_event(dict(event='turn_started', turn_id=str(index)))
            app.handle_event(dict(event='turn_finished', turn_id=str(index), text=f'Answer {index:03d}.', complete=True))
        app.render_conversation()
        retained_button = app.rendered_turns['1']['button']
        old_button = app.rendered_turns['0']['button']
        app.handle_event(dict(event='turn_started', turn_id='100'))
        app.render_conversation()
        self.assertEqual(len(app.rendered_turns), 100)
        self.assertNotIn('0', app.rendered_turns)
        self.assertFalse(old_button.winfo_exists())
        self.assertIs(app.rendered_turns['1']['button'], retained_button)
        self.assertNotIn('Answer 000.', app.chat.get('1.0', 'end'))
        self.assertIn('Answer 001.', app.chat.get('1.0', 'end'))


if __name__ == '__main__':
    unittest.main()
