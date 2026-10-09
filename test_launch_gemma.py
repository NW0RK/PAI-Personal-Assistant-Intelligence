import io
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch
import launch_gemma as launch


class LauncherTests(unittest.TestCase):
    def test_mouse_buttons_match_keyboard_controls(self):
        for text, key in (('Start all', '\r'), ('Pause/Resume', 'p'), ('Cancel reply', 'c'), ('Quit', 'q')):
            column = launch.CONTROL_BAR.index(text)
            self.assertEqual(launch.clicked_control(column, 2), key)
            self.assertIsNone(launch.clicked_control(column, 3))
        self.assertIsNone(launch.clicked_control(1000, 2))

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        with patch('sys.argv', ['launch_gemma.py']):
            self.args = launch.parse_args()
        self.dashboard = launch.Dashboard(self.args)
        self.stack.enter_context(patch.object(self.dashboard, 'preflight', return_value=True))
        self.stack.enter_context(patch.object(self.dashboard, 'draw'))
        self.health = self.stack.enter_context(patch.object(launch, 'health', return_value='Unavailable'))
        self.stack.enter_context(patch.object(launch, 'port_in_use', return_value=False))
        self.stack.enter_context(patch.object(launch.Path, 'is_file', return_value=True))
        self.stack.enter_context(patch.object(launch.Path, 'mkdir'))
        self.stack.enter_context(patch('builtins.open', return_value=Mock()))
        self.process = self.stack.enter_context(patch.object(launch.subprocess, 'Popen'))
        self.process.return_value.poll.return_value = None
        self.start_assistant = self.stack.enter_context(patch.object(self.dashboard, 'launch_assistant'))

    def test_ready_external_server_is_reused_and_never_stopped(self):
        self.health.return_value = 'Ready'
        self.dashboard.start()
        self.start_assistant.assert_called_once()
        self.process.assert_not_called()
        self.dashboard.close()
        self.process.return_value.terminate.assert_not_called()

    def test_start_is_idempotent_and_waits_for_server_readiness(self):
        self.dashboard.start()
        self.dashboard.start()
        self.process.assert_called_once()
        self.start_assistant.assert_not_called()
        self.assertEqual(self.dashboard.phase, 'Loading')
        self.health.return_value = 'Ready'
        self.dashboard.tick()
        self.start_assistant.assert_called_once()
        self.dashboard.close()
        self.process.return_value.terminate.assert_called_once()

    def test_external_loading_server_is_not_duplicated(self):
        self.health.return_value = 'Loading'
        self.dashboard.start()
        self.assertEqual(self.dashboard.phase, 'Loading')
        self.process.assert_not_called()

    def test_retry_after_owned_server_crash_releases_assistant_and_restarts(self):
        assistant = Mock()
        assistant.stdin.closed = False
        self.dashboard.assistant = assistant
        self.dashboard.phase = 'Running'
        old_server = Mock()
        old_server.poll.return_value = 1
        self.dashboard.server = old_server
        self.dashboard.restart()
        assistant.wait.assert_called_once_with(timeout=5)
        assistant.stdin.close.assert_called_once()
        self.assertIsNone(self.dashboard.assistant)
        self.assertEqual(self.dashboard.phase, 'Loading')
        self.process.assert_called_once()
        old_server.terminate.assert_not_called()

    def test_occupied_port_shows_actionable_error(self):
        with patch.object(launch, 'port_in_use', return_value=True):
            self.dashboard.start()
        self.assertEqual(self.dashboard.phase, 'Error')
        self.assertIn('--port', self.dashboard.notice)
        self.process.assert_not_called()

    def test_server_crash_reports_log_path(self):
        self.dashboard.start()
        self.process.return_value.poll.return_value = 1
        self.dashboard.tick()
        self.assertEqual(self.dashboard.phase, 'Error')
        self.assertIn('launcher-server.log', self.dashboard.notice)

    def test_startup_timeout_remains_retryable(self):
        self.dashboard.start()
        self.dashboard.deadline = 0
        self.dashboard.tick()
        self.assertEqual(self.dashboard.phase, 'Error')
        self.dashboard.start()
        self.process.assert_called_once()
        self.assertEqual(self.dashboard.phase, 'Loading')

    def test_custom_port_propagates_to_both_services(self):
        self.args.port = 9090
        command = launch.server_command(self.args)
        self.assertEqual(command[command.index('--port') + 1], '9090')
        dashboard = launch.Dashboard(self.args)
        self.args.speech_mode = 'full'
        self.process.return_value.stdout = io.StringIO('{"event":"status","state":"Listening","detail":""}\n')
        dashboard.launch_assistant()
        command = self.process.call_args.args[0]
        self.assertEqual(command[command.index('--server-url') + 1], 'http://127.0.0.1:9090')
        self.assertEqual(command[command.index('--speech-mode') + 1], 'full')
        dashboard.reader.join(timeout=1)
        dashboard.tick()
        self.assertEqual(dashboard.state, 'Listening')

    def test_status_ignores_logs_stale_sessions_and_malformed_payloads(self):
        self.dashboard.notice = 'old notice'
        self.dashboard.events.put((-1, '{"event":"status","state":"Error","detail":"stale"}'))
        for line in ('[Speaking] plain log', '[]', '{bad json',
                     '{"event":"status","state":"Speaking","detail":17}'):
            self.dashboard.events.put((0, line))
        self.dashboard.tick()
        self.assertEqual(self.dashboard.state, 'Stopped')
        self.dashboard.events.put((0, '{"event":"status","state":"Speaking","detail":""}'))
        self.dashboard.tick()
        self.assertEqual(self.dashboard.state, 'Speaking')
        self.assertEqual(self.dashboard.notice, '')


if __name__ == '__main__':
    unittest.main()
