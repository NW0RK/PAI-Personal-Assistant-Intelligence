import struct
import io
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import live_gemma as live


VOICE = struct.pack("<h", 2000) * live.FRAME_SAMPLES
QUIET = bytes(live.FRAME_BYTES)


class SpeechDetectorTests(unittest.TestCase):
    def test_speech_survives_small_unaligned_reads(self):
        detector = live.SpeechDetector(0.3, 0.01)
        recording = VOICE * 25 + QUIET * 10
        results = []
        for offset in range(0, len(recording), 137):
            results.extend(detector.feed(recording[offset:offset + 137]))
        self.assertEqual(results, [VOICE * 25])
        self.assertEqual(detector.pending, b"")

    def test_multiple_utterances_and_pre_roll(self):
        detector = live.SpeechDetector(0.3, 0.01)
        results = detector.feed(QUIET * 10 + VOICE * 20 + QUIET * 10 + VOICE * 22 + QUIET * 10)
        self.assertEqual(results, [QUIET * 6 + VOICE * 20, VOICE * 22])

    def test_brief_noise_does_not_trigger(self):
        detector = live.SpeechDetector(0.3, 0.01)
        self.assertEqual(detector.feed(VOICE * 3 + QUIET * 10), [])

    def test_long_speech_is_split_at_limit(self):
        detector = live.SpeechDetector(0.3, 0.01)
        results = []
        for _ in range(1500):
            results.extend(detector.feed(VOICE))
        self.assertEqual(results, [VOICE * 1500])
        self.assertFalse(detector.frames)

    def test_keeps_short_pause_inside_speech(self):
        detector = live.SpeechDetector(0.3, 0.01)
        self.assertEqual(detector.feed(VOICE * 20 + QUIET * 4 + VOICE * 20 + QUIET * 10),
                         [VOICE * 20 + QUIET * 4 + VOICE * 20])


class CaptureTests(unittest.TestCase):
    def test_read_pending_drains_audio(self):
        mic = live.MicCapture(live.DEFAULT_DEVICE)
        mic.buffer.extend(VOICE)
        self.assertEqual(mic.read_pending(), VOICE)
        self.assertEqual(mic.read_pending(), b"")


class StreamingTests(unittest.TestCase):
    def test_empty_choices_and_content(self):
        response = [b'data: {"choices": []}\n',
                    b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
                    b'data: [DONE]\n']
        with patch.object(live.urllib.request, "urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = response
            self.assertEqual(live.chat_completion([]), ("Hello", ""))
            request = json.loads(urlopen.call_args.args[0].data)
            self.assertEqual(request["reasoning_effort"], "none")
            self.assertFalse(request["chat_template_kwargs"]["enable_thinking"])

    def test_reasoning_only_reply_retries_with_larger_budget(self):
        empty = [b'data: {"choices":[{"delta":{"reasoning_content":"thinking"},"finish_reason":"length"}]}\n',
                 b'data: {"choices":[],"usage":{"completion_tokens":512}}\n',
                 b'data: [DONE]\n']
        answered = [b'data: {"choices":[{"delta":{"content":"Answer"},"finish_reason":"stop"}]}\n',
                    b'data: [DONE]\n']
        with patch.object(live.urllib.request, "urlopen") as urlopen, patch("builtins.print") as output:
            urlopen.return_value.__enter__.side_effect = [empty, answered]
            self.assertEqual(live.chat_completion([]), ("Answer", ""))
            requests = [json.loads(call.args[0].data) for call in urlopen.call_args_list]
            self.assertEqual([request["max_tokens"] for request in requests], [512, 1024])
            self.assertIn("finish_reason=length, completion_tokens=512, reasoning_chars=8",
                          output.call_args.args[0])

    def test_persistent_empty_reply_reports_details(self):
        response = [b'data: {"choices":[{"delta":{"content":"  "},"finish_reason":"stop"}]}\n',
                    b'data: [DONE]\n']
        with patch.object(live.urllib.request, "urlopen") as urlopen, patch("builtins.print"):
            urlopen.return_value.__enter__.return_value = response
            with self.assertRaisesRegex(RuntimeError, "no answer after retry: finish_reason=stop"):
                live.chat_completion([])
            self.assertEqual(urlopen.call_count, 2)

    def test_incomplete_stream_is_not_accepted_as_a_reply(self):
        with patch.object(live.urllib.request, "urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = [b'data: {"choices":[{"delta":{"content":"partial"}}]}\n']
            with self.assertRaisesRegex(RuntimeError, r"ended before \[DONE\]"):
                live.chat_completion([])

    def test_stream_error_is_reported(self):
        with patch.object(live.urllib.request, "urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = [b'data: {"error": "failed"}\n']
            with self.assertRaisesRegex(RuntimeError, "failed"):
                live.chat_completion([])


class ControllerTests(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        self.directory = tempfile.TemporaryDirectory(dir=os.path.dirname(__file__))
        self.addCleanup(self.directory.cleanup)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(live, "MEDIA_DIR", self.directory.name))
        self.stack.enter_context(patch("builtins.print"))
        self.job = self.stack.enter_context(patch.object(live, "InferenceJob"))
        self.audio = self.stack.enter_context(patch.object(live, "AudioJob"))
        self.args = SimpleNamespace(device="test", silence=0.3, threshold=0.01,
                                    no_tts=True, test=False, server_url="http://localhost:9999", piper_path="piper")
        self.overlay = Mock()
        self.overlay.poll_controls.return_value = []
        self.controller = live.LiveController(self.args, self.overlay)
        self.controller.mic = Mock()
        self.controller.mic.read_pending.return_value = b""
        self.addCleanup(self.controller.close)
        self.controller.listen()

    def begin(self):
        self.controller.start_turn(VOICE * 20)
        turn = self.controller.active
        for path in turn.paths:
            with open(path, "wb") as output:
                output.write(b"media")
        return turn.id

    def complete(self):
        self.controller.handle_event((self.controller.active.id, "complete", "Hello"))

    def test_history_has_distinct_media_and_cleans_up(self):
        counts = []
        for _ in range(12):
            self.begin()
            request = self.job.call_args.args[1]
            users = [m for m in request if m["role"] == "user"]
            urls = [m["content"][1]["image_url"]["url"] for m in users]
            self.assertEqual(len(set(urls)), len(urls))
            for url in urls:
                self.assertTrue(os.path.exists(os.path.join(self.directory.name, url[7:])))
            counts.append(len(users))
            self.complete()
            self.assertLessEqual(len(os.listdir(self.directory.name)), 20)
        self.assertEqual(counts, list(range(1, 12)) + [11])
        self.assertEqual(len(self.controller.messages), 21)
        self.controller.close()
        self.assertEqual(os.listdir(self.directory.name), [])

    def test_voice_only_turn_omits_image_and_screen_toggle_affects_next_turn(self):
        self.controller.control(dict(action='screen_sharing', enabled=False))
        self.begin()
        turn = self.controller.active
        self.assertEqual(len(turn.paths), 1)
        self.assertEqual([c['type'] for c in turn.user['content']], ['text', 'input_audio'])
        self.controller.control(dict(action='screen_sharing', enabled=True))
        self.assertEqual(len(turn.paths), 1)
        self.complete()
        self.begin()
        self.assertEqual(len(self.controller.active.paths), 2)

    def test_clear_cancels_workers_removes_media_and_rejects_late_events(self):
        self.begin()
        self.complete()
        turn_id = self.begin()
        self.controller.control('clear')
        self.assertEqual(len(self.controller.messages), 1)
        self.assertFalse(self.controller.turn_media)
        self.assertFalse(self.controller.owned)
        self.assertEqual(self.controller.state, 'Listening')
        self.controller.handle_event((turn_id, 'complete', 'late reply'))
        self.assertEqual(len(self.controller.messages), 1)
        self.assertEqual(os.listdir(self.directory.name), [])

    def test_clear_preserves_pause(self):
        self.begin()
        self.complete()
        self.controller.control('pause')
        self.controller.control('clear')
        self.assertEqual(self.controller.state, 'Paused')
        self.assertEqual(len(self.controller.messages), 1)

    def test_reconfigure_microphone_preserves_history_and_pause(self):
        self.begin()
        self.complete()
        self.controller.control('pause')
        self.controller.control(dict(action='configure', preferences=dict(
            device='New device', silence=.5, threshold=.02, voice=False, captions=True)))
        self.assertEqual(self.controller.state, 'Paused')
        self.assertEqual(self.controller.mic.device, 'New device')
        self.assertEqual(self.controller.args.silence, .5)
        self.assertEqual(len(self.controller.messages), 3)

    def test_cancel_generation_discards_history_and_late_events(self):
        turn = self.begin()
        self.controller.handle_event((turn, "partial", "Some answer"))
        self.overlay.show.assert_called_with("Some answer", duration=None)
        self.controller.control("cancel")
        self.assertEqual(self.controller.state, "Listening")
        self.assertEqual(len(self.controller.messages), 1)
        self.overlay.show.assert_called_with("Some answer (incomplete)", duration=8)
        self.assertEqual(os.listdir(self.directory.name), [])
        self.begin()
        self.controller.handle_event((turn, "complete", "late"))
        self.assertEqual(len(self.controller.messages), 1)
        self.assertFalse(self.controller.active.committed)

    def test_pause_resume_and_repeated_controls(self):
        self.begin()
        self.controller.control("pause")
        self.assertEqual(self.controller.state, "Paused")
        self.assertIsNone(self.controller.active)
        self.controller.mic.start.reset_mock()
        self.controller.control("pause")
        self.controller.control("cancel")
        self.controller.mic.start.assert_not_called()
        old_detector = self.controller.detector
        self.controller.control("resume")
        self.controller.control("resume")
        self.controller.mic.start.assert_called_once()
        self.assertIsNot(self.controller.detector, old_detector)

    def test_cancel_synthesis_and_playback_preserves_complete_turn(self):
        self.args.no_tts = False
        for stage in ("Preparing voice", "Speaking"):
            self.begin()
            self.complete()
            if stage == "Speaking":
                self.audio.return_value.poll.return_value = True
                self.controller.tick()
            self.assertEqual(self.controller.state, stage)
            length = len(self.controller.messages)
            self.controller.control("cancel")
            self.assertEqual(len(self.controller.messages), length)
            self.assertEqual(self.controller.state, "Listening")
            self.audio.return_value.close.assert_called()
            self.overlay.show.assert_called_with("Hello", duration=8)

    def test_captions_visible_before_synthesis_and_survive_failure(self):
        self.args.no_tts = False
        self.begin()
        def fail(*args):
            self.overlay.show.assert_called_with("Hello", duration=None)
            raise RuntimeError("synthesis failed")
        self.audio.side_effect = fail
        self.complete()
        self.overlay.show.assert_called_with("Hello", duration=8)
        self.assertIn("Voice output failed", self.controller.notice)
        self.assertEqual(self.controller.state, "Listening")

    def test_playback_error_preserves_captions_and_cleans_reply(self):
        self.args.no_tts = False
        self.begin()
        self.complete()
        reply = self.controller.speech.synthesis_path
        with open(reply, "wb") as output:
            output.write(b"audio")
        self.audio.return_value.poll.side_effect = RuntimeError("playback failed")
        self.controller.tick()
        self.overlay.show.assert_called_with("Hello", duration=8)
        self.assertFalse(os.path.exists(reply))
        self.assertEqual(len(self.controller.messages), 3)

    def test_microphone_error_retries_only_on_resume(self):
        self.controller.mic.check_alive.side_effect = RuntimeError("unplugged")
        self.controller.tick()
        self.assertEqual(self.controller.state, "Error")
        self.controller.mic.start.reset_mock()
        self.controller.tick()
        self.controller.mic.start.assert_not_called()
        self.controller.control("resume")
        self.controller.mic.start.assert_called_once()

    def test_capture_stopped_through_entire_reply(self):
        self.args.no_tts = False
        self.begin()
        self.controller.mic.stop.assert_called()
        self.controller.mic.start.reset_mock()
        self.complete()
        self.controller.mic.start.assert_not_called()
        self.audio.return_value.poll.return_value = True
        self.controller.tick()
        self.assertEqual(self.controller.state, "Speaking")
        self.controller.mic.start.assert_not_called()
        self.controller.tick()
        self.assertEqual(self.controller.state, "Listening")
        self.controller.mic.start.assert_called_once()

    def test_failure_does_not_retry_or_commit_history(self):
        turn = self.begin()
        self.controller.handle_event((turn, "failure", "network error"))
        self.job.assert_called_once()
        self.assertEqual(len(self.controller.messages), 1)
        self.assertEqual(os.listdir(self.directory.name), [])
        self.assertEqual(self.controller.state, "Listening")

    def test_quit_closes_workers_and_owned_media(self):
        self.begin()
        self.controller.control("quit")
        self.controller.control("quit")
        self.job.return_value.close.assert_called_once()
        self.assertTrue(self.controller.quitting)
        self.assertEqual(os.listdir(self.directory.name), [])

    def test_test_mode_never_opens_microphone(self):
        self.args.test = True
        self.controller.mic.start.reset_mock()
        self.begin()
        self.complete()
        self.controller.mic.start.assert_not_called()


class PartialStreamingTests(unittest.TestCase):
    def test_partial_updates_throttled_and_reasoning_hidden(self):
        from unittest.mock import Mock
        response = [b'data: {"choices":[{"delta":{"content":"A","reasoning_content":"secret"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"B"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"C"}}]}\n', b'data: [DONE]\n']
        callback = Mock()
        with patch.object(live.urllib.request, "urlopen") as urlopen, patch.object(live.time, "monotonic", side_effect=[0, .05, .11]):
            urlopen.return_value.__enter__.return_value = response
            result = live.chat_completion([], on_partial=callback, server_url="http://localhost:9999")
            self.assertEqual(result, ("ABC", "secret"))
            self.assertEqual([call.args[0] for call in callback.call_args_list], ["A", "ABC"])
            self.assertEqual(urlopen.call_args.args[0].full_url, "http://localhost:9999/v1/chat/completions")


def stalled_worker(connection, turn_id, *args):
    import time
    connection.send((turn_id, "partial", "ready"))
    time.sleep(30)


class ProcessTests(unittest.TestCase):
    def test_windows_pipe_close_preserves_completed_event(self):
        from unittest.mock import Mock
        job = live.InferenceJob.__new__(live.InferenceJob)
        job.connection = Mock()
        job.connection.poll.side_effect = [True, BrokenPipeError()]
        job.connection.recv.return_value = ("test", "complete", "Hello")
        job.turn_id, job.ended = "test", False
        self.assertEqual(job.poll(), [("test", "complete", "Hello")])

    def test_unexpected_worker_exit_becomes_failure(self):
        from unittest.mock import Mock
        job = live.InferenceJob.__new__(live.InferenceJob)
        job.connection = Mock()
        job.connection.poll.side_effect = BrokenPipeError()
        job.turn_id, job.ended = "test", False
        self.assertEqual(job.poll()[0][:2], ("test", "failure"))
        self.assertEqual(job.poll(), [])

    def test_screenshot_failure_reported_by_worker(self):
        from unittest.mock import Mock
        connection = Mock()
        with patch.object(live, "write_wav"), patch.object(live, "capture_screenshot", side_effect=RuntimeError("screen unavailable")):
            live.inference_worker(connection, "test", [], b"", ("a.wav", "b.jpg"), "unused")
        connection.send.assert_called_with(("test", "failure", "screen unavailable"))
        connection.close.assert_called_once()

    def test_voice_only_worker_never_captures_screen(self):
        from unittest.mock import Mock
        connection = Mock()
        with patch.object(live, 'write_wav') as write, patch.object(live, 'capture_screenshot') as capture, \
                patch.object(live, 'chat_completion', return_value=('Hello', '')):
            live.inference_worker(connection, 'test', [], b'audio', ('a.wav',), 'unused')
        write.assert_called_once_with('a.wav', b'audio')
        capture.assert_not_called()
        self.assertIn((('test', 'complete', 'Hello'),), [call.args for call in connection.send.call_args_list])

    def test_stalled_worker_emits_failure_after_deadline(self):
        from unittest.mock import Mock
        job = live.InferenceJob.__new__(live.InferenceJob)
        job.connection = Mock()
        job.connection.poll.return_value = False
        job.turn_id, job.ended, job.last_progress = "test", False, 0
        with patch.object(live.time, "monotonic", return_value=46):
            self.assertEqual(job.poll()[0][:2], ("test", "failure"))
            self.assertEqual(job.poll(), [])

    def test_terminal_input_is_nonblocking_and_handles_split_commands(self):
        import os
        reader, writer = os.pipe()
        with os.fdopen(reader, "rb", buffering=0) as source:
            commands = live.TerminalCommands(source)
            try:
                self.assertEqual(commands.poll(), [])
                os.write(writer, b"pau")
                self.assertEqual(commands.poll(), [])
                os.write(writer, b"se\nresume\ncancel\n")
                self.assertEqual(commands.poll(), ["pause", "resume", "cancel"])
            finally:
                os.close(writer)
            self.assertEqual(commands.poll(), ["quit"])
            self.assertEqual(commands.poll(), [])

    def test_real_spawned_worker_cancellation_under_one_second(self):
        import time
        with patch.object(live, "inference_worker", stalled_worker):
            job = live.InferenceJob("test", [], b"", (), "unused")
        try:
            deadline = time.monotonic() + 5
            events = []
            while not events and time.monotonic() < deadline:
                events = job.poll()
                time.sleep(.02)
            self.assertEqual(events, [("test", "partial", "ready")])
            started = time.monotonic()
            job.close()
            self.assertLess(time.monotonic() - started, 1)
        except BaseException:
            if not job.process._closed:
                job.close()
            raise

    def test_real_audio_subprocess_cancellation_under_one_second(self):
        import time
        job = live.AudioJob([live.sys.executable, "-c", "import time; time.sleep(30)"], 60)
        started = time.monotonic()
        job.close()
        self.assertLess(time.monotonic() - started, 1)
        self.assertIsNotNone(job.process.poll())

    def test_audio_failure_and_timeout_reported(self):
        import time
        job = live.AudioJob([live.sys.executable, "-c", "import sys; sys.stderr.write('bad audio'); sys.exit(2)"], 5)
        try:
            job.process.wait(timeout=5)
            with self.assertRaisesRegex(RuntimeError, "bad audio"):
                job.poll()
        finally:
            job.close()
        job = live.AudioJob([live.sys.executable, "-c", "import time; time.sleep(30)"], -1)
        try:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                job.poll()
        finally:
            job.close()

    def test_piper_resolution_order(self):
        with patch.object(live.shutil, "which", side_effect=lambda path: {"chosen": "chosen", "piper": "found", "found": "found"}.get(path)):
            self.assertEqual(live.resolve_piper("chosen"), "chosen")
            self.assertEqual(live.resolve_piper("missing"), "found")


if __name__ == "__main__":
    unittest.main()

