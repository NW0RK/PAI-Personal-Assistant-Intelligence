import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import live_gemma as live
from speech_pipeline import PhraseBuffer, SpeechConfig, SpeechPipeline


class Job:
    def __init__(self):
        self.done = False
        self.closed = False

    def poll(self):
        return self.done

    def close(self):
        self.closed = True


class SegmentationTests(unittest.TestCase):
    def test_fragmented_punctuation_and_final_tail(self):
        buffer = PhraseBuffer(SpeechConfig())
        for fragment in ('Hello', '.', '"'):
            buffer.pending += fragment
            self.assertIsNone(buffer.pop())
        buffer.pending += ' Next line\nFinal fragment'
        self.assertEqual(buffer.pop(), 'Hello."')
        self.assertEqual(buffer.pop(), 'Next line')
        self.assertIsNone(buffer.pop())
        buffer.finished = True
        self.assertEqual(buffer.pop(), 'Final fragment')
        self.assertIsNone(buffer.pop())

    def test_limit_and_unbroken_tokens(self):
        buffer = PhraseBuffer(SpeechConfig(phrase_limit=10))
        buffer.pending = 'one two three four'
        self.assertEqual(buffer.pop(), 'one two')
        buffer.pending = 'abcdefghijklmnop'
        self.assertIsNone(buffer.pop())
        buffer.pending += ' next'
        self.assertEqual(buffer.pop(), 'abcdefghijklmnop')

    def test_full_mode_waits_and_keeps_answer_together(self):
        buffer = PhraseBuffer(SpeechConfig(mode='full'))
        buffer.pending = 'First. Second.\nThird.'
        self.assertIsNone(buffer.pop())
        buffer.finished = True
        self.assertEqual(buffer.pop(), 'First. Second.\nThird.')


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.syntheses, self.plays, self.removed, self.marks = [], [], [], []
        def synthesize(text, path):
            job = Job()
            self.syntheses.append((text, path, job))
            return job
        def play(path):
            job = Job()
            self.plays.append((path, job))
            return job
        self.pipeline = SpeechPipeline(SpeechConfig(), synthesize, play,
            lambda n: f'chunk-{n}.wav', self.removed.extend,
            lambda event, **fields: self.marks.append((event, fields)))
        self.addCleanup(self.pipeline.close)

    def test_overlap_backpressure_and_order_without_text_loss(self):
        self.pipeline.feed('One. Two. Three. Four. Five. ')
        self.pipeline.schedule()
        self.syntheses[-1][2].done = True
        self.pipeline.tick()
        self.assertEqual(len(self.plays), 1)
        self.assertFalse(self.pipeline.buffer.finished)
        self.assertIsNotNone(self.pipeline.synthesis)
        for _ in range(2):
            self.syntheses[-1][2].done = True
            self.pipeline.tick()
        self.assertEqual(len(self.pipeline.ready), 2)
        self.assertIsNone(self.pipeline.synthesis)
        self.assertIn('Four.', self.pipeline.buffer.pending)
        self.pipeline.finish()
        for _ in range(10):
            for _, _, job in self.syntheses:
                job.done = True
            for _, job in self.plays:
                job.done = True
            self.pipeline.tick()
            self.assertLessEqual(len(self.pipeline.ready), 2)
        self.assertTrue(self.pipeline.drained)
        self.assertEqual([t for t, _, _ in self.syntheses], ['One.', 'Two.', 'Three.', 'Four.', 'Five.'])
        self.assertEqual([p for p, _ in self.plays], [f'chunk-{n}.wav' for n in range(1, 6)])
        self.assertEqual(set(self.removed), {f'chunk-{n}.wav' for n in range(1, 6)})

    def test_close_stops_both_jobs_and_removes_unfinished_files(self):
        self.pipeline.feed('One. Two. ')
        self.pipeline.schedule()
        self.syntheses[0][2].done = True
        self.pipeline.tick()
        self.pipeline.close()
        self.pipeline.close()
        self.assertTrue(all(j.closed for _, _, j in self.syntheses))
        self.assertTrue(all(j.closed for _, j in self.plays))
        self.assertEqual(set(self.removed), {'chunk-1.wav', 'chunk-2.wav'})
        self.pipeline.feed('Late. ')
        self.pipeline.tick()
        self.assertEqual(len(self.syntheses), 2)


class StreamingControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.path.dirname(__file__))
        self.addCleanup(self.temp.cleanup)
        self.patch = patch.object(live, 'MEDIA_DIR', self.temp.name)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.output = patch('builtins.print').start()
        self.addCleanup(patch.stopall)
        self.inference = Mock()
        self.inference.poll.return_value = []
        self.jobs = []
        def audio(command, timeout, text=None):
            job = Job()
            self.jobs.append(job)
            if text is not None:
                with open(command[-1], 'wb') as stream:
                    stream.write(b'fixture')
            return job
        args = SimpleNamespace(device='test', silence=1, threshold=.01, test=True,
            no_tts=False, server_url='unused', piper_path='unused', speech_mode='streamed')
        self.controller = live.LiveController(args, Mock(), inference_factory=lambda *a: self.inference,
                                              audio_factory=audio, clock=lambda: 123.0)
        self.controller.overlay.poll_controls.return_value = []
        self.addCleanup(self.controller.close)
        self.controller.start_turn(b'')
        self.turn = self.controller.active

    def emit(self, kind, text):
        self.controller.handle_event((self.turn.id, kind, text))

    def start_playback(self):
        self.emit('delta', 'First sentence. ')
        self.jobs[0].done = True
        self.controller.tick()
        self.assertEqual(self.controller.state, 'Speaking')
        self.assertEqual(self.turn.generation, 'running')

    def test_speech_starts_before_completion_and_waits_for_final_audio(self):
        self.start_playback()
        self.emit('delta', 'Final fragment')
        self.emit('complete', 'First sentence. Final fragment')
        self.assertTrue(self.turn.committed)
        self.assertEqual(self.controller.state, 'Speaking')
        for _ in range(5):
            for job in self.jobs:
                job.done = True
            self.controller.tick()
        self.assertIsNone(self.controller.active)
        self.assertEqual(self.controller.state, 'Listening')
        self.assertEqual(os.listdir(self.temp.name), [])
        stages = [e['stage'] for e in self.turn.timings]
        self.assertLess(stages.index('playback_start'), stages.index('generation_complete'))
        self.assertTrue(all('text' not in e for e in self.turn.timings))

    def test_failure_after_playback_stops_audio_and_discards_history(self):
        self.start_playback()
        self.emit('failure', 'network failed')
        self.assertTrue(all(j.closed for j in self.jobs))
        self.assertEqual(len(self.controller.messages), 1)
        self.assertEqual(self.turn.generation, 'failed')
        self.controller.overlay.show.assert_called_with('First sentence.  (incomplete)', duration=8)
        self.assertEqual(os.listdir(self.temp.name), [])

    def test_audio_failure_keeps_generation_and_captions_alive(self):
        self.start_playback()
        self.jobs[-1].poll = Mock(side_effect=RuntimeError('device lost'))
        self.controller.tick()
        self.assertIsNotNone(self.controller.inference)
        self.assertIsNone(self.controller.speech)
        self.emit('delta', 'Remaining answer.')
        self.emit('complete', 'First sentence. Remaining answer.')
        self.assertEqual(len(self.controller.messages), 3)
        self.assertIn('Voice output failed', self.controller.notice)
        self.assertEqual(len(self.jobs), 2)

    def test_synthesis_start_failure_does_not_cancel_model(self):
        self.controller.speech.synthesize = Mock(side_effect=RuntimeError('Piper unavailable'))
        self.emit('delta', 'First phrase. ')
        self.assertTrue(self.turn.voice_failed)
        self.inference.close.assert_not_called()
        self.assertIsNone(self.controller.speech)
        self.emit('complete', 'First phrase.')
        self.assertEqual(len(self.controller.messages), 3)
        self.assertEqual(os.listdir(self.temp.name), [])

    def test_full_mode_defers_all_speech_until_completion(self):
        self.controller.control('cancel')
        self.controller.args.speech_mode = 'full'
        self.controller.start_turn(b'')
        self.turn = self.controller.active
        self.emit('delta', 'First. ')
        self.emit('partial', 'First. ')
        self.emit('delta', 'Second.')
        self.controller.tick()
        self.assertEqual(self.jobs, [])
        self.emit('complete', 'First. Second.')
        self.assertEqual(len(self.jobs), 1)

    def test_cancelled_turn_cannot_feed_next_turn(self):
        self.start_playback()
        self.controller.control('cancel')
        self.controller.start_turn(b'')
        self.emit('delta', 'Late phrase. ')
        self.emit('complete', 'Late phrase.')
        self.assertEqual(self.controller.active.text, '')
        self.assertEqual(self.controller.speech.sequence, 0)
        self.assertEqual(len(self.controller.messages), 1)

    def test_pause_and_late_events_during_overlap(self):
        self.start_playback()
        self.emit('delta', 'Second sentence. ')
        self.controller.control('pause')
        self.assertEqual(self.controller.state, 'Paused')
        self.assertTrue(all(j.closed for j in self.jobs))
        self.emit('complete', 'late')
        self.assertEqual(len(self.controller.messages), 1)
        self.assertEqual(os.listdir(self.temp.name), [])

    def test_lossless_deltas_are_independent_of_caption_throttle(self):
        deltas, captions = [], []
        response = [b'data: {"choices":[{"delta":{"content":"A"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"B"}}]}\n', b'data: [DONE]\n']
        with patch.object(live.urllib.request, 'urlopen') as urlopen, patch.object(live.time, 'monotonic', return_value=0):
            urlopen.return_value.__enter__.return_value = response
            live.chat_completion([], on_delta=deltas.append, on_partial=captions.append)
        self.assertEqual(deltas, ['A', 'B'])
        self.assertEqual(captions, ['A'])

    def test_partial_stream_failure_never_retries_visible_answer(self):
        with patch.object(live.urllib.request, 'urlopen') as urlopen:
            urlopen.return_value.__enter__.return_value = [
                b'data: {"choices":[{"delta":{"content":"Already spoken. "}}]}\n']
            with self.assertRaises(RuntimeError):
                live.chat_completion([], on_delta=Mock())
            self.assertEqual(urlopen.call_count, 1)


if __name__ == '__main__':
    unittest.main()
