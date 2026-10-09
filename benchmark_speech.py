"""Compare speech modes with real Piper/ffplay; optionally use a running model.

The default replays fixed text timings, isolating speech scheduling from model
variability. --audio sends a prerecorded WAV and the current screen to the local
model instead. Neither mode opens the microphone. Both modes play audible speech.
"""
import argparse
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import wave

import live_gemma as live


class ReplayInference:
    def __init__(self, turn_id, *unused):
        self.turn_id = turn_id
        self.start = time.monotonic()
        self.index = 0
        self.fragments = [(0.2, 'Hello there. '), (2.5, 'This is a voice timing check. '),
                          (5.0, 'The test is complete.')]

    def poll(self):
        events = []
        elapsed = time.monotonic() - self.start
        while self.index < len(self.fragments) and elapsed >= self.fragments[self.index][0]:
            events.append((self.turn_id, 'delta', self.fragments[self.index][1]))
            self.index += 1
        if self.index == len(self.fragments) and elapsed >= 5.2:
            events.append((self.turn_id, 'complete', ''.join(t for _, t in self.fragments)))
            self.index += 1
        return events

    def close(self):
        pass


def metrics(records):
    def first(stage):
        return next(r['at'] for r in records if r['stage'] == stage)
    starts = [r['at'] for r in records if r['stage'] == 'playback_start']
    ends = [r['at'] for r in records if r['stage'] == 'playback_end']
    if not starts or any(r['stage'] == 'voice_failed' for r in records):
        raise RuntimeError('Voice output failed; latency sample is invalid')
    origin = first('utterance_detected')
    return dict(first_playback_s=starts[0] - origin,
                total_s=first('turn_complete') - origin,
                max_gap_s=max([max(0, start - end) for start, end in zip(starts[1:], ends)] or [0]),
                playback_before_generation_done=starts[0] < first('generation_complete'))


def run(mode, pcm, server_url, piper, use_model):
    args = SimpleNamespace(device='unused', silence=1, threshold=.01, test=True,
                           no_tts=False, server_url=server_url, piper_path=piper, speech_mode=mode)
    # Discard conversational output. Only numeric stage timings enter the report.
    with redirect_stdout(io.StringIO()):
        controller = live.LiveController(args, inference_factory=None if use_model else ReplayInference)
        try:
            controller.start_turn(pcm)
            turn = controller.active
            deadline = time.monotonic() + 180
            while controller.active and time.monotonic() < deadline:
                controller.tick()
                time.sleep(.03)
            if controller.active or not controller.last_success:
                raise RuntimeError('Benchmark turn did not complete: ' + controller.notice)
            return metrics(turn.timings)
        finally:
            controller.close()


def summarize(rows):
    summary = {}
    for key in ('first_playback_s', 'total_s', 'max_gap_s'):
        values = sorted(row[key] for row in rows)
        summary[key] = dict(median=round(statistics.median(values), 3),
                            p95=round(values[math.ceil(.95 * len(values)) - 1], 3))
    summary['overlapping_runs'] = sum(row['playback_before_generation_done'] for row in rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, help='Use a real model request with this mono 16 kHz WAV')
    parser.add_argument('--server-url', default=live.SERVER_URL)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('speech-benchmark.json'))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be positive')
    piper = live.resolve_piper()
    if not piper or not live.shutil.which(live.FFPLAY):
        parser.error('Piper and ffplay must be installed')
    pcm = b''
    if args.audio:
        with wave.open(str(args.audio), 'rb') as source:
            if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 16000):
                parser.error('Audio must be mono, 16-bit PCM at 16000 Hz')
            pcm = source.readframes(source.getnframes())
    rows = {'full': [], 'streamed': []}
    for iteration in range(args.runs + 1):
        # Alternate order to reduce systematic cache/order bias; discard warm-up.
        for mode in (('full', 'streamed') if iteration % 2 == 0 else ('streamed', 'full')):
            print(f'{mode}: {"warm-up" if iteration == 0 else "sample " + str(iteration)}', flush=True)
            sample = run(mode, pcm, args.server_url, piper, bool(args.audio))
            if iteration:
                rows[mode].append(sample)
    report = dict(source='live model' if args.audio else 'fixed text replay; real Piper and ffplay',
                  runs_per_mode=args.runs, warmups_per_mode=1,
                  note='Playback timestamps measure process launch, not acoustic onset. Small samples are not a stable p95 estimate.',
                  results={mode: summarize(samples) for mode, samples in rows.items()}, samples=rows)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['results'], indent=2))


if __name__ == '__main__':
    live.multiprocessing.freeze_support()
    main()
