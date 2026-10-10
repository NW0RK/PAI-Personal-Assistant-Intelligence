"""Small UI utilities and a bounded, model-free microphone calibration."""
import ctypes
import math
import os
import threading
import time


def work_areas(root):
    areas = []
    if os.name == 'nt':
        class Rect(ctypes.Structure):
            _fields_ = [(name, ctypes.c_long) for name in ('left', 'top', 'right', 'bottom')]
        class Info(ctypes.Structure):
            _fields_ = [('size', ctypes.c_ulong), ('monitor', Rect), ('work', Rect), ('flags', ctypes.c_ulong)]
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.POINTER(Rect), ctypes.c_void_p)
        def collect(monitor, dc, rect, data):
            info = Info()
            info.size = ctypes.sizeof(info)
            ctypes.windll.user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(Info)]
            if ctypes.windll.user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                r = info.work
                area = (r.left, r.top, r.right, r.bottom)
                if info.flags & 1:
                    areas.insert(0, area)
                else:
                    areas.append(area)
            return 1
        callback = callback_type(collect)
        ctypes.windll.user32.EnumDisplayMonitors(None, None, callback, None)
    return areas or [(0, 0, root.winfo_screenwidth(), root.winfo_screenheight())]


def clamp_geometry(rectangle, areas):
    x, y, width, height = rectangle
    def overlap(area):
        left, top, right, bottom = area
        return max(0, min(x + width, right) - max(x, left)) * max(0, min(y + height, bottom) - max(y, top))
    left, top, right, bottom = max(areas, key=overlap)
    width, height = min(width, right - left), min(height, bottom - top)
    return [max(left, min(x, right - width)), max(top, min(y, bottom - height)), width, height]


def caption_geometry(position, width, height, area, margin=24):
    left, top, right, bottom = area
    width, height = min(width, right - left), min(height, bottom - top)
    x = right - width - margin if position.endswith('right') else left + margin
    y = bottom - height - margin if position.startswith('Bottom') else top + margin
    return clamp_geometry([x, y, width, height], [area])


def markdown_conversation(turns):
    parts = ['# Gemma Live conversation', '']
    for turn in turns.values():
        parts.extend(['## You', '', turn.get('input_text', '') if turn.get('input_kind') == 'text' else '*Voice message (no transcript)*', ''])
        if turn.get('screen'):
            parts.extend(['*Screen shared with this message.*', ''])
        parts.extend(['## Gemma', '', turn['text'] or '*No reply.*', ''])
        if not turn['complete']:
            parts.extend(['*Incomplete reply.*', ''])
    return '\n'.join(parts)


def suggested_threshold(noise, speech):
    """Keep the cutoff between the noise floor and clear speech; reject bad samples."""
    if len(noise) < 10 or len(speech) < 10:
        raise ValueError('Not enough microphone data. Check the device and try again.')
    if any(not math.isfinite(n) or n < 0 for n in noise + speech):
        raise ValueError('Invalid microphone levels. Rescan microphones and try again.')
    noise_floor = sorted(noise)[int((len(noise) - 1) * .95)]
    voice_level = sorted(speech)[int((len(speech) - 1) * .8)]
    if voice_level < .0003 or voice_level <= noise_floor * 1.8:
        raise ValueError('Speech was too quiet or background noise was too loud. Move closer and retry.')
    if voice_level > .9:
        raise ValueError('The microphone is clipping. Lower its input volume and retry.')
    return round(min(.5, max(.0001, math.sqrt(max(noise_floor, .00005) * voice_level))), 5)


class Calibration:
    def __init__(self, device, output, *, capture_factory=None, clock=time.monotonic):
        from live_gemma import MicCapture
        self.capture = (capture_factory or MicCapture)(device)
        self.output = output
        self.clock = clock
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run, name='Microphone calibration', daemon=False)

    def start(self):
        self.thread.start()

    def emit(self, **fields):
        self.output.put(dict(event='calibration', **fields))

    def run(self):
        from live_gemma import FRAME_BYTES, rms_energy
        samples = {}
        try:
            self.capture.start()
            for stage, seconds in (('noise', 3), ('speech', 5)):
                self.emit(stage=stage)
                levels = []
                pending = b''
                deadline = self.clock() + seconds
                while self.clock() < deadline:
                    if self.cancelled.wait(.05):
                        self.emit(stage='cancelled')
                        return
                    self.capture.check_alive()
                    pending += self.capture.read_pending()
                    while len(pending) >= FRAME_BYTES:
                        levels.append(rms_energy(pending[:FRAME_BYTES]))
                        pending = pending[FRAME_BYTES:]
                    self.emit(stage='level', level=levels[-1] if levels else 0)
                samples[stage] = levels
            self.emit(stage='result', threshold=suggested_threshold(samples['noise'], samples['speech']))
        except Exception as exc:
            self.emit(stage='error', detail=str(exc))
        finally:
            try:
                self.capture.stop()
            except Exception as exc:
                self.emit(stage='error', detail=f'Microphone cleanup failed: {exc}')
            finally:
                self.done.set()
