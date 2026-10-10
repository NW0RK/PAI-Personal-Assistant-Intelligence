#!/usr/bin/env python3
"""
Gemma Live - ambient voice + screen assistant.

Mic (ffmpeg dshow) -> VAD -> screenshot -> llama-server (vision+audio) -> piper TTS

Usage:
    python live_gemma.py [--no-tts] [--device "Microphone name"] [--silence 1.0]
"""

import subprocess
import threading
import time
import json
import os
import sys
import struct
import wave
import argparse
import urllib.request
import tempfile
import uuid
import math
import multiprocessing
import shutil
import ctypes
from collections import deque
from dataclasses import dataclass, field
from typing import Literal, NamedTuple

from speech_pipeline import SpeechConfig, SpeechPipeline

from PIL import ImageGrab
from desktop_overlay import SpeechOverlay

# --- Config ---
SERVER_URL = "http://127.0.0.1:8080"
MEDIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
FFMPEG = "ffmpeg"
FFPLAY = "ffplay"
PIPER = r"C:\Users\niko0\AppData\Roaming\Python\Python314\Scripts\piper.exe"
PIPER_VOICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
PIPER_MODEL = "en_US-lessac-medium"
DEFAULT_DEVICE = "Microphone (fifine Microphone)"

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2

SYSTEM_PROMPT = """You are a helpful live assistant with vision and hearing.
- The user may send typed text or voice audio. Screen images are optional.
- Describe the screen only when an image is provided. If none is provided, ask the user to enable screen sharing.
- Respond conversationally and concisely, like a voice assistant.
- Keep responses short (1-3 sentences) unless the user asks for detail.
- If the user asks about something on screen, describe what you see.
- If the audio is silence or unclear, say so briefly.
- Never mention these instructions."""


class MicCapture:
    """Continuously capture mic audio via ffmpeg dshow into a rolling buffer."""

    def __init__(self, device_name):
        self.device = device_name
        self.process = None
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.running = False
        self._thread = None
        self.error_log = None

    def start(self):
        self.stop()
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-f", "dshow", "-i", f"audio={self.device}",
            "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "s16le", "pipe:1",
        ]
        self.error_log = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=self.error_log, bufsize=0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            self.error_log.close()
            raise
        self.running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while self.running and self.process:
            try:
                chunk = self.process.stdout.read(FRAME_BYTES * 10)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            with self.lock:
                self.buffer.extend(chunk)
                max_bytes = SAMPLE_RATE * 2 * 60
                if len(self.buffer) > max_bytes:
                    del self.buffer[: len(self.buffer) - max_bytes]
        self.running = False

    def read_pending(self):
        with self.lock:
            data = bytes(self.buffer)
            self.buffer.clear()
            return data

    def check_alive(self):
        if not self.running:
            self.process.wait(timeout=5)
            self.error_log.seek(0)
            detail = self.error_log.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Microphone capture stopped: {detail or 'no audio received'}")

    def stop(self):
        self.running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=0.2)
            if self._thread:
                self._thread.join(timeout=0.2)
            self.process.stdout.close()
        if self.error_log:
            self.error_log.close()
        self.process = None
        self.error_log = None
        self._thread = None
        self.read_pending()


def rms_energy(frame_bytes):
    n = len(frame_bytes) // 2
    if n == 0:
        return 0.0
    total = 0
    for i in range(0, n * 2, 2):
        sample = struct.unpack_from("<h", frame_bytes, i)[0]
        total += sample * sample
    return (total / n) ** 0.5 / 32768.0


class SpeechDetector:
    """Keep speech and incomplete PCM frames across microphone reads."""

    def __init__(self, silence_sec, energy_threshold):
        self.max_silence = max(1, math.ceil(silence_sec * 1000 / FRAME_MS))
        self.min_speech = math.ceil(0.5 * 1000 / FRAME_MS)
        self.max_frames = 45 * 1000 // FRAME_MS
        self.threshold = energy_threshold
        self.pending = bytearray()
        self.frames = []
        self.pre_roll = deque(maxlen=200 // FRAME_MS)
        self.silence_count = 0
        self.voiced_count = 0

    def feed(self, pcm_bytes):
        self.pending.extend(pcm_bytes)
        utterances = []
        complete_bytes = len(self.pending) // FRAME_BYTES * FRAME_BYTES
        for offset in range(0, complete_bytes, FRAME_BYTES):
            frame = bytes(self.pending[offset:offset + FRAME_BYTES])
            voiced = rms_energy(frame) > self.threshold
            if not self.frames:
                if not voiced:
                    self.pre_roll.append(frame)
                    continue
                self.frames = list(self.pre_roll)
                self.pre_roll.clear()
            self.frames.append(frame)
            self.voiced_count += int(voiced)
            self.silence_count = 0 if voiced else self.silence_count + 1
            if self.silence_count >= self.max_silence or len(self.frames) >= self.max_frames:
                if self.voiced_count >= self.min_speech:
                    end = len(self.frames) - self.silence_count
                    utterances.append(b"".join(self.frames[:end]))
                self.frames.clear()
                self.silence_count = 0
                self.voiced_count = 0
        del self.pending[:complete_bytes]
        return utterances


def capture_screenshot(path):
    im = ImageGrab.grab()
    im.thumbnail((1280, 1280))
    im.convert("RGB").save(path, "JPEG", quality=75)


def write_wav(path, pcm_bytes):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm_bytes)


def chat_completion(messages, max_tokens=512, on_partial=None, server_url=None, on_delta=None):
    """Stream a chat completion. Returns (content, reasoning_content)."""
    result = _stream_completion(messages, max_tokens, on_partial=on_partial, server_url=server_url, on_delta=on_delta)
    if not result[0].strip():
        print(f"[Reply warning] {result[2]}; retrying once...", flush=True)
        result = _stream_completion(messages, max_tokens * 2, temperature=0.2,
                                    on_partial=on_partial, server_url=server_url, on_delta=on_delta)
    if not result[0].strip():
        raise RuntimeError(f"Gemma returned no answer after retry: {result[2]}")
    return result[0].strip(), result[1]


def _stream_completion(messages, max_tokens, temperature=0.7, on_partial=None, server_url=None, on_delta=None):
    """Keep stream termination details so an empty reply is diagnosable."""
    body = json.dumps({
        "model": "gemma",
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": temperature,
        # Voice turns need an answer within the budget, without a thinking phase.
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
        "stream_options": {"include_usage": True},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url or SERVER_URL}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    content_parts = []
    reasoning_parts = []
    finish_reason = None
    completion_tokens = None
    completed = False
    last_partial = float('-inf')
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                completed = True
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError as e:
                raise RuntimeError("Invalid JSON in Gemma response stream") from e
            if obj.get("error"):
                raise RuntimeError(str(obj["error"]))
            usage = obj.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                completion_tokens = usage["completion_tokens"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            if choices[0].get("finish_reason") is not None:
                finish_reason = choices[0]["finish_reason"]
            delta = choices[0].get("delta") or {}
            c = delta.get("content")
            if c:
                content_parts.append(c)
                if on_delta:
                    on_delta(c)
                now = time.monotonic()
                if on_partial and now - last_partial >= 0.1:
                    on_partial("".join(content_parts))
                    last_partial = now
            r = delta.get("reasoning_content")
            if r:
                reasoning_parts.append(r)
    if not completed:
        raise RuntimeError("Gemma response stream ended before [DONE]")
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    detail = (f"finish_reason={finish_reason or 'unknown'}, "
              f"completion_tokens={completion_tokens if completion_tokens is not None else 'unknown'}, "
              f"reasoning_chars={len(reasoning)}")
    return content, reasoning, detail


class TextDelta(NamedTuple):
    turn_id: str
    kind: str
    text: str


@dataclass
class TurnContext:
    id: str
    paths: tuple[str, ...]
    user: dict
    text: str = ""
    committed: bool = False
    generation: Literal["running", "complete", "failed", "cancelled"] = "running"
    audio_state: str = "Thinking"
    received_delta: bool = False
    voice_failed: bool = False
    timings: list = field(default_factory=list)
    resume_state: str = "Listening"


def inference_worker(connection, turn_id, messages, pcm, paths, server_url):
    """Own blocking screen/network work; the controller can terminate this process."""
    try:
        if pcm is not None:
            connection.send((turn_id, "stage", "Saving your recording"))
            write_wav(paths[0], pcm)
        image_path = next((path for path in paths if path.endswith('.jpg')), None)
        if image_path:
            connection.send((turn_id, "stage", "Capturing the screen"))
            capture_screenshot(image_path)
        connection.send((turn_id, "stage", "Waiting for the local model"))
        def partial(text):
            connection.send((turn_id, "partial", text))
        def delta(text):
            connection.send(TextDelta(turn_id, "delta", text))
        connection.send((turn_id, "timing", ("request_start", time.monotonic())))
        content, _ = chat_completion(messages, on_partial=partial, server_url=server_url, on_delta=delta)
        connection.send((turn_id, "complete", content))
    except Exception as exc:
        connection.send((turn_id, "failure", str(exc)))
    finally:
        connection.close()


class InferenceJob:
    def __init__(self, turn_id, messages, pcm, paths, server_url):
        context = multiprocessing.get_context("spawn")
        self.connection, sender = context.Pipe(duplex=False)
        self.process = context.Process(target=inference_worker,
            args=(sender, turn_id, messages, pcm, paths, server_url), daemon=True)
        self.turn_id = turn_id
        self.ended = False
        self.last_progress = time.monotonic()
        try:
            self.process.start()
        except Exception:
            self.connection.close()
            raise
        finally:
            sender.close()

    def poll(self):
        events = []
        try:
            while self.connection.poll():
                event = self.connection.recv()
                self.last_progress = time.monotonic()
                events.append(event)
                if event[1] in ("complete", "failure"):
                    self.ended = True
        except (EOFError, OSError):
            if not self.ended:
                events.append((self.turn_id, "failure", "Inference worker stopped unexpectedly"))
                self.ended = True
        if not self.ended and time.monotonic() - self.last_progress > 45:
            events.append((self.turn_id, "failure", "No inference progress for 45 seconds; the request was cancelled. Check the local server."))
            self.ended = True
        return events

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=0.2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=0.2)
        self.connection.close()
        self.process.close()


class AudioJob:
    """Pollable audio subprocess with file-backed input and diagnostics."""
    def __init__(self, command, timeout, text=None):
        self.error = tempfile.TemporaryFile()
        self.input = tempfile.TemporaryFile() if text is not None else None
        if self.input:
            self.input.write(text.encode("utf-8"))
            self.input.seek(0)
        self.deadline = time.monotonic() + timeout
        try:
            self.process = subprocess.Popen(command, stdin=self.input or subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=self.error,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            self.error.close()
            if self.input:
                self.input.close()
            raise

    def poll(self):
        code = self.process.poll()
        if code is None:
            if time.monotonic() >= self.deadline:
                raise RuntimeError("Audio operation timed out")
            return False
        if code:
            self.error.seek(0)
            detail = self.error.read().decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"Audio process exited with code {code}: {detail}")
        return True

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=0.2)
        self.error.close()
        if self.input:
            self.input.close()


def resolve_piper(explicit=None):
    for candidate in (explicit, shutil.which("piper"), PIPER):
        if candidate:
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
    return None


class LiveController:
    def __init__(self, args, overlay=None, *, inference_factory=None, audio_factory=None, clock=None):
        self.args, self.overlay = args, overlay
        self.inference_factory = inference_factory or InferenceJob
        self.audio_factory = audio_factory or AudioJob
        self.clock = clock or time.monotonic
        self.mic = MicCapture(args.device)
        self.state = "Starting"
        self.notice = ""
        self.quitting = False
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.turn_media = deque()
        self.owned = set()
        self.active = None
        self.inference = None
        self.speech = None
        self.last_success = False
        self.detector = SpeechDetector(args.silence, args.threshold)
        # Older programmatic callers keep their explicit audio+screen workflow.
        self.screen_sharing = getattr(args, "screen_sharing", True)
        self.last_meter = 0
        self.voice_warning = ""

    def emit(self, event, **fields):
        print(json.dumps(dict(event=event, **fields), ensure_ascii=False), flush=True)

    def status(self, state, notice=None):
        self.state = state
        if self.active:
            self.active.audio_state = self.speech.state if self.speech else "Thinking"
        if notice is not None:
            self.notice = notice
        snapshot = {"event": "status", "state": state, "detail": self.notice,
                    "turn_id": self.active.id if self.active else None,
                    "generation": self.active.generation if self.active else None,
                    "audio": self.active.audio_state if self.active else None}
        print(json.dumps(snapshot, ensure_ascii=False), flush=True)
        if self.overlay:
            self.overlay.status(state, self.notice)

    def show(self, text, duration=None):
        if self.overlay:
            self.overlay.show(text, duration=duration)

    def mark(self, event, *, timestamp=None, **fields):
        if self.active:
            record = dict(event="timing", turn_id=self.active.id, stage=event,
                          at=self.clock() if timestamp is None else timestamp, **fields)
            self.active.timings.append(record)
            print(json.dumps(record), flush=True)

    def update_speech(self, poll=True):
        if not self.active:
            return
        if self.speech:
            try:
                if poll:
                    self.speech.tick()
                else:
                    self.speech.schedule()
            except Exception as exc:
                self.fail_turn(str(exc), voice=True)
        if not self.active:
            return
        if self.active.committed and (not self.speech or self.speech.drained):
            self.finish_turn()
            return
        state = self.speech.state if self.speech else "Thinking"
        self.active.audio_state = state
        if state != self.state:
            self.status(state)

    def remove_media(self, paths):
        for path in paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(f"[Cleanup warning] {path}: {exc}", flush=True)
                continue
            self.owned.discard(path)

    def listen(self):
        if self.quitting or self.state == "Paused":
            return
        if getattr(self.args, "text_only", False):
            self.status("Paused", "Text session ready. Listening is paused.")
            return
        if self.args.test:
            self.status("Listening")
            return
        self.detector = SpeechDetector(self.args.silence, self.args.threshold)
        try:
            notice = self.notice
            self.status("Starting", "Starting microphone")
            if not self.args.device:
                raise ValueError("Select a microphone in Settings to enable listening.")
            self.mic.start()
            self.status("Listening", notice)
        except Exception as exc:
            self.mic.stop()
            print(f"[Microphone error] {exc}", flush=True)
            self.emit("failure", kind="microphone", code="microphone_unavailable", detail=str(exc), turn_id=None)
            self.status("Error", "Microphone unavailable. Open microphone settings or send a typed message.")

    def start_text_turn(self, text, request_id=None):
        if not isinstance(text, str) or not text.strip() or self.active or self.quitting:
            self.emit("failure", kind="input", code="input_unavailable",
                      detail="Finish or stop the current reply before sending a non-empty message.", request_id=request_id)
            return
        self.start_turn(None, text=text, request_id=request_id)

    def start_turn(self, pcm, *, text=None, request_id=None):
        if self.active or (text is None and self.state in ("Paused", "Error")) or self.quitting:
            return
        resume_state = "Paused" if self.state in ("Paused", "Error") else "Listening"
        detected_at = self.clock()
        self.mic.stop()
        turn_id = uuid.uuid4().hex
        suffixes = (() if pcm is None else (".wav",)) + ((".jpg",) if self.screen_sharing else ())
        paths = tuple(os.path.join(MEDIA_DIR, turn_id + suffix) for suffix in suffixes)
        self.owned.update(paths)
        user = {"role": "user", "content": [
            {"type": "text", "text": text or ""},
        ]}
        if self.screen_sharing:
            user["content"].append({"type": "image_url", "image_url": {"url": "file://" + os.path.basename(paths[-1])}})
        if pcm is not None:
            user["content"].append({"type": "input_audio", "input_audio": {"url": "file://" + os.path.basename(paths[0])}})
        self.active = TurnContext(turn_id, paths, user, resume_state=resume_state)
        self.notice = ""
        self.emit("turn_started", turn_id=turn_id, screen_sharing=self.screen_sharing,
                  input_kind="text" if text is not None else "voice", input_text=text or "", request_id=request_id)
        self.emit("microphone", level=0, speech=False)
        self.mark("utterance_detected", timestamp=detected_at)
        if not self.args.no_tts:
            def chunk_path(index):
                path = os.path.join(MEDIA_DIR, f"{turn_id}-reply-{index}.wav")
                self.owned.add(path)
                return path
            self.speech = SpeechPipeline(
                SpeechConfig(mode=getattr(self.args, "speech_mode", "streamed")),
                lambda text, path: self.audio_factory([self.args.piper_path, "--model", PIPER_MODEL,
                    "--data-dir", PIPER_VOICE_DIR, "--output_file", path], 60, text),
                lambda path: self.audio_factory([FFPLAY, "-nodisp", "-autoexit", "-loglevel", "error", path], 120),
                chunk_path, self.remove_media, self.mark)
        self.last_success = False
        if self.overlay:
            self.overlay.hide()
        self.status("Thinking")
        try:
            self.inference = self.inference_factory(turn_id, self.messages + [user], pcm, paths, self.args.server_url)
        except Exception as exc:
            self.fail_turn(str(exc))

    def handle_event(self, event):
        turn_id, kind, text = event
        turn = self.active
        if not turn or turn_id != turn.id or turn.committed:
            return
        if kind == "partial":
            turn.text = text
            self.show(text)
            self.emit("partial", turn_id=turn_id, text=text)
        elif kind == "delta":
            if not turn.received_delta and text.strip():
                self.mark("first_text")
            turn.received_delta = turn.received_delta or bool(text.strip())
            turn.text += text
            if self.speech:
                self.speech.feed(text)
                self.update_speech(poll=False)
        elif kind == "timing":
            self.mark(text[0], timestamp=text[1])
        elif kind == "stage":
            self.status(self.speech.state if self.speech else "Thinking", text)
        elif kind == "failure":
            self.fail_turn(text)
        elif kind == "complete":
            if self.inference:
                self.inference.close()
                self.inference = None
            turn.text = text
            turn.committed = True
            turn.generation = "complete"
            self.mark("generation_complete")
            self.last_success = True
            self.messages.extend([turn.user, {"role": "assistant", "content": text}])
            self.turn_media.append(turn.paths)
            if len(self.turn_media) > 10:
                self.messages = [self.messages[0]] + self.messages[-20:]
                self.remove_media(self.turn_media.popleft())
            print(f"[Gemma] {text}", flush=True)
            if not turn.voice_failed:
                self.notice = ""
            self.show(text)
            self.emit("reply", turn_id=turn_id, text=text, complete=True)
            if self.speech:
                if not turn.received_delta:
                    self.speech.feed(text)
                self.speech.finish()
            self.update_speech(poll=False)
            if self.active:
                self.status(self.state)

    def fail_turn(self, detail, voice=False):
        print(f"[{'Voice' if voice else 'Turn'} error] {detail}", flush=True)
        self.notice = ("Voice output failed; the answer is in captions. Check Piper and audio output."
                       if voice else "Reply failed. Check the local server and try speaking again. " + detail[:180])
        self.emit("failure", turn_id=self.active.id if self.active else None,
                  kind="voice" if voice else "server", code="voice_unavailable" if voice else "reply_failed", detail=self.notice)
        if voice:
            if self.active:
                self.active.voice_failed = True
            if self.speech:
                self.speech.close()
                self.speech = None
            self.mark("voice_failed")
            if self.active and not self.active.committed:
                self.status("Thinking", self.notice)
                return
        elif self.active:
            self.active.generation = "failed"
        self.finish_turn()

    def finish_turn(self, resume=True):
        if self.inference:
            self.inference.close()
            self.inference = None
        if self.speech:
            self.speech.close()
            self.speech = None
        if self.active:
            if self.active.generation == "running":
                self.active.generation = "cancelled"
            self.mark("turn_complete", outcome=self.active.generation)
        turn, self.active = self.active, None
        if turn:
            self.emit("turn_finished", turn_id=turn.id, text=turn.text,
                      complete=turn.committed, outcome=turn.generation)
            if turn.text:
                suffix = "" if turn.committed else " (incomplete)"
                self.show(turn.text + suffix, duration=8)
            if not turn.committed:
                self.remove_media(turn.paths)
        if resume:
            if turn and turn.resume_state == "Paused":
                self.status("Paused", self.notice)
            else:
                self.listen()

    def control(self, action):
        if self.quitting:
            return
        if isinstance(action, dict):
            if action.get("action") == "screen_sharing" and isinstance(action.get("enabled"), bool):
                self.screen_sharing = action["enabled"]
                self.emit("screen_sharing", enabled=self.screen_sharing)
            elif action.get("action") == "send_text":
                self.start_text_turn(action.get("text"), action.get("request_id"))
            elif action.get("action") == "caption_position":
                self.args.caption_position = action.get("position", "Top left")
                if self.overlay:
                    self.overlay.position(self.args.caption_position)
            elif action.get("action") == "configure":
                self.configure(action)
            return
        if action == "clear":
            paused = self.state == "Paused" or bool(self.active and self.active.resume_state == "Paused")
            self.mic.stop()
            self.finish_turn(resume=False)
            self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            self.turn_media.clear()
            self.remove_media(tuple(self.owned))
            if self.overlay:
                self.overlay.hide()
            self.emit("cleared")
            self.status("Paused" if paused else "Starting", "")
            if not paused:
                self.listen()
            return
        if action == "quit":
            self.quitting = True
            self.mic.stop()
            self.finish_turn(resume=False)
        elif action == "pause" and self.state != "Paused":
            self.mic.stop()
            self.finish_turn(resume=False)
            self.status("Paused")
        elif action == "resume" and self.state in ("Paused", "Error"):
            self.args.text_only = False
            self.status("Starting", "")
            self.listen()
        elif action == "cancel" and self.active:
            self.finish_turn()

    def configure(self, settings):
        from app_settings import validate_preferences
        try:
            preferences = validate_preferences(settings.get("preferences", {}))
        except ValueError as exc:
            self.emit("failure", kind="settings", code="settings_invalid", detail=str(exc), turn_id=None)
            return
        paused = self.state == "Paused" or bool(self.active and self.active.resume_state == "Paused")
        self.mic.stop()
        self.finish_turn(resume=False)
        self.args.device = preferences["device"]
        if not self.args.device:
            self.args.text_only = True
        self.mic.device = self.args.device
        self.args.silence = preferences["silence"]
        self.args.threshold = preferences["threshold"]
        self.args.no_tts = not preferences["voice"]
        self.voice_warning = configure_voice(self.args)
        if not preferences["captions"] and self.overlay:
            self.overlay.close()
            self.overlay = None
        elif preferences["captions"] and not self.overlay:
            try:
                self.overlay = SpeechOverlay(captions_only=getattr(self.args, "captions_only", False),
                                             caption_position=getattr(self.args, "caption_position", "Top left"))
            except Exception as exc:
                self.emit("failure", kind="captions", code="captions_unavailable", detail=f"Floating captions unavailable: {exc}", turn_id=None)
        self.emit("configured", preferences=preferences, voice_available=not self.args.no_tts,
                  request_id=settings.get('request_id'), warning=self.voice_warning)
        self.status("Paused" if paused else "Starting", "")
        if not paused:
            self.listen()

    def tick(self):
        if self.overlay:
            for action in self.overlay.poll_controls():
                self.control(action)
        if self.quitting:
            return
        if self.inference:
            for event in self.inference.poll():
                self.handle_event(event)
        self.update_speech()
        if self.state == "Listening" and not self.args.test:
            try:
                self.mic.check_alive()
                pcm = self.mic.read_pending()
                utterances = self.detector.feed(pcm)
                if self.clock() - self.last_meter >= 0.1:
                    self.emit("microphone", level=rms_energy(pcm), speech=bool(self.detector.frames))
                    self.last_meter = self.clock()
                if utterances:
                    self.start_turn(utterances[0])
            except Exception as exc:
                self.mic.stop()
                print(f"[Microphone error] {exc}", flush=True)
                self.emit("failure", kind="microphone", code="microphone_unavailable", detail=str(exc))
                self.status("Error", "Microphone stopped. Open microphone settings or send a typed message.")

    def close(self):
        self.control("quit")
        self.remove_media(tuple(self.owned))
        if self.overlay:
            self.overlay.close()


class TerminalCommands:
    """Poll the launcher's pipe without a blocking buffered stdin reader.

    On Windows, a thread blocked in buffered stdin can hold a CRT lock needed
    when multiprocessing opens its spawn pipe. Keep all command reads here.
    """
    def __init__(self, stream=None):
        self.fd = (stream or sys.stdin).fileno()
        self.pending = bytearray()
        self.closed = False
        if os.name == "nt":
            import msvcrt
            self.handle = msvcrt.get_osfhandle(self.fd)
            self.peek = ctypes.windll.kernel32.PeekNamedPipe
            self.peek.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
                                  ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p]
            self.peek.restype = ctypes.c_int

    def poll(self):
        if self.closed:
            return []
        try:
            if os.name == "nt":
                available = ctypes.c_ulong()
                if not self.peek(self.handle, None, 0, None, ctypes.byref(available), None):
                    self.closed = True
                    return ["quit"]
                if not available.value:
                    return []
                count = min(available.value, 4096)
            else:
                import select
                if not select.select([self.fd], [], [], 0)[0]:
                    return []
                count = 4096
            data = os.read(self.fd, count)
        except OSError:
            data = b""
        if not data:
            self.closed = True
            return ["quit"]
        self.pending.extend(data)
        actions = []
        while b"\n" in self.pending:
            line, _, remaining = self.pending.partition(b"\n")
            self.pending = bytearray(remaining)
            action = line.decode("utf-8", errors="replace").strip()
            if action in ("pause", "resume", "cancel", "quit", "clear"):
                actions.append(action)
            elif action.startswith("{"):
                try:
                    command = json.loads(action)
                    if isinstance(command, dict):
                        actions.append(command)
                except ValueError:
                    pass
        return actions


def configure_voice(args):
    if args.no_tts:
        return ""
    args.piper_path = resolve_piper(args.piper_path)
    missing = []
    if not args.piper_path:
        missing.append("Piper")
    if not shutil.which(FFPLAY):
        missing.append("ffplay")
    if not all(os.path.isfile(os.path.join(PIPER_VOICE_DIR, PIPER_MODEL + ext))
               for ext in (".onnx", ".onnx.json")):
        missing.append("voice model files")
    if missing:
        args.no_tts = True
        return "Voice disabled: missing " + ", ".join(missing) + ". Replies remain visible."
    return ""


def main():
    parser = argparse.ArgumentParser(description="Gemma Live assistant")
    parser.add_argument("--speech-mode", choices=("streamed", "full"), default="streamed",
                        help="Speak completed phrases immediately, or wait for the full answer")
    parser.add_argument("--no-tts", action="store_true", help="Disable voice output")
    parser.add_argument("--no-overlay", action="store_true", help="Disable desktop captions and controls")
    parser.add_argument("--captions-only", action="store_true", help="Show floating captions without a second control panel")
    parser.add_argument("--screen-sharing", action="store_true", help="Attach screen images to voice turns (off by default)")
    parser.add_argument("--text-only", action="store_true", help="Start with listening paused; no microphone required")
    parser.add_argument("--caption-position", choices=("Top left", "Top right", "Bottom left", "Bottom right"), default="Top left")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Microphone dshow name")
    parser.add_argument("--list-devices", action="store_true", help="List available cameras and microphones")
    parser.add_argument("--silence", type=float, default=1.0, help="Seconds of silence to end utterance")
    parser.add_argument("--threshold", type=float, default=0.01, help="RMS energy threshold for speech")
    parser.add_argument("--test", action="store_true", help="Inject a test utterance and exit")
    parser.add_argument("--test-audio", help="WAV recording for --test (mono, 16-bit, 16000 Hz)")
    parser.add_argument("--server-url", default=SERVER_URL, help="Local llama-server URL")
    parser.add_argument("--piper-path", help="Piper executable; falls back to PATH and the original location")
    parser.add_argument("--terminal-controls", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.server_url = args.server_url.rstrip("/")
    if not math.isfinite(args.silence) or args.silence <= 0:
        parser.error("--silence must be a positive finite number")
    if not math.isfinite(args.threshold) or not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    if args.list_devices:
        subprocess.run([FFMPEG, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"])
        return 0
    os.makedirs(MEDIA_DIR, exist_ok=True)
    overlay = None
    if not args.no_overlay:
        try:
            overlay = SpeechOverlay(captions_only=args.captions_only, caption_position=args.caption_position)
        except Exception as exc:
            print(f"[Overlay warning] {exc}", flush=True)
    controller = LiveController(args, overlay)
    terminal_commands = TerminalCommands() if args.terminal_controls else None
    try:
        controller.status("Starting")
        # Generation reports server failures without blocking the controller on a health request.
        if not args.test and not args.text_only and not shutil.which(FFMPEG):
            controller.status("Error", "FFmpeg is missing. Install it on PATH, then press Resume.")
        else:
            controller.status("Starting")
        controller.voice_warning = configure_voice(args)
        controller.emit("configured", voice_available=not args.no_tts, warning=controller.voice_warning)
        if args.test:
            with wave.open(args.test_audio or os.path.join(MEDIA_DIR, "test_speech.wav"), "rb") as source:
                if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, SAMPLE_RATE):
                    raise ValueError("Test audio must be mono, 16-bit PCM at 16000 Hz")
                controller.start_turn(source.readframes(source.getnframes()))
        elif controller.state != "Error":
            controller.listen()
        while not controller.quitting:
            if terminal_commands:
                for action in terminal_commands.poll():
                    controller.control(action)
            controller.tick()
            if args.test and not controller.active:
                return 0 if controller.last_success else 1
            if controller.state == "Error" and not overlay and not args.terminal_controls:
                return 1
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("Exiting...", flush=True)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        return 1
    finally:
        controller.close()
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
