"""Ordered, bounded speech scheduling independent of model and UI processes."""
from collections import deque
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SpeechConfig:
    mode: str = "streamed"
    phrase_limit: int = 180
    ready_limit: int = 2

    def __post_init__(self):
        if self.mode not in ("streamed", "full") or self.phrase_limit < 1 or self.ready_limit < 1:
            raise ValueError("Invalid speech configuration")


class PhraseBuffer:
    def __init__(self, config):
        self.config = config
        self.pending = ""
        self.finished = False

    def pop(self):
        text = self.pending.lstrip()
        if not text:
            self.pending = ""
            return None
        if self.config.mode == "full":
            end = len(text) if self.finished else 0
        else:
            boundary = re.search(r"[.!?](?:[\"'’”)]*)\s|\n", text)
            end = boundary.end() if boundary else 0
            limit = self.config.phrase_limit
            if len(text) > limit and (not end or end > limit):
                spaces = list(re.finditer(r"\s", text[:limit + 1]))
                if spaces:
                    end = spaces[-1].end()
                else:
                    # Never cut a token in half, even if it exceeds the limit.
                    space = re.search(r"\s", text[limit:])
                    end = limit + space.end() if space else 0
            if not end and self.finished:
                end = len(text)
        if not end:
            return None
        self.pending = text[end:]
        return text[:end].strip()


class SpeechPipeline:
    """One synthesis and one playback job; ready audio is strictly bounded.

    Factories return jobs with poll() -> bool and close(). The owner calls tick
    regularly and handles exceptions by closing this pipeline and using captions.
    """
    def __init__(self, config, synthesize, play, path_for, remove, mark):
        self.buffer = PhraseBuffer(config)
        self.config = config
        self.synthesize, self.play = synthesize, play
        self.path_for, self.remove, self.mark = path_for, remove, mark
        self.ready = deque()
        self.synthesis = None
        self.playback = None
        self.synthesis_path = self.playback_path = None
        self.paths = set()
        self.sequence = 0
        self.closed = False

    def feed(self, text):
        if not self.closed:
            self.buffer.pending += text

    def finish(self):
        self.buffer.finished = True

    @property
    def state(self):
        if self.playback:
            return "Speaking"
        if self.synthesis or self.ready:
            return "Preparing voice"
        return "Thinking"

    @property
    def drained(self):
        return self.closed or (self.buffer.finished and not self.buffer.pending.strip()
                               and not self.synthesis and not self.playback and not self.ready)

    def schedule(self):
        if self.closed or self.synthesis or len(self.ready) >= self.config.ready_limit:
            return
        phrase = self.buffer.pop()
        if phrase:
            self.sequence += 1
            self.mark("phrase_ready", chunk=self.sequence)
            path = self.path_for(self.sequence)
            self.paths.add(path)
            self.synthesis_path = path
            self.synthesis = self.synthesize(phrase, path)

    def tick(self):
        if self.closed:
            return
        if self.playback and self.playback.poll():
            self.playback.close()
            self.playback = None
            self.mark("playback_end")
            self.remove([self.playback_path])
            self.paths.discard(self.playback_path)
            self.playback_path = None
        if self.synthesis and self.synthesis.poll():
            self.synthesis.close()
            self.synthesis = None
            self.mark("synthesis_complete", chunk=self.sequence)
            self.ready.append((self.sequence, self.synthesis_path))
            self.synthesis_path = None
        if not self.playback and self.ready:
            chunk, path = self.ready.popleft()
            self.playback_path = path
            self.playback = self.play(path)
            self.mark("playback_start", chunk=chunk)
        self.schedule()

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Attempt all cleanup even if one adapter fails to stop.
        try:
            if self.synthesis:
                self.synthesis.close()
        finally:
            try:
                if self.playback:
                    self.playback.close()
            finally:
                self.synthesis = self.playback = None
                self.ready.clear()
                self.buffer.pending = ""
                self.remove(tuple(self.paths))
                self.paths.clear()
