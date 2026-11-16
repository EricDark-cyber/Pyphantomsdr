"""Sortie audio temps reel (sounddevice) avec file interne, volume et mute."""
from __future__ import annotations

import queue

import numpy as np


class AudioSink:
    def __init__(self, rate: int = 48000, volume: float = 0.8, muted: bool = False):
        self.rate = int(rate)
        self.volume = float(volume)
        self.muted = bool(muted)
        self.q: queue.Queue = queue.Queue(maxsize=128)
        self._last = None
        self.stream = None
        try:
            import sounddevice as sd

            self.stream = sd.RawOutputStream(
                samplerate=self.rate, channels=1, dtype="float32", callback=self._cb
            )
            self.stream.start()
        except Exception:
            self.stream = None

    def _cb(self, outdata, frames, _time, _status):
        try:
            block = self.q.get_nowait()
        except queue.Empty:
            block = self._last
        out = np.zeros(frames, dtype=np.float32)
        if block is not None:
            self._last = block
            if not self.muted:
                m = min(frames, len(block))
                out[:m] = block[:m] * self.volume
        outdata[:] = out.tobytes()

    def push(self, block: np.ndarray) -> None:
        if self.stream is None or len(block) == 0:
            return
        try:
            self.q.put_nowait(np.asarray(block, dtype=np.float32))
        except queue.Full:
            pass

    def close(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
