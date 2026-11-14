"""Enregistrement IQ (.cf32 + metadonnees JSON) et audio (.wav)."""
from __future__ import annotations

import json
import time
import wave

import numpy as np


class IqRecorder:
    def __init__(self, path: str, center_freq: float, sample_rate: float):
        self.path = path
        self.f = open(path, "wb")
        meta = {
            "format": "cf32",
            "center_freq_hz": float(center_freq),
            "sample_rate_hz": float(sample_rate),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(path + ".json", "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)

    def write(self, iq: np.ndarray) -> None:
        self.f.write(np.asarray(iq, dtype=np.complex64).tobytes())

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass


class AudioRecorder:
    def __init__(self, path: str, rate: int):
        self.w = wave.open(path, "wb")
        self.w.setnchannels(1)
        self.w.setsampwidth(2)
        self.w.setframerate(int(rate))

    def write(self, audio: np.ndarray) -> None:
        a = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
        self.w.writeframes((a * 32767.0).astype("<i2").tobytes())

    def close(self) -> None:
        try:
            self.w.close()
        except Exception:
            pass
