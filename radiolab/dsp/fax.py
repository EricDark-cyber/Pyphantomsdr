"""Décodeur FAX météo avec synchronisation IOC/APT.

FAX : Facsimile, cartes météo HF.
"""
import numpy as np


class FAXDecoder:
    def __init__(self, fs=8000):
        self.fs = fs
        self.messages = []
        self.buf = np.zeros(0, dtype=np.float32)
        self.ioc = 576  # Index of Cooperation
        self.rpm = 120  # Rotations per minute

    def process(self, audio):
        if audio is None or len(audio) == 0:
            return []
        x = np.asarray(audio, dtype=np.float32).flatten()
        self.buf = np.concatenate([self.buf, x])
        if len(self.buf) > self.fs * 60:
            self.buf = self.buf[-self.fs * 60:]
        return self._decode()

    def _decode(self):
        msgs = []
        if self._detect_start():
            if "FAX start" not in self.messages[-3:]:
                self.messages.append("FAX start")
                msgs.append("FAX: transmission détectée")
        if self._detect_stop():
            if "FAX stop" not in self.messages[-3:]:
                self.messages.append("FAX stop")
                msgs.append("FAX: fin de transmission")
        return msgs

    def _detect_start(self):
        """Détection du signal APT start (300 Hz)."""
        if len(self.buf) < self.fs:
            return False
        n = int(self.fs * 0.5)
        t = np.arange(n) / self.fs
        ref = np.exp(2j * np.pi * 300 * t)
        chunk = self.buf[:n]
        corr = np.abs(np.sum(chunk * ref))
        return corr > n * 0.3

    def _detect_stop(self):
        """Détection du signal APT stop (450 Hz)."""
        if len(self.buf) < self.fs * 2:
            return False
        n = int(self.fs * 0.5)
        t = np.arange(n) / self.fs
        ref = np.exp(2j * np.pi * 450 * t)
        chunk = self.buf[-n:]
        corr = np.abs(np.sum(chunk * ref))
        return corr > n * 0.3
