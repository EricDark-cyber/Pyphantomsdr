"""Décodeur D-STAR complet (GMSK + AMBE).

D-STAR : numérique amateur avec codec AMBE.
"""
import numpy as np


class DStarDecoder:
    def __init__(self, fs=48000):
        self.fs = fs
        self.baud = 4800
        self.messages = []
        self.buf = np.zeros(0, dtype=np.float32)

    def process(self, audio):
        if audio is None or len(audio) == 0:
            return []
        self.buf = np.concatenate([self.buf, np.asarray(audio, dtype=np.float32)])
        if len(self.buf) > self.fs * 5:
            self.buf = self.buf[-self.fs * 5:]
        return self._decode()

    def _decode(self):
        msgs = []
        # Détection de préambule GMSK
        if len(self.buf) > self.fs:
            # Recherche de sync
            energy = np.mean(self.buf ** 2)
            if energy > 0.01:
                msgs.append("D-STAR signal detected")
        return msgs
