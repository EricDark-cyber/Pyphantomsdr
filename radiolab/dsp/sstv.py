"""Décodeur SSTV avec synchronisation VIS.

SSTV : Slow Scan Television.
"""
import numpy as np


class SSTVDecoder:
    def __init__(self, fs=8000):
        self.fs = fs
        self.messages = []
        self.buf = np.zeros(0, dtype=np.float32)
        self.vis_codes = {
            0x02: "Robot 36",
            0x04: "Robot 72",
            0x08: "Scottie 1",
            0x0C: "Scottie 2",
            0x20: "Martin 1",
            0x24: "Martin 2",
        }

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
        vis = self._detect_vis()
        if vis is not None:
            mode = self.vis_codes.get(vis, f"Mode {vis}")
            if mode not in self.messages[-3:]:
                self.messages.append(mode)
                msgs.append(f"SSTV: {mode}")
        return msgs

    def _detect_vis(self):
        """Détection du code VIS."""
        if len(self.buf) < self.fs:
            return None
        # Recherche de la calibration (1200 Hz)
        n = int(self.fs * 0.03)
        t = np.arange(n) / self.fs
        ref_1200 = np.exp(2j * np.pi * 1200 * t)
        for i in range(0, len(self.buf) - n * 10, n):
            chunk = self.buf[i:i+n]
            if len(chunk) < n:
                break
            corr = np.abs(np.sum(chunk * ref_1200))
            if corr > n * 0.5:
                # VIS trouvé, décoder
                return self._decode_vis(i)
        return None

    def _decode_vis(self, start):
        """Décode le code VIS."""
        bits = []
        n = int(self.fs * 0.03)
        t = np.arange(n) / self.fs
        ref_1100 = np.exp(2j * np.pi * 1100 * t)
        ref_1300 = np.exp(2j * np.pi * 1300 * t)
        for i in range(8):
            idx = start + n * (i + 1)
            if idx + n > len(self.buf):
                break
            chunk = self.buf[idx:idx+n]
            corr_0 = np.abs(np.sum(chunk * ref_1300))
            corr_1 = np.abs(np.sum(chunk * ref_1100))
            bits.append(1 if corr_1 > corr_0 else 0)
        if len(bits) >= 7:
            val = 0
            for i, b in enumerate(bits[:7]):
                val |= b << i
            return val
        return None
