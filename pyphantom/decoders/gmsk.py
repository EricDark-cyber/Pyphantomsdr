"""Démodulation GMSK améliorée (utilisée par AIS à 9600 bauds)."""
import numpy as np


class GMSKDemodulator:
    """Démodulateur GMSK avec détection de phase et timing."""

    def __init__(self, fs, baud=9600.0):
        self.fs = float(fs)
        self.baud = float(baud)
        self.sps = self.fs / self.baud
        self.history = np.zeros(0, dtype=np.float32)
        self.timing = 0.0

    def process(self, x):
        if len(x) == 0:
            return []
        x = x.astype(np.float32)
        buf = np.concatenate([self.history, x])

        # Détection de phase différentielle (classique pour GMSK)
        analytic = buf + 1j * np.roll(buf, 1)
        dphi = np.angle(analytic[1:] * np.conj(analytic[:-1]))

        # Lissage léger
        win = max(int(self.sps / 6), 1)
        if win > 1:
            kernel = np.ones(win) / win
            dphi = np.convolve(dphi, kernel, mode="same")

        bits = []
        pos = self.timing
        step = self.sps
        while pos < len(dphi) - 1:
            idx = int(pos)
            a = max(0, idx - 1)
            b = min(len(dphi), idx + 2)
            bit = 1 if np.mean(dphi[a:b]) > 0 else 0
            bits.append(bit)
            pos += step

        self.timing = pos - len(dphi)
        keep = int(self.sps * 6)
        self.history = buf[-keep:] if len(buf) > keep else buf
        return bits
