"""Démodulation MSK (Minimum Shift Keying) améliorée.

Utilisée notamment par ACARS (2400 bauds).
Méthode : détection de phase différentielle + récupération d'horloge.
"""
import numpy as np


class MSKDemodulator:
    """Démodulateur MSK avec récupération de timing simple."""

    def __init__(self, fs, baud=2400.0):
        self.fs = float(fs)
        self.baud = float(baud)
        self.sps = self.fs / self.baud          # échantillons par symbole
        self.phase = 0.0
        self.timing = 0.0
        self.history = np.zeros(0, dtype=np.float32)

    def process(self, x):
        """Démodule un bloc d'audio et retourne une liste de bits."""
        if len(x) == 0:
            return []
        x = x.astype(np.float32)
        buf = np.concatenate([self.history, x])

        bits = []
        # Filtre passe-bande approximatif autour de la porteuse MSK
        # (les deux fréquences sont f_c ± baud/4)
        n = len(buf)
        t = np.arange(n) / self.fs
        # Détection non cohérente par quadrature
        i = buf * np.cos(2 * np.pi * self.baud / 4 * t)
        q = buf * np.sin(2 * np.pi * self.baud / 4 * t)
        # Lissage
        win = max(int(self.sps / 4), 1)
        if win > 1:
            kernel = np.ones(win) / win
            i = np.convolve(i, kernel, mode="same")
            q = np.convolve(q, kernel, mode="same")

        # Phase différentielle
        analytic = i + 1j * q
        dphi = np.angle(analytic[1:] * np.conj(analytic[:-1]))

        # Échantillonnage au rythme symbole
        step = self.sps
        pos = self.timing
        while pos < len(dphi) - 1:
            idx = int(pos)
            # Moyenne sur une petite fenêtre autour du point d'échantillonnage
            a = max(0, idx - 1)
            b = min(len(dphi), idx + 2)
            bit = 1 if np.mean(dphi[a:b]) > 0 else 0
            bits.append(bit)
            pos += step

        # Conserver l'état
        self.timing = pos - len(dphi)
        keep = int(self.sps * 4)
        self.history = buf[-keep:] if len(buf) > keep else buf
        return bits


def demod_msk(audio, fs, baud=2400.0):
    """Fonction utilitaire : démodule un bloc complet."""
    dem = MSKDemodulator(fs, baud)
    return dem.process(audio)
