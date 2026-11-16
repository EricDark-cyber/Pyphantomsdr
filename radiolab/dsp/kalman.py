"""Filtre de Kalman pour RadioLab.

Estimation récursive du signal utile en présence de bruit.
Utile pour lisser un signal ou suivre une porteuse.
"""
import numpy as np


class KalmanFilter1D:
    """Filtre de Kalman scalaire pour le lissage audio."""

    def __init__(self, process_noise=1e-4, measurement_noise=1e-2):
        self.q = float(process_noise)      # bruit de processus
        self.r = float(measurement_noise)  # bruit de mesure
        self.x = 0.0                       # état estimé
        self.p = 1.0                       # variance de l'erreur

    def reset(self):
        self.x = 0.0
        self.p = 1.0

    def update(self, z):
        """Met à jour l'estimation avec une nouvelle mesure z."""
        # Prédiction
        p_pred = self.p + self.q
        # Gain de Kalman
        k = p_pred / (p_pred + self.r)
        # Correction
        self.x = self.x + k * (z - self.x)
        self.p = (1 - k) * p_pred
        return self.x

    def process(self, audio):
        """Applique le filtre à un tableau audio."""
        if audio is None or len(audio) == 0:
            return audio
        out = np.empty_like(audio, dtype=np.float32)
        for i, z in enumerate(audio):
            out[i] = self.update(float(z))
        return out


class KalmanNR:
    """Réduction de bruit par filtre de Kalman pour RadioLab."""

    def __init__(self, fs=48000):
        self.fs = int(fs)
        self.enabled = False
        self.kf = KalmanFilter1D(process_noise=1e-5, measurement_noise=1e-3)
        self.strength = 0.5

    @property
    def available(self):
        return True

    def set_enabled(self, on):
        self.enabled = bool(on)
        if on:
            self.kf.reset()

    def set_strength(self, s):
        """0 = pas de filtrage, 1 = filtrage maximal."""
        self.strength = float(np.clip(s, 0.0, 1.0))
        # Ajuster le bruit de mesure selon la force
        self.kf.r = 1e-4 + (1 - self.strength) * 1e-2
        self.kf.q = 1e-6 + self.strength * 1e-4

    def process(self, audio):
        if not self.enabled or audio is None or len(audio) == 0:
            return audio
        filtered = self.kf.process(audio)
        # Mélange signal original / filtré selon la force
        return ((1 - self.strength) * audio + self.strength * filtered).astype(np.float32)
