"""Filtre de Kalman Étendu (EKF) pour RadioLab.

Pour les systèmes non linéaires : suivi de phase, estimation
de canal, ou lissage avec modèle non linéaire.
"""
import numpy as np


class ExtendedKalmanFilter:
    """EKF scalaire avec fonction d'observation non linéaire."""

    def __init__(self, process_noise=1e-4, measurement_noise=1e-2):
        self.q = float(process_noise)
        self.r = float(measurement_noise)
        self.x = 0.0   # état estimé
        self.p = 1.0   # variance

    def reset(self):
        self.x = 0.0
        self.p = 1.0

    def _f(self, x):
        """Fonction de transition (identité par défaut)."""
        return x

    def _h(self, x):
        """Fonction d'observation non linéaire.
        Exemple : saturation douce (tanh) pour modéliser un limiteur.
        """
        return np.tanh(x)

    def _H(self, x):
        """Jacobienne de h (dérivée)."""
        return 1.0 - np.tanh(x) ** 2  # d(tanh)/dx = 1 - tanh²

    def update(self, z):
        """Met à jour avec une mesure z (possiblement non linéaire)."""
        # Prédiction
        x_pred = self._f(self.x)
        p_pred = self.p + self.q

        # Jacobienne au point prédit
        H = self._H(x_pred)

        # Innovation
        y = z - self._h(x_pred)

        # Gain de Kalman
        s = H * p_pred * H + self.r
        k = p_pred * H / s

        # Correction
        self.x = x_pred + k * y
        self.p = (1 - k * H) * p_pred
        return self.x

    def process(self, audio):
        if audio is None or len(audio) == 0:
            return audio
        out = np.empty(len(audio), dtype=np.float32)
        for i, z in enumerate(audio):
            out[i] = self.update(float(z))
        return out


class EKFFilter:
    """Interface EKF pour RadioLab."""

    def __init__(self, fs=48000):
        self.fs = int(fs)
        self.enabled = False
        self.ekf = ExtendedKalmanFilter()
        self.strength = 0.5

    @property
    def available(self):
        return True

    def set_enabled(self, on):
        self.enabled = bool(on)
        if on:
            self.ekf.reset()

    def set_params(self, q, r):
        self.ekf.q = float(q)
        self.ekf.r = float(r)

    def process(self, audio):
        if not self.enabled or audio is None or len(audio) == 0:
            return audio
        filtered = self.ekf.process(audio)
        # Mélange
        s = self.strength
        return ((1 - s) * np.asarray(audio) + s * filtered).astype(np.float32)
