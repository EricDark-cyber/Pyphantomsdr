"""Débruiteur simple et fluide pour RadioLab.

Filtre passe-bas adaptatif très léger, sans overlap-add.
Garantit un audio continu.
"""
import numpy as np


class NeuralNR:
    def __init__(self, fs=48000):
        self.fs = int(fs)
        self.enabled = False
        # Filtre simple : moyenne mobile adaptative
        self.alpha = 0.95
        self.state = 0.0
        self.noise_est = 0.01

    @property
    def available(self):
        return True

    def set_enabled(self, on):
        self.enabled = bool(on)
        if on:
            self.state = 0.0
            self.noise_est = 0.01

    def process(self, audio):
        """Filtre sample par sample, toujours la même longueur."""
        if not self.enabled or audio is None or len(audio) == 0:
            return audio

        x = np.asarray(audio, dtype=np.float32).flatten()
        n = len(x)
        out = np.empty(n, dtype=np.float32)

        for i in range(n):
            # Estimation du bruit (valeur absolue lissée)
            self.noise_est = 0.999 * self.noise_est + 0.001 * abs(x[i])
            # Filtre adaptatif : plus de lissage si le signal est faible
            a = self.alpha if abs(x[i]) < self.noise_est * 3 else 0.5
            self.state = a * self.state + (1 - a) * x[i]
            out[i] = self.state

        return out
