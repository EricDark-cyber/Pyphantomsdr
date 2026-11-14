"""Techniques de démodulation TETRA inspirées de GNU Radio.

Inclut :
- AGC
- Récupération de porteuse (Costas-like)
- Récupération de timing améliorée
"""
import numpy as np


class TetraDemodEnhancer:
    """Améliore la démodulation π/4-DQPSK avec des techniques GNU Radio."""

    def __init__(self, sps):
        self.sps = float(sps)
        # AGC
        self.agc_gain = 1.0
        self.agc_ref = 1.0
        self.agc_rate = 0.01
        # Carrier recovery (Costas-like)
        self.phase = 0.0
        self.freq = 0.0
        self.alpha = 0.02   # phase gain
        self.beta = 0.0005  # frequency gain
        # Timing
        self.mu = 0.0
        self.omega = self.sps
        self.gain_mu = 0.05
        self.omega_rel = 0.05

    def process(self, symbols):
        """Applique AGC + correction de porteuse sur des symboles complexes."""
        if len(symbols) == 0:
            return symbols
        out = np.zeros_like(symbols)
        for i, s in enumerate(symbols):
            # AGC
            mag = np.abs(s) + 1e-8
            err = self.agc_ref - mag * self.agc_gain
            self.agc_gain += self.agc_rate * err
            self.agc_gain = np.clip(self.agc_gain, 0.1, 10.0)
            s = s * self.agc_gain

            # Carrier recovery (4th power Costas for π/4-DQPSK)
            s_rot = s * np.exp(-1j * self.phase)
            # Error from 4th power
            p4 = s_rot ** 4
            err_phase = np.angle(p4) / 4.0
            self.freq += self.beta * err_phase
            self.phase += self.freq + self.alpha * err_phase
            self.phase = self.phase % (2 * np.pi)
            out[i] = s_rot
        return out

    def get_freq_error(self):
        """Retourne l'estimation d'erreur de fréquence (rad/symbole)."""
        return self.freq
