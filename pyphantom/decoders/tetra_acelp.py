"""Décodeur voix TETRA ACELP amélioré.

Version plus complète, inspirée de la structure ACELP TETRA (ETSI EN 300 395-2)
et des approches utilisées dans les décodeurs open-source.
"""
import numpy as np


class TetraACELP:
    """Décodeur ACELP TETRA (frame 30 ms = 137 bits → 240 échantillons @ 8 kHz)."""

    FRAME_BITS = 137
    FRAME_SAMPLES = 240
    SAMPLE_RATE = 8000
    LPC_ORDER = 10

    def __init__(self):
        self.reset()

    def reset(self):
        self.mem_syn = np.zeros(self.LPC_ORDER)
        self.pitch_lag = 50
        self.pitch_gain = 0.6
        self.fixed_gain = 0.4
        self.prev_exc = np.zeros(143)  # mémoire excitation

    def decode_frame(self, bits):
        """Décode une trame de 137 bits."""
        if len(bits) < self.FRAME_BITS:
            return np.zeros(self.FRAME_SAMPLES, dtype=np.float32)

        bits = np.asarray(bits[:self.FRAME_BITS], dtype=np.int8)

        # --- Extraction des paramètres (simplifiée mais structurée) ---
        # Pitch lag (8 bits)
        lag_idx = self._bits_to_int(bits[0:8])
        self.pitch_lag = 20 + (lag_idx % 123)

        # Pitch gain (4 bits)
        pg = self._bits_to_int(bits[8:12])
        self.pitch_gain = 0.2 + (pg / 15.0) * 0.7

        # Fixed codebook gain (5 bits)
        fg = self._bits_to_int(bits[12:17])
        self.fixed_gain = 0.05 + (fg / 31.0) * 0.8

        # --- Génération de l'excitation ---
        exc = np.zeros(self.FRAME_SAMPLES, dtype=np.float32)

        # Adaptive codebook (pitch)
        for i in range(self.FRAME_SAMPLES):
            idx = i - self.pitch_lag
            if idx >= 0:
                exc[i] = self.pitch_gain * exc[idx]
            elif len(self.prev_exc) + idx >= 0:
                exc[i] = self.pitch_gain * self.prev_exc[len(self.prev_exc) + idx]

        # Fixed codebook (bruit structuré)
        pulse_pos = self._bits_to_int(bits[17:25]) % self.FRAME_SAMPLES
        for p in range(4):  # 4 pulses
            pos = (pulse_pos + p * 60) % self.FRAME_SAMPLES
            sign = 1.0 if (bits[25 + p] if 25 + p < len(bits) else 0) else -1.0
            exc[pos] += sign * self.fixed_gain

        # --- Filtre de synthèse LPC ---
        # Coefficients LPC simplifiés (dans un vrai décodeur ils sont décodés)
        a = np.array([
            1.0,
            -0.85, 0.35, -0.15, 0.08,
            -0.04, 0.02, -0.01, 0.005,
            -0.002, 0.001
        ])

        audio = np.zeros(self.FRAME_SAMPLES, dtype=np.float32)
        mem = list(self.mem_syn)
        for n in range(self.FRAME_SAMPLES):
            s = exc[n]
            for k in range(1, len(a)):
                if n >= k:
                    s -= a[k] * audio[n - k]
                else:
                    s -= a[k] * mem[self.LPC_ORDER - k + n]
            audio[n] = np.clip(s, -1.0, 1.0)

        # Mise à jour des mémoires
        self.mem_syn = audio[-self.LPC_ORDER:]
        self.prev_exc = np.concatenate([self.prev_exc[-100:], exc])[-143:]

        # Post-filtre léger + normalisation
        audio = self._postfilter(audio)
        peak = np.max(np.abs(audio)) + 1e-8
        return (audio / peak * 0.7).astype(np.float32)

    def _bits_to_int(self, bits):
        v = 0
        for b in bits:
            v = (v << 1) | int(b)
        return v

    def _postfilter(self, x):
        """Post-filtre simple pour améliorer la qualité perceptuelle."""
        # Léger renforcement des formants
        y = x.copy()
        for i in range(1, len(y)):
            y[i] += 0.15 * y[i - 1]
        return y

    def process_bits(self, bits):
        """Traite un flux de bits et retourne l'audio."""
        bits = list(bits)
        frames = []
        while len(bits) >= self.FRAME_BITS:
            frames.append(self.decode_frame(bits[:self.FRAME_BITS]))
            bits = bits[self.FRAME_BITS:]
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames)
