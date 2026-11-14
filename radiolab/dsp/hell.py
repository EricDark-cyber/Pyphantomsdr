"""Décodeur Hellschreiber avec rendu amélioré.

Hellschreiber : texte transmis comme image.
"""
import numpy as np


class HellDecoder:
    def __init__(self, fs=8000):
        self.fs = fs
        self.baud = 122.5
        self.messages = []
        self.buf = np.zeros(0, dtype=np.float32)
        self.sps = int(fs / self.baud)
        self.chars = self._build_font()

    def _build_font(self):
        """Police Feld-Hell simplifiée."""
        return {
            'A': [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
            'B': [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
            'C': [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
            'E': [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
            ' ': [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        }

    def process(self, audio):
        if audio is None or len(audio) == 0:
            return []
        x = np.asarray(audio, dtype=np.float32).flatten()
        self.buf = np.concatenate([self.buf, x])
        if len(self.buf) > self.fs * 30:
            self.buf = self.buf[-self.fs * 30:]
        return self._decode()

    def _decode(self):
        msgs = []
        if len(self.buf) < self.sps * 49:
            return msgs
        pixels = self._demod_hell()
        if len(pixels) >= 49:
            text = self._recognize(pixels)
            if text and text not in self.messages[-5:]:
                self.messages.append(text)
                msgs.append(text)
        return msgs

    def _demod_hell(self):
        """Démodulation Hellschreiber."""
        pixels = []
        n = self.sps
        for i in range(0, len(self.buf) - n, n):
            chunk = self.buf[i:i+n]
            # Détection de tonalité 900 Hz
            energy = np.mean(chunk ** 2)
            pixels.append(1 if energy > 0.01 else 0)
        return pixels

    def _recognize(self, pixels):
        """Reconnaissance de caractères."""
        chars = []
        for i in range(0, len(pixels) - 49, 49):
            char_pixels = pixels[i:i+49]
            # Comparaison avec la police
            best = ' '
            best_score = 0
            for c, pattern in self.chars.items():
                score = sum(1 for j in range(min(len(char_pixels), len(pattern)))
                           if (char_pixels[j] > 0) == (pattern[j % 7] > 0))
                if score > best_score:
                    best_score = score
                    best = c
            chars.append(best)
        return "".join(chars).strip() or None
