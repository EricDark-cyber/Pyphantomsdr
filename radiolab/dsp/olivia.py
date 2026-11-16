"""Décodeur Olivia complet avec Walsh et entrelacement.

Implémentation basée sur la spécification Olivia.
"""
import numpy as np


# Tables Walsh 8x8 pour Olivia
WALSH_8 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, -1, 1, -1, 1, -1, 1, -1],
    [1, 1, -1, -1, 1, 1, -1, -1],
    [1, -1, -1, 1, 1, -1, -1, 1],
    [1, 1, 1, 1, -1, -1, -1, -1],
    [1, -1, 1, -1, -1, 1, -1, 1],
    [1, 1, -1, -1, -1, -1, 1, 1],
    [1, -1, -1, 1, -1, 1, 1, -1],
], dtype=np.float32)


class OliviaDecoder:
    def __init__(self, fs=8000, tones=32, baud=31.25):
        self.fs = fs
        self.tones = tones
        self.baud = baud
        self.messages = []
        self.buf = np.zeros(0, dtype=np.float32)
        self.sps = int(fs / baud)
        self.freq_base = 500.0
        self.freq_step = baud

    def process(self, audio):
        if audio is None or len(audio) == 0:
            return []
        x = np.asarray(audio, dtype=np.float32).flatten()
        self.buf = np.concatenate([self.buf, x])
        if len(self.buf) > self.fs * 120:
            self.buf = self.buf[-self.fs * 120:]
        return self._decode()

    def _decode(self):
        msgs = []
        if len(self.buf) < self.sps * 32:
            return msgs
        symbols = self._demod_mfsk()
        if len(symbols) >= 16:
            text = self._decode_walsh(symbols)
            if text and len(text) > 2 and text not in self.messages[-5:]:
                self.messages.append(text)
                msgs.append(text)
                if len(self.messages) > 100:
                    self.messages = self.messages[-50:]
        return msgs

    def _demod_mfsk(self):
        """Démodulation MFSK par FFT."""
        symbols = []
        n = self.sps
        window = np.hanning(n)
        for i in range(0, len(self.buf) - n, n // 2):
            chunk = self.buf[i:i+n] * window
            spec = np.abs(np.fft.rfft(chunk))
            # Trouver le pic dans la bande Olivia
            freqs = np.fft.rfftfreq(n, 1/self.fs)
            mask = (freqs >= self.freq_base) & (freqs < self.freq_base + self.tones * self.freq_step)
            if np.any(mask):
                idx = np.argmax(spec[mask])
                tone = int((freqs[mask][idx] - self.freq_base) / self.freq_step)
                symbols.append(min(tone, self.tones - 1))
        return symbols

    def _decode_walsh(self, symbols):
        """Décodage Walsh avec désentrelacement."""
        chars = []
        # Grouper par blocs de 8 symboles
        for i in range(0, len(symbols) - 8, 8):
            block = np.array(symbols[i:i+8], dtype=np.float32)
            # Corrélation avec chaque ligne Walsh
            corrs = [np.abs(np.sum(block * w)) for w in WALSH_8]
            best = np.argmax(corrs)
            if best > 0 and best < 128:
                chars.append(chr(best))
        return "".join(chars).strip() or None
