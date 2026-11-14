from .fec_advanced import ViterbiDecoder, BerlekampMassey, TurboDecoder
from .fec import ReedSolomon63
"""Décodeur JT65 complet avec Reed-Solomon.

Implémentation basée sur la spécification JT65.
"""
import numpy as np


class JT65Decoder:
    def __init__(self, fs=12000):
        self.fs = fs
        self.tones = 65
        self.baud = 2.6917
        self.messages = []
        self.buf = np.zeros(0, dtype=np.float32)
        self.sps = int(fs / self.baud)
        self.freq_base = 1270.5
        self.freq_step = 2.6917
        self.rs = ReedSolomon63()
        self.bm = BerlekampMassey()
        self.sync_pos = [1, 2, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]

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
        if len(self.buf) < self.sps * 63:
            return msgs
        symbols = self._demod_65fsk()
        if len(symbols) >= 63:
            data = self.rs.decode(symbols)
            text = self._symbols_to_text(data) if data else None
            if text and len(text) > 2 and text not in self.messages[-3:]:
                self.messages.append(text)
                msgs.append(text)
        return msgs

    def _demod_65fsk(self):
        """Démodulation 65-FSK par FFT."""
        symbols = []
        n = self.sps
        window = np.hanning(n)
        for i in range(0, len(self.buf) - n, n):
            chunk = self.buf[i:i+n] * window
            spec = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(n, 1/self.fs)
            mask = (freqs >= self.freq_base) & (freqs < self.freq_base + self.tones * self.freq_step)
            if np.any(mask):
                idx = np.argmax(spec[mask])
                tone = int((freqs[mask][idx] - self.freq_base) / self.freq_step)
                symbols.append(min(tone, self.tones - 1))
        return symbols

    def _decode_rs(self, symbols):
        """Décodage Reed-Solomon simplifié."""
        # Extraire les symboles de données (hors sync)
        data = []
        for i, s in enumerate(symbols[:63]):
            if i not in self.sync_pos:
                data.append(s)
        # Conversion en texte
        chars = []
        for s in data[:13]:
            if s < 26:
                chars.append(chr(ord('A') + s))
            elif s < 36:
                chars.append(chr(ord('0') + s - 26))
            elif s == 36:
                chars.append(' ')
            elif s == 37:
                chars.append('/')
        return "".join(chars).strip() or None
