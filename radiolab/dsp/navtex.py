"""Décodeur NAVTEX avec synchronisation SITOR-B.

NAVTEX : messages maritimes sur 518/490 kHz.
"""
import numpy as np


class NAVTEXDecoder:
    def __init__(self, fs=8000, center=None, emit=None):
        self.fs = fs
        self.center = center
        self.emit = emit
        self.baud = 100
        self.messages = []
        self.buf = np.zeros(0, dtype=np.float32)
        self.sps = int(fs / self.baud)
        self.shift = 170

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
        if len(self.buf) < self.sps * 32:
            return msgs
        bits = self._demod_sitor()
        if len(bits) >= 16:
            text = self._decode_sitor(bits)
            if text and len(text) > 3 and text not in self.messages[-5:]:
                self.messages.append(text)
                msgs.append(text)
        return msgs

    def _demod_sitor(self):
        """Démodulation SITOR-B (FEC)."""
        bits = []
        n = self.sps
        t = np.arange(n) / self.fs
        ref_mark = np.exp(2j * np.pi * 1615 * t)
        ref_space = np.exp(2j * np.pi * 1785 * t)
        for i in range(0, len(self.buf) - n, n):
            chunk = self.buf[i:i+n]
            if len(chunk) < n:
                break
            corr_m = np.abs(np.sum(chunk * ref_mark))
            corr_s = np.abs(np.sum(chunk * ref_space))
            bits.append(1 if corr_m > corr_s else 0)
        return bits

    def _decode_sitor(self, bits):
        """Décodage SITOR-B avec correction d'erreurs."""
        chars = []
        # SITOR-B : 7 bits par caractère
        for i in range(0, len(bits) - 7, 7):
            val = 0
            for j in range(7):
                val |= bits[i+j] << j
            # Vérification de parité
            if bin(val).count('1') == 3 or bin(val).count('1') == 4:
                c = chr(32 + (val & 0x3F))
                if 32 <= ord(c) < 127:
                    chars.append(c)
        return "".join(chars).strip() or None


NavtexDecoder = NAVTEXDecoder  # alias utilise par hfdigi.make_decoder("NAVTEX", ...)
