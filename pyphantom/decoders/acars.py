"""Décodeur ACARS avec démodulation MSK optimisée.

MSK : Minimum Shift Keying à 2400 bauds.
"""
import numpy as np


class ACARSDecoder:
    def __init__(self, fs=48000):
        self.fs = fs
        self.baud = 2400
        self.messages = []
        self.buf = np.zeros(0, dtype=np.float32)
        self.sps = int(fs / self.baud)
        self.freq_mark = 2400
        self.freq_space = 1200

    def process(self, audio):
        if audio is None or len(audio) == 0:
            return []
        x = np.asarray(audio, dtype=np.float32).flatten()
        self.buf = np.concatenate([self.buf, x])
        if len(self.buf) > self.fs * 10:
            self.buf = self.buf[-self.fs * 10:]
        return self._decode()

    def _decode(self):
        msgs = []
        if len(self.buf) < self.sps * 32:
            return msgs
        bits = self._demod_msk()
        if len(bits) >= 16:
            text = self._decode_frame(bits)
            if text and text not in self.messages[-5:]:
                self.messages.append(text)
                msgs.append(text)
        return msgs

    def _demod_msk(self):
        """Démodulation MSK par corrélation."""
        bits = []
        n = self.sps
        t = np.arange(n) / self.fs
        ref_mark = np.exp(2j * np.pi * self.freq_mark * t)
        ref_space = np.exp(2j * np.pi * self.freq_space * t)
        for i in range(0, len(self.buf) - n, n):
            chunk = self.buf[i:i+n]
            corr_m = np.abs(np.sum(chunk * ref_mark))
            corr_s = np.abs(np.sum(chunk * ref_space))
            bits.append(1 if corr_m > corr_s else 0)
        return bits

    def _decode_frame(self, bits):
        """Décodage de trame ACARS."""
        # Recherche de préambule
        chars = []
        for i in range(0, len(bits) - 8, 8):
            val = 0
            for j in range(8):
                val = (val << 1) | bits[i+j]
            if 32 <= val < 127:
                chars.append(chr(val))
            elif val == 0x03:  # ETX
                break
        return "".join(chars).strip() or None
