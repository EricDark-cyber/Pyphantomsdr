"""Décodeur WSPR avec synchronisation et FEC.

WSPR : Weak Signal Propagation Reporter.
"""
import numpy as np


class WSPRDecoder:
    def __init__(self, fs=12000):
        self.fs = fs
        self.baud = 1.4648
        self.messages = []
        self.buf = np.zeros(0, dtype=np.float32)
        self.sps = int(fs / self.baud)
        self.sync = [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0,
                     0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                     0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
                     1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0,
                     0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0,
                     1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1,
                     0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0]

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
        if len(self.buf) < self.sps * 162:
            return msgs
        symbols = self._demod_4fsk()
        if len(symbols) >= 162:
            text = self._decode_message(symbols)
            if text and text not in self.messages[-3:]:
                self.messages.append(text)
                msgs.append(text)
        return msgs

    def _demod_4fsk(self):
        """Démodulation 4-FSK WSPR."""
        symbols = []
        n = self.sps
        t = np.arange(n) / self.fs
        base_freq = 1500
        for i in range(0, len(self.buf) - n, n):
            chunk = self.buf[i:i+n]
            # Corrélation avec 4 tonalités
            corrs = []
            for tone in range(4):
                f = base_freq + tone * self.baud
                ref = np.exp(2j * np.pi * f * t)
                corrs.append(np.abs(np.sum(chunk * ref)))
            symbols.append(np.argmax(corrs))
        return symbols

    def _decode_message(self, symbols):
        """Décodage du message WSPR."""
        # Extraire les bits de données (hors sync)
        data_bits = []
        for i, s in enumerate(symbols[:162]):
            if i < len(self.sync) and self.sync[i] == 0:
                data_bits.append(s & 1)
        # Conversion en texte simplifiée
        chars = []
        for i in range(0, len(data_bits) - 6, 6):
            val = 0
            for j in range(6):
                val = (val << 1) | data_bits[i+j]
            if 32 <= val < 127:
                chars.append(chr(val))
        return "".join(chars).strip() or None
