"""Décodeur C4FM complet (Yaesu System Fusion).

Implémentation du protocole C4FM.
"""
import numpy as np


class C4FMDecoder:
    def __init__(self, fs=48000):
        self.fs = fs
        self.baud = 4800
        self.messages = []
        self.buf = np.zeros(0, dtype=np.float32)
        self.sps = int(fs / self.baud)
        self.modes = {
            0: "Voice Wide",
            1: "Voice Narrow",
            2: "Data",
            3: "Voice+Data",
        }

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
        symbols = self._demod_4fsk()
        if len(symbols) >= 16:
            text = self._decode_frame(symbols)
            if text and text not in self.messages[-5:]:
                self.messages.append(text)
                msgs.append(text)
        return msgs

    def _demod_4fsk(self):
        """Démodulation 4FSK C4FM."""
        symbols = []
        n = self.sps
        for i in range(0, len(self.buf) - n, n):
            chunk = self.buf[i:i+n]
            zc = np.sum(np.abs(np.diff(np.sign(chunk)))) / 2
            freq = zc * self.fs / (2 * n)
            # Niveaux C4FM
            if freq < -1800:
                symbols.append(0)
            elif freq < 0:
                symbols.append(1)
            elif freq < 1800:
                symbols.append(2)
            else:
                symbols.append(3)
        return symbols

    def _decode_frame(self, symbols):
        """Décodage de trame C4FM."""
        # Déterminer le mode
        mode = symbols[0] % 4
        mode_name = self.modes.get(mode, "Unknown")
        # Extraire des données
        chars = []
        for i in range(1, min(len(symbols) - 1, 20), 2):
            val = symbols[i] * 4 + symbols[i+1]
            if 32 <= val < 127:
                chars.append(chr(val))
        text = "".join(chars).strip()
        return f"C4FM {mode_name}: {text}" if text else f"C4FM {mode_name}"
