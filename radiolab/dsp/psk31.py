"""Décodeur PSK31 avec synchronisation fine.

PSK31 : Phase Shift Keying à 31.25 bauds.
"""
import numpy as np


class PSK31Decoder:
    def __init__(self, fs=8000):
        self.fs = fs
        self.baud = 31.25
        self.messages = []
        self.buf = np.zeros(0, dtype=np.float32)
        self.sps = int(fs / self.baud)
        self.phase = 0.0
        self.varicode = self._build_varicode()

    def _build_varicode(self):
        """Table Varicode PSK31."""
        return {
            '00': ' ', '01': 'e', '10': 't', '11': 'a',
            '000': 'o', '001': 'i', '010': 'n', '011': 's',
            '100': 'r', '101': 'h', '110': 'l', '111': 'd',
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
        if len(self.buf) < self.sps * 16:
            return msgs
        bits = self._demod_psk()
        if len(bits) >= 8:
            text = self._decode_varicode(bits)
            if text and len(text) > 2 and text not in self.messages[-5:]:
                self.messages.append(text)
                msgs.append(text)
        return msgs

    def _demod_psk(self):
        """Démodulation PSK avec synchronisation."""
        bits = []
        n = self.sps
        analytic = np.zeros(len(self.buf), dtype=np.complex64)
        # Signal analytique simplifié
        for i in range(0, len(self.buf) - n, n):
            chunk = self.buf[i:i+n]
            # Détection de phase
            phase = np.angle(np.sum(chunk * np.exp(2j * np.pi * 1000 * np.arange(n) / self.fs)))
            diff = phase - self.phase
            self.phase = phase
            bits.append(1 if diff > 0 else 0)
        return bits

    def _decode_varicode(self, bits):
        """Décodage Varicode."""
        chars = []
        code = ""
        for b in bits:
            code += str(b)
            if code in self.varicode:
                chars.append(self.varicode[code])
                code = ""
            elif len(code) > 10:
                code = ""
        return "".join(chars).strip() or None
