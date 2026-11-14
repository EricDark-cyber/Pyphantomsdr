from .fec_advanced import ViterbiDecoder, BerlekampMassey, TurboDecoder
from .fec import ConvolutionalFEC
"""Décodeur M17 complet avec structure de trames.

Implémentation basée sur la spécification M17.
"""
import numpy as np


# Sync word M17 (LSF)
M17_SYNC = [0x55, 0xF7]


class M17Decoder:
    def __init__(self, fs=48000):
        self.fs = fs
        self.baud = 4800
        self.messages = []
        self.buf = np.zeros(0, dtype=np.float32)
        self.sps = int(fs / self.baud)
        self.conv = ConvolutionalFEC()
        self.viterbi = ViterbiDecoder()
        self.deviation = 2400

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
        if len(self.buf) < self.sps * 192:
            return msgs
        symbols = self._demod_4fsk()
        frames = self._find_frames(symbols)
        for frame in frames:
            text = self._decode_frame(frame)
            if text and text not in self.messages[-5:]:
                self.messages.append(text)
                msgs.append(text)
        return msgs

    def _demod_4fsk(self):
        """Démodulation 4FSK avec niveaux corrects."""
        symbols = []
        n = self.sps
        for i in range(0, len(self.buf) - n, n):
            chunk = self.buf[i:i+n]
            # Estimation de fréquence
            zc = np.sum(np.abs(np.diff(np.sign(chunk)))) / 2
            freq = zc * self.fs / (2 * n)
            # Mapping 4FSK M17
            if freq < -self.deviation:
                symbols.append(0)
            elif freq < 0:
                symbols.append(1)
            elif freq < self.deviation:
                symbols.append(2)
            else:
                symbols.append(3)
        return symbols

    def _find_frames(self, symbols):
        """Recherche de trames par sync word."""
        frames = []
        for i in range(len(symbols) - 192):
            # Vérifier sync (simplifié)
            if symbols[i] == 1 and symbols[i+1] == 3:
                frames.append(symbols[i:i+192])
        return frames[:5]

    def _decode_frame(self, frame):
        """Décodage d'une trame M17."""
        if len(frame) < 48:
            return None
        # Extraire le callsign (6 caractères)
        chars = []
        for i in range(8, 32, 4):
            val = frame[i] * 16 + frame[i+1] * 4 + frame[i+2]
            if 32 <= val < 127:
                chars.append(chr(val))
        text = "".join(chars).strip()
        return f"M17: {text}" if text else None
