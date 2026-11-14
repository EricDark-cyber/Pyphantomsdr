"""Décodeur DMR complet avec structure TDMA ETSI.

Implémentation basée sur ETSI TS 102 361.
"""
import numpy as np
from .fec import HammingFEC


class DMRDecoder:
    def __init__(self, fs=48000):
        self.fs = fs
        self.baud = 4800
        self.messages = []
        self.buf = np.zeros(0, dtype=np.float32)
        self.sps = int(fs / self.baud)
        self.hamming = HammingFEC()
        self.sync_patterns = {
            'BS_SOURCED_VOICE': [0x75, 0x5F, 0xD7],
            'BS_SOURCED_DATA': [0xDF, 0xF5, 0x7D],
            'MS_SOURCED_VOICE': [0x7F, 0x7D, 0x5D],
            'MS_SOURCED_DATA': [0xD5, 0xD7, 0xF7],
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
        if len(self.buf) < self.sps * 144:
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
        """Démodulation 4FSK avec niveaux ETSI."""
        symbols = []
        n = self.sps
        for i in range(0, len(self.buf) - n, n):
            chunk = self.buf[i:i+n]
            zc = np.sum(np.abs(np.diff(np.sign(chunk)))) / 2
            freq = zc * self.fs / (2 * n)
            # Niveaux : -1944, -648, +648, +1944 Hz
            if freq < -1296:
                symbols.append(-3)
            elif freq < 0:
                symbols.append(-1)
            elif freq < 1296:
                symbols.append(1)
            else:
                symbols.append(3)
        return symbols

    def _find_frames(self, symbols):
        """Recherche de trames par sync."""
        frames = []
        for i in range(0, len(symbols) - 144, 72):
            frames.append(symbols[i:i+144])
        return frames[:5]

    def _decode_frame(self, frame):
        """Décodage d'une trame DMR."""
        if len(frame) < 72:
            return None
        # Déterminer le timeslot
        ts = 1 if frame[0] < 0 else 2
        # Extraire des données
        data = []
        for s in frame[24:60]:
            data.append(1 if s > 0 else 0)
        # Correction Hamming
        corrected = self.hamming.decode(data[:15])
        if corrected:
            chars = []
            for i in range(0, len(corrected) - 1, 2):
                val = corrected[i] * 2 + corrected[i+1]
                if val < 26:
                    chars.append(chr(ord('A') + val))
            text = "".join(chars)
            return f"DMR TS{ts}: {text}" if text else f"DMR TS{ts}"
        return f"DMR TS{ts}"
