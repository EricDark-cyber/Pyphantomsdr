"""Décodeur JS8Call (dérivé de FT8, texte libre).

JS8Call utilise la même modulation que FT8 mais permet
des messages texte libres au lieu de messages structurés.
"""
import numpy as np
from .ft8_ldpc import FT8LDPC
from .ft8_crc import check_crc14


class JS8Decoder:
    """Décodeur JS8Call."""

    def __init__(self, fs=12000):
        self.fs = fs
        self.ldpc = FT8LDPC()
        self.messages = []
        self.buf = np.zeros(0, dtype=np.float32)

    def process(self, audio):
        """Traite un buffer audio et retourne les messages décodés."""
        if audio is None or len(audio) == 0:
            return []
        self.buf = np.concatenate([self.buf, np.asarray(audio, dtype=np.float32)])
        # Garder 15 secondes
        max_len = int(self.fs * 15)
        if len(self.buf) > max_len:
            self.buf = self.buf[-max_len:]
        return self.try_decode()

    def try_decode(self):
        """Tente de décoder un message JS8."""
        # Placeholder : utilise la même structure que FT8
        # mais interprète les bits comme du texte libre
        msgs = []
        # TODO: synchronisation Costas + démodulation 8-FSK
        return msgs

    def decode_text(self, bits):
        """Décode des bits JS8 en texte."""
        if len(bits) < 77:
            return None
        # JS8 encode le texte dans les 77 bits
        chars = []
        for i in range(0, 72, 6):
            if i + 6 > len(bits):
                break
            val = 0
            for b in bits[i:i+6]:
                val = (val << 1) | b
            if 32 <= val < 127:
                chars.append(chr(val))
        return "".join(chars).strip() or None
