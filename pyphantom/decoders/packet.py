"""Décodeur Packet/APRS avec HDLC complet.

AX.25 : protocole packet radio.
"""
import numpy as np


class PacketDecoder:
    def __init__(self, fs=48000, baud=1200):
        self.fs = fs
        self.baud = baud
        self.messages = []
        self.buf = np.zeros(0, dtype=np.float32)
        self.sps = int(fs / baud)
        self.flag = [0, 1, 1, 1, 1, 1, 1, 0]  # Flag HDLC

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
        if len(self.buf) < self.sps * 64:
            return msgs
        bits = self._demod_afsk()
        frames = self._find_frames(bits)
        for frame in frames:
            text = self._decode_ax25(frame)
            if text and text not in self.messages[-5:]:
                self.messages.append(text)
                msgs.append(text)
        return msgs

    def _demod_afsk(self):
        """Démodulation AFSK 1200/2200 Hz."""
        bits = []
        n = self.sps
        t = np.arange(n) / self.fs
        ref_mark = np.exp(2j * np.pi * 1200 * t)
        ref_space = np.exp(2j * np.pi * 2200 * t)
        for i in range(0, len(self.buf) - n, n):
            chunk = self.buf[i:i+n]
            corr_m = np.abs(np.sum(chunk * ref_mark))
            corr_s = np.abs(np.sum(chunk * ref_space))
            bits.append(1 if corr_m > corr_s else 0)
        return bits

    def _find_frames(self, bits):
        """Recherche de trames HDLC."""
        frames = []
        i = 0
        while i < len(bits) - 8:
            if bits[i:i+8] == self.flag:
                # Début de trame
                frame = []
                i += 8
                while i < len(bits) - 8:
                    if bits[i:i+8] == self.flag:
                        break
                    frame.append(bits[i])
                    i += 1
                if len(frame) > 16:
                    frames.append(frame)
            i += 1
        return frames[:5]

    def _decode_ax25(self, frame):
        """Décodage AX.25."""
        # Destuffing
        bits = []
        ones = 0
        for b in frame:
            if ones == 5 and b == 0:
                ones = 0
                continue
            bits.append(b)
            ones = ones + 1 if b else 0
        # Conversion en texte
        chars = []
        for i in range(0, len(bits) - 8, 8):
            val = 0
            for j in range(8):
                val = (val << 1) | bits[i+j]
            val >>= 1  # AX.25 shift
            if 32 <= val < 127:
                chars.append(chr(val))
        return "".join(chars).strip() or None
