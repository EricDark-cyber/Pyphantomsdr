"""Decodeur FT8 / FT4 pour PyPhantom SDR (base : decodeur de l'auteur, structure corrigee).

Trame FT8 : 79 symboles de 0.16 s = 12.64 s, 8 tons espaces de 6.25 Hz.
Blocs Costas (7 symboles) en positions 0-6, 36-42, 72-78 ; donnees : 58 symboles
(positions 7-35 et 43-71) x 3 bits = 174 bits -> LDPC(174,91) -> 91 bits ->
CRC-14 -> 77 bits -> message (indicatif, grille, report).
"""
from collections import deque

import numpy as np

from .ft8_crc import check_crc14
from .ft8_ldpc import FT8LDPC
from .ft8_unpack import unpack_ft8

FT8_COSTAS = [3, 1, 4, 0, 6, 5, 2]
FT8_SYMBOLS = 79
FT8_SYMBOL_S = 0.16
FT8_TONE_HZ = 6.25
GRAY = [0, 1, 3, 2, 6, 7, 5, 4]

FT4_SYMBOLS = 105
FT4_SYMBOL_S = 0.048
FT4_TONE_HZ = 20.833
FT4_COSTAS = [0, 1, 3, 2, 5, 6, 4, 7]


class FT8Decoder:
    def __init__(self, fs, emit=None, mode="FT8", center=1500.0):
        self.fs = float(fs)
        self.emit = emit
        self.mode = mode
        self.center = float(center)
        self.buf = np.zeros(0, dtype=np.float32)
        self.messages = []
        self.samples_received = 0
        self.last_snr = -99
        self.activity = False
        self.history = deque(maxlen=200)
        self.ldpc = FT8LDPC()
        if mode == "FT4":
            self.nsym = FT4_SYMBOLS
            self.tsym = FT4_SYMBOL_S
            self.tone = FT4_TONE_HZ
            self.costas = FT4_COSTAS
        else:
            self.nsym = FT8_SYMBOLS
            self.tsym = FT8_SYMBOL_S
            self.tone = FT8_TONE_HZ
            self.costas = FT8_COSTAS
        self.period = self.nsym * self.tsym

    def process(self, audio):
        if audio is None or len(audio) == 0:
            return
        y = np.asarray(audio, dtype=np.float32)
        if len(y) > 10:
            y = y - np.mean(y)
        self.buf = np.concatenate([self.buf, y])
        self.samples_received += len(y)
        if len(y) > 0:
            lvl = 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-8)
            self.history.append(lvl)
        max_len = int(self.fs * (self.period + 5))
        if len(self.buf) > max_len:
            self.buf = self.buf[-max_len:]

    def try_decode(self):
        frame_len = int(self.fs * self.period)
        if len(self.buf) < frame_len:
            return None
        chunk = self.buf[-frame_len:]
        self.buf = self.buf[:-frame_len // 2]
        snr = self._estimate_snr(chunk)
        self.last_snr = snr
        self.activity = snr > -20
        if snr < -22:
            return None
        tones = self._detect_tones(chunk)
        if tones is None or len(tones) < 40:
            return None
        sync_pos = self._find_costas(tones)
        if sync_pos is None:
            return None
        bits = self._tones_to_bits(tones, sync_pos)
        msg = self._decode_bits(bits)
        if msg and msg not in self.messages[-6:]:
            self.messages.append(msg)
            if len(self.messages) > 40:
                self.messages = self.messages[-30:]
            if self.emit:
                self.emit(msg + "\n")
            return msg
        return None

    def _decode_bits(self, bits):
        if len(bits) >= 174:
            soft = np.array([1.0 if b == 0 else -1.0 for b in bits[:174]], dtype=np.float64)
            dec = self.ldpc.decode(soft)
            hard = [int(b) for b in dec[:91]] if dec is not None else list(bits[:174])[:91]
            if len(hard) >= 91 and check_crc14(hard[:91]):
                m = unpack_ft8(hard[:77])
                if m:
                    return f"{m}  {self.last_snr:+.0f}dB"
        if len(bits) >= 77 and check_crc14(bits[:91] if len(bits) >= 91 else bits[:77] + [0] * 14):
            m = unpack_ft8(bits[:77])
            if m:
                return f"{m}  {self.last_snr:+.0f}dB"
        return None

    def _estimate_snr(self, chunk):
        if len(chunk) < 200:
            return -99
        spec = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / self.fs)
        sig_mask = np.abs(freqs - self.center) < self.tone * 5
        noise_mask = (np.abs(freqs - self.center) > self.tone * 10) & (freqs > 300) & (freqs < 2800)
        if not np.any(sig_mask) or not np.any(noise_mask):
            return -99
        sig = np.mean(spec[sig_mask] ** 2)
        noise = np.mean(spec[noise_mask] ** 2) + 1e-12
        return 10 * np.log10(sig / noise + 1e-12)

    def _detect_tones(self, chunk):
        sym_len = max(int(self.fs * self.tsym), 8)
        nsym = min(self.nsym + 10, len(chunk) // sym_len)
        if nsym < 20:
            return None
        tones = []
        freqs = np.fft.rfftfreq(sym_len, 1.0 / self.fs)
        for i in range(nsym):
            seg = chunk[i * sym_len:(i + 1) * sym_len]
            if len(seg) < sym_len // 2:
                break
            spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
            best, best_e = 0, 0.0
            for t in range(8):
                f = self.center + t * self.tone
                if f < 100 or f > self.fs / 2 - 100:
                    continue
                idx = int(np.argmin(np.abs(freqs - f)))
                e = float(np.mean(spec[max(0, idx - 1):idx + 2]))
                if e > best_e:
                    best_e = e
                    best = t
            tones.append(best)
        return tones

    def _find_costas(self, tones):
        c = self.costas
        clen = len(c)
        best_pos, best_score = None, 0
        for i in range(len(tones) - clen):
            score = sum(1 for j in range(clen) if tones[i + j] == c[j])
            if score > best_score and score >= clen - 1:
                best_score = score
                best_pos = i
        return best_pos

    def _tones_to_bits(self, tones, sync_pos):
        data_idx = list(range(sync_pos + 7, sync_pos + 36)) + list(range(sync_pos + 43, sync_pos + 72))
        bits = []
        for i in data_idx:
            if i >= len(tones):
                break
            g = GRAY[tones[i] % 8]
            bits.append((g >> 2) & 1)
            bits.append((g >> 1) & 1)
            bits.append(g & 1)
        return bits

    def on_timer(self):
        self.try_decode()

    def status(self):
        secs = self.samples_received / self.fs if self.fs else 0
        act = "signal" if self.activity else "silence"
        n = len(self.messages)
        if n:
            return f"{self.mode}: {n} msg | SNR {self.last_snr:+.0f}dB | {act} | {secs:.0f}s"
        return f"{self.mode}: {act} | SNR {self.last_snr:+.0f}dB | {secs:.0f}s"


class FT4Decoder(FT8Decoder):
    def __init__(self, fs, emit=None, center=1500.0):
        super().__init__(fs, emit, mode="FT4", center=center)
