"""Décodeur RTTY avec filtre optimisé.

RTTY : Radio Teletype, Baudot 45.45 ou 50 bauds.
"""
import numpy as np


class RTTYDecoder:
    def __init__(self, fs=8000, baud=45.45, shift=170):
        self.fs = fs
        self.baud = baud
        self.shift = shift
        self.messages = []
        self.buf = np.zeros(0, dtype=np.float32)
        self.sps = int(fs / baud)
        self.freq_mark = 2125
        self.freq_space = 2125 - shift
        self.baudot = self._build_baudot()

    def _build_baudot(self):
        """Table Baudot ITA2."""
        letters = " E\nA SIU\rDRJNFCKTZLWHYPQOBG\nMXV"
        figures = " 3\n- '87\r$4\n,!:(5+)2#6019?&\n./;"
        return {'letters': letters, 'figures': figures}

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
        bits = self._demod_fsk()
        if len(bits) >= 10:
            text = self._decode_baudot(bits)
            if text and len(text) > 2 and text not in self.messages[-5:]:
                self.messages.append(text)
                msgs.append(text)
        return msgs

    def _demod_fsk(self):
        """Démodulation FSK avec filtre adapté."""
        bits = []
        n = self.sps
        t = np.arange(n) / self.fs
        # Filtres adaptés
        ref_mark = np.exp(2j * np.pi * self.freq_mark * t)
        ref_space = np.exp(2j * np.pi * self.freq_space * t)
        # Filtre passe-bas pour lisser
        alpha = 0.3
        mark_filt = 0
        space_filt = 0
        for i in range(0, len(self.buf) - n, n // 2):
            chunk = self.buf[i:i+n]
            if len(chunk) < n:
                break
            corr_m = np.abs(np.sum(chunk * ref_mark))
            corr_s = np.abs(np.sum(chunk * ref_space))
            mark_filt = alpha * corr_m + (1 - alpha) * mark_filt
            space_filt = alpha * corr_s + (1 - alpha) * space_filt
            bits.append(1 if mark_filt > space_filt else 0)
        return bits

    def _decode_baudot(self, bits):
        """Décodage Baudot."""
        chars = []
        mode = 'letters'
        for i in range(0, len(bits) - 7, 7):
            # Start bit + 5 data + stop
            if bits[i] != 0:
                continue
            val = 0
            for j in range(5):
                val |= bits[i+1+j] << j
            if val == 0x1B:  # FIGS
                mode = 'figures'
            elif val == 0x1F:  # LTRS
                mode = 'letters'
            elif val < len(self.baudot[mode]):
                c = self.baudot[mode][val]
                if c != '\\n':
                    chars.append(c)
        return "".join(chars).strip() or None
