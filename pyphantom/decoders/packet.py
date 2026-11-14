"""Packet AX.25 (APRS 1200 bauds en FM, packet HF 300 bauds en BLU) : AFSK + HDLC complet.

HDLC : drapeau 0x7E, codage NRZI (un 0 change de tonalité), bourrage d'un 0 après cinq 1,
octets transmis bit de poids faible d'abord, FCS = CRC-16/X.25 (résidu 0xF0B8 sur la trame).
"""
import collections

import numpy as np

FLAG = [0, 1, 1, 1, 1, 1, 1, 0]      # 0x7E, transmis bit de poids faible d'abord


def crc16_x25(data):
    """CRC-16/X.25 (poly 0x1021 réfléchi 0x8408, init 0xFFFF, xorout 0xFFFF) — check 0x906E."""
    crc = 0xFFFF
    for b in data:
        crc ^= b & 0xFF
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc ^ 0xFFFF


def _crc16_raw(data):
    crc = 0xFFFF
    for b in data:
        crc ^= b & 0xFF
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc


FCS_RESIDUE = 0xF0B8                   # crc16_x25 sans xorout sur trame + FCS valide


def parse_ax25(frame):
    """Trame AX.25 (sans FCS) -> "SRC>DEST,PATH:info", ou None si malformée."""
    if frame is None or len(frame) < 16:
        return None
    addrs, i, saw_last = [], 0, False
    while i + 7 <= len(frame):
        a = frame[i:i + 7]
        call = bytes((b >> 1) & 0x7F for b in a[:6]).decode("ascii", "replace").strip()
        ssid = (a[6] >> 1) & 0x0F
        addrs.append(call + ("-%d" % ssid if ssid else ""))
        i += 7
        if a[6] & 1:
            saw_last = True
            break
    if not saw_last or len(addrs) < 2 or i + 2 > len(frame):
        return None
    dest, src = addrs[0], addrs[1]
    path = addrs[2:]
    if frame[i] != 0x03:               # trame UI (APRS)
        return None
    info = frame[i + 2:].decode("latin-1")
    return "%s>%s%s:%s" % (src, dest, ("," + ",".join(path)) if path else "", info)


class AFSKDecoder:
    """Démodulateur AFSK (Bell 202 ou 300 bauds) + trames HDLC AX.25.

    `emit` reçoit chaque trame décodée (une ligne de texte).
    """

    def __init__(self, fs, baud=1200.0, f_mark=1200.0, f_space=2200.0, emit=None):
        self.fs = float(fs)
        self.baud = float(baud)
        self.f_mark = float(f_mark)
        self.f_space = float(f_space)
        self.emit = emit
        self.sps = self.fs / self.baud
        self.buf = np.zeros(0, dtype=np.float32)
        self.phase = 0.0                # position (échantillons) du premier symbole
        self.synced = False
        self.bits = []
        self._parsed = 0                # bits déjà passés au décodage HDLC
        self._flags = []                # positions des drapeaux trouvés (persistantes)
        self._pair_idx = 0              # paires de drapeaux déjà traitées
        self.messages = []
        self.frames = 0                 # trames HDLC valides (FCS ok)
        self.history = collections.deque(maxlen=1500)   # niveaux en dB, pour l'affichage
        self.thr_db = None

    def process(self, audio):
        if audio is None or len(audio) == 0:
            return []
        x = np.asarray(audio, dtype=np.float32).flatten()
        self.history.extend((20 * np.log10(np.abs(x[::4]) + 1e-12)).tolist())
        self.buf = np.concatenate([self.buf, x])
        if len(self.buf) > self.fs * 120:
            drop = int(len(self.buf) - self.fs * 120)
            self.buf = self.buf[drop:]
            self._parsed = 0
        bits = self._demod(self.phase)
        if not self.synced:
            ph, run = self._search_phase()
            if run >= 6:
                self.phase, self.synced = ph, True
                bits = self._demod(self.phase)
                self._parsed = 0
                self._flags = []
                self._pair_idx = 0
        self.bits = bits
        self._parse()
        return []

    def _demod(self, phase):
        """Bits NRZI démodulés à partir de la phase d'horloge donnée (vectorisé)."""
        n = len(self.buf)
        m = max(int(np.ceil(self.sps)), 2)
        if n < m + 8:
            return []
        nbits = int((n - m - phase) // self.sps) + 1
        if nbits <= 0:
            return []
        i0 = np.round(phase + self.sps * np.arange(nbits)).astype(np.int64)
        i0 = np.minimum(i0, n - m)
        idx = i0[:, None] + np.arange(m)[None, :]
        seg = self.buf[idx].astype(np.complex64)
        t = np.arange(m) / self.fs
        cm = np.abs(seg @ np.exp(-2j * np.pi * self.f_mark * t))
        cs = np.abs(seg @ np.exp(-2j * np.pi * self.f_space * t))
        lvl = (cm > cs).astype(np.int8)
        bits = np.ones(nbits, dtype=np.int8)
        bits[1:] = (lvl[1:] == lvl[:-1]).astype(np.int8)   # NRZI : 0 = transition
        return bits.tolist()

    def _flag_run(self, bits):
        """Plus longue rafale de drapeaux espacés de 8 bits (préambule HDLC)."""
        b = np.asarray(bits, dtype=np.int8)
        if len(b) < 8:
            return 0
        f = np.asarray(FLAG, dtype=np.int8)
        match = (np.lib.stride_tricks.sliding_window_view(b, 8) == f).all(axis=1)
        pos = np.nonzero(match)[0]
        best = run = 0
        prev = -10
        for p in pos:
            run = run + 1 if p == prev + 8 else 1
            best = max(best, run)
            prev = p
        return best

    def _search_phase(self):
        best_ph, best_run = self.phase, 0
        for k in range(24):
            ph = k * self.sps / 24.0
            run = self._flag_run(self._demod(ph))
            if run > best_run:
                best_ph, best_run = ph, run
        return best_ph, best_run

    def _parse(self):
        bits = self.bits
        n = len(bits)
        if n < 16:
            return
        b = np.asarray(bits, dtype=np.int8)
        f = np.asarray(FLAG, dtype=np.int8)
        k = max(self._parsed - 8, 0)
        while k <= n - 8:
            if (b[k:k + 8] == f).all():
                if not self._flags or self._flags[-1] != k:
                    self._flags.append(k)
                k += 8
            else:
                k += 1
        self._parsed = n
        while self._pair_idx < len(self._flags) - 1:
            a, c = self._flags[self._pair_idx], self._flags[self._pair_idx + 1]
            self._pair_idx += 1
            if c - a <= 8 + 16:
                continue
            frame = self._destuff(bits[a + 8:c])
            if len(frame) < 18 or _crc16_raw(frame) != FCS_RESIDUE:
                continue
            text = parse_ax25(bytes(frame[:-2]))
            if text:
                self.frames += 1
                if text not in self.messages[-10:]:
                    self.messages.append(text)
                    if self.emit:
                        self.emit(text + "\n")

    @staticmethod
    def _destuff(bits):
        """Supprime le bourrage (0 après cinq 1) et assemble les octets, bit faible d'abord."""
        out, ones = [], 0
        for b in bits:
            if ones == 5:
                if b == 1:
                    return []           # six 1 : annulation de trame
                ones = 0
                continue
            out.append(b)
            ones = ones + 1 if b else 0
        by = bytearray()
        for i in range(0, len(out) - 7, 8):
            v = 0
            for k in range(8):
                v |= out[i + k] << k
            by.append(v)
        return by


PacketDecoder = AFSKDecoder             # ancien nom
