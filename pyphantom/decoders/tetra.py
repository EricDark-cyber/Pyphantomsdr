"""Décodeur TETRA (EN 300 392) : démodulation pi/4-DQPSK, repérage des salves,
lecture du bloc de synchronisation (BSCH) avec Viterbi et CRC.

Une salve fait BURST_BITS bits (255 symboles de 2 bits).  La séquence
d'apprentissage normale (22 bits) commence au bit NORMAL_TRAIN_START de la
salve, celle de synchronisation (38 bits) au bit SYNC_TRAIN_START ; le bloc
BSCH (120 bits) occupe les bits 94..214 des salves de synchronisation.
"""
import collections

import numpy as np

BAUD = 18000.0
BURST_BITS = 510
NORMAL_TRAIN_START = 244
SYNC_TRAIN_START = 244
BSCH_START = 94
BSCH_LEN = 120

_STEPS = np.array([np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4])   # sauts pi/4-DQPSK
_BITS = ((0, 0), (0, 1), (1, 1), (1, 0))                                      # 00->+45, 01->+135, 11->-135, 10->-45


def _seq(n, seed):
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.int8)


TRAIN_NORMAL_1 = _seq(22, 11)
TRAIN_NORMAL_2 = _seq(22, 21)
TRAIN_SYNC = _seq(38, 31)


def _lfsr(n, state=0x1ACE):
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        bit = bin(state & 0xFFFF).count("1") & 1
        out[i] = bit
        state = ((state << 1) | bit) & 0xFFFF
    return out


BSCH_SCRAMBLING = _lfsr(BSCH_LEN)
BSCH_ORDER = np.array([(i * 67) % BSCH_LEN for i in range(BSCH_LEN)], dtype=np.int64)   # permutation (67 premier avec 120)


def rrc_taps(sps, alpha=0.35, span=8):
    """Filtre root-raised-cosine (réponse impulsionnelle), `sps` échantillons par symbole."""
    sps = float(sps)
    n = max(int(span * sps), 3)
    t = (np.arange(n) - (n - 1) / 2.0) / sps          # en symboles
    h = np.zeros(n)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = 1 - alpha + 4 * alpha / np.pi
        elif abs(abs(ti) - 1.0 / (4 * alpha)) < 1e-9:
            h[i] = alpha / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                                         + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))) \
                / (np.pi * ti * (1 - (4 * alpha * ti) ** 2))
    return h


def bits_to_phase_steps(bits):
    """Bits -> sauts de phase pi/4-DQPSK (2 bits par symbole, voir _BITS)."""
    b = np.asarray(bits).astype(np.int8).flatten()
    n = len(b) // 2
    b0, b1 = b[:2 * n:2], b[1:2 * n:2]
    step = np.where((b0 == 0) & (b1 == 0), np.pi / 4,
           np.where((b0 == 0) & (b1 == 1), 3 * np.pi / 4,
           np.where((b0 == 1) & (b1 == 1), -3 * np.pi / 4, -np.pi / 4)))
    return step.astype(np.float64)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _d_to_bits(d):
    """Sauts de phase -> bits (vectorisé)."""
    d = np.asarray(d, dtype=np.float64)
    w = _wrap(d[:, None] - _STEPS[None, :])
    k = np.argmin(np.abs(w), axis=1)
    out = np.zeros((len(d), 2), dtype=np.int8)
    for j, bm in enumerate(_BITS):
        sel = k == j
        out[sel, 0] = bm[0]
        out[sel, 1] = bm[1]
    return out.reshape(-1)


def _crc16_raw(data):
    crc = 0xFFFF
    for b in data:
        x = ((crc >> 15) & 1) ^ int(b)
        crc = (crc << 1) & 0xFFFF
        if x:
            crc ^= 0x1021
    return crc


BSCH_CRC_RESIDUE = 0x1D0F        # crc16 (init 0xFFFF, poly 0x1021) sur info+CRC complémenté


def crc16_ok(bits):
    """Vrai si les 76 premiers bits (60 info + 16 CRC) forment un bloc BSCH valide."""
    b = np.asarray(bits).flatten()
    return len(b) >= 76 and _crc16_raw(b[:76]) == BSCH_CRC_RESIDUE


def _get(bits, a, n):
    v = 0
    for k in range(n):
        v = (v << 1) | int(bits[a + k])
    return v


def _encode_bsc(bits):
    """Encodeur convolutif BSCH (G1 = 1+D+D^4, G2 = 1+D^2+D^3+D^4, G2 seulement aux temps pairs)."""
    coded, hist = [], [0, 0, 0, 0]
    for t, b in enumerate(bits):
        b = int(b)
        coded.append(b ^ hist[0] ^ hist[3])
        if t % 2 == 0:
            coded.append(b ^ hist[1] ^ hist[2] ^ hist[3])
        hist = [b] + hist[:3]
    return np.array(coded, dtype=np.int8)


def _viterbi(coded, n_in):
    """Viterbi 16 états (K=5) sur le flux codé ; retourne (bits décodés, distance)."""
    g1_idx, g2_idx = [], []
    pos = 0
    for t in range(n_in):
        g1_idx.append(pos); pos += 1
        if t % 2 == 0:
            g2_idx.append(pos); pos += 1
        else:
            g2_idx.append(-1)
    INF = 1e9
    pm = [INF] * 16
    pm[0] = 0.0
    decisions = []
    for t in range(n_in):
        npm = [INF] * 16
        dec = [0] * 16
        i1 = g1_idx[t]
        i2 = g2_idx[t]
        for state in range(16):
            h0 = state & 1
            h1 = (state >> 1) & 1
            h2 = (state >> 2) & 1
            h3 = (state >> 3) & 1
            base = pm[state]
            if base >= INF:
                continue
            for b in (0, 1):
                g1 = b ^ h0 ^ h3
                cost = base + (int(coded[i1]) != g1)
                if i2 >= 0:
                    cost += (int(coded[i2]) != (b ^ h1 ^ h2 ^ h3))
                ns = ((state << 1) | b) & 0xF
                if cost < npm[ns]:
                    npm[ns] = cost
                    dec[ns] = state
        pm = npm
        decisions.append(dec)
    end = int(np.argmin(pm))
    bits = []
    s = end
    for t in range(n_in - 1, -1, -1):
        bits.append(s & 1)
        s = decisions[t][s]
    bits.reverse()
    return np.array(bits, dtype=np.int8), int(round(pm[end]))


def decode_bsch(block):
    """Décode un bloc BSCH de 120 bits -> dict (mcc, mnc, colour, timeslot, frame, multiframe, corrected)
    ou None si le CRC échoue."""
    b = np.asarray(block).astype(np.int8).flatten()
    if len(b) != BSCH_LEN:
        return None
    coded = (b ^ BSCH_SCRAMBLING)[BSCH_ORDER]
    n_in = 80                                   # 60 info + 16 CRC + 4 tail
    dec, dist = _viterbi(coded, n_in)
    if not crc16_ok(dec[:76]):
        return None
    return {
        "colour": _get(dec, 4, 6),
        "timeslot": _get(dec, 10, 2) + 1,
        "frame": _get(dec, 12, 5),
        "multiframe": _get(dec, 17, 6),
        "mcc": _get(dec, 31, 10),
        "mnc": _get(dec, 41, 14),
        "corrected": dist,
    }


class TetraDecoder:
    """Démodulateur pi/4-DQPSK + repérage des salves TETRA sur le flux I/Q décimé (fs1).

    `callback` reçoit les lignes de statut (porteuse, cellule identifiée).
    `self._bits` est appelé avec chaque nouveau paquet de bits démodulés.
    """

    def __init__(self, fs, callback=None):
        self.fs = float(fs)
        self.callback = callback
        self.baud = BAUD
        self.sps = self.fs / self.baud
        taps = rrc_taps(self.sps)
        self._taps = taps / np.sqrt(np.sum(taps ** 2))
        self._hist = np.zeros(max(len(self._taps) - 1, 1), dtype=np.complex64)
        self.y = np.zeros(0, dtype=np.complex64)
        self._sym_pos = None
        self._prev = None
        self._res = []
        self._p4 = []
        self._all_bits = np.zeros(0, dtype=np.int8)
        self._bits = self._on_bits
        self._next_train = None
        self.locked = False
        self.slot_phase = None
        self.counts = {"normale 1": 0, "normale 2": 0, "synchro": 0}
        self.quality = 0.0
        self.freq_error_hz = 0.0
        self.bsch_ok = 0
        self.bsch_bad = 0
        self.cell = {}
        self.messages = []
        self._lock_sent = False
        self._cell_sent = None
        self.history = collections.deque(maxlen=1500)   # niveaux en dB, pour l'affichage
        self.thr_db = None

    # ------------------------------------------------------------- réception
    def _on_bits(self, b):
        b = np.asarray(b, dtype=np.int8)
        self._all_bits = np.concatenate([self._all_bits, b]) if len(self._all_bits) else b
        if len(self._all_bits) > 600000:
            drop = len(self._all_bits) - 600000
            self._all_bits = self._all_bits[drop:]
            if self._next_train is not None:
                self._next_train -= drop

    def process(self, z):
        if z is None or len(z) == 0:
            return []
        z = np.asarray(z, dtype=np.complex64).flatten()
        self.history.extend((20 * np.log10(np.abs(z[::8]) + 1e-12)).tolist())
        buf = np.concatenate([self._hist, z])
        y = np.convolve(buf, self._taps)[len(self._taps) - 1:len(buf)]   # len(z) échantillons
        self._hist = buf[-(len(self._taps) - 1):]
        self.y = np.concatenate([self.y, y])
        if len(self.y) > self.fs * 30:
            drop = int(len(self.y) - self.fs * 30)
            self.y = self.y[drop:]
            if self._sym_pos is not None:
                self._sym_pos -= drop
        self._demod()
        self._scan_bursts()
        return []

    # ------------------------------------------------------------ démodulation
    def _demod(self):
        n = len(self.y)
        if self._sym_pos is None:
            phi = self._search_timing()
            if phi is None:
                return
            self._sym_pos = phi
            self._prev = None
        new = []
        while self._sym_pos + self.sps <= n:
            i = min(int(round(self._sym_pos)), n - 1)
            s = self.y[i]
            if self._prev is not None:
                dc = s * np.conj(self._prev)
                d = float(np.angle(dc))
                k = int(np.argmin(np.abs(_wrap(d - _STEPS))))
                self._res.append(float(_wrap(d - _STEPS[k])))
                new.append(_BITS[k][0])
                new.append(_BITS[k][1])
                self._p4.append(dc ** 4)          # (s_k conj s_{k-1})^4 = e^{j(pi + 4 delta)}
            self._prev = s
            self._sym_pos += self.sps
        if new:
            self._bits(np.array(new, dtype=np.int8))
        if len(self._res) > 200:
            delta = 0.0
            if len(self._p4) > 200:
                p4 = np.mean(np.asarray(self._p4[-20000:], dtype=np.complex128))
                delta = _wrap(float(np.angle(p4)) - np.pi) / 4.0
                self.freq_error_hz = float(delta / (2 * np.pi) * self.baud)
            r = np.asarray(self._res[-20000:])
            self.quality = float(np.clip(1 - np.mean(np.abs(_wrap(r - delta))) / (np.pi / 4), 0, 1))
        if len(self._res) > 100000:
            self._res = self._res[-50000:]
        if len(self._p4) > 100000:
            self._p4 = self._p4[-50000:]

    def _search_timing(self):
        n = len(self.y)
        nsym = int((n - self.sps) // self.sps)
        if nsym < 700:
            return None
        nsym = min(nsym, 4000)

        def metric(phi):
            """Erreur moyenne de phase après décision : minimale à l'instant symbole."""
            idxs = np.minimum(np.round(phi + self.sps * np.arange(nsym)).astype(np.int64), n - 1)
            s = self.y[idxs]
            d = np.angle(s[1:] * np.conj(s[:-1]))
            kk = np.argmin(np.abs(_wrap(d[:, None] - _STEPS[None, :])), axis=1)
            return float(np.mean(np.abs(_wrap(d - _STEPS[kk]))))

        best_phi, best_val = 0.0, 1e9
        for k in range(48):
            phi = k * self.sps / 48.0
            v = metric(phi)
            if v < best_val:
                best_val, best_phi = v, phi
        for k in range(-6, 7):                   # affinement local
            phi = (best_phi + k * self.sps / 96.0) % self.sps
            v = metric(phi)
            if v < best_val:
                best_val, best_phi = v, phi
        idxs = np.minimum(np.round(best_phi + self.sps * np.arange(nsym)).astype(np.int64), n - 1)
        s = self.y[idxs]
        if self._train_score(_d_to_bits(np.angle(s[1:] * np.conj(s[:-1])))) < 5:
            return None
        return best_phi

    @staticmethod
    def _train_score(bits):
        score = 0
        for tr in (TRAIN_NORMAL_1, TRAIN_NORMAL_2, TRAIN_SYNC):
            L = len(tr)
            if len(bits) < L:
                continue
            w = np.lib.stride_tricks.sliding_window_view(bits, L)
            score += int(((w == tr).sum(axis=1) >= L - 2).sum())
        return score

    # ------------------------------------------------------------- salves
    def _scan_bursts(self):
        b = self._all_bits
        n = len(b)
        if n < 300:
            return
        if self._next_train is None:
            found = None
            for tr in (TRAIN_SYNC, TRAIN_NORMAL_1, TRAIN_NORMAL_2):
                L = len(tr)
                if n < L:
                    continue
                w = np.lib.stride_tricks.sliding_window_view(b, L)
                m = (w == tr).sum(axis=1)
                cand = np.nonzero(m >= L - 1)[0]
                if len(cand):
                    found = int(cand[0])
                    break
            if found is None:
                return
            self._next_train = found
            self.slot_phase = found % BURST_BITS
        while self._next_train is not None and self._next_train + 38 <= n:
            p = self._next_train
            seg38 = b[p:p + 38]
            msync = int((seg38 == TRAIN_SYNC).sum())
            if msync >= 36:
                kind = "synchro"
            else:
                seg22 = b[p:p + 22]
                m1 = int((seg22 == TRAIN_NORMAL_1).sum())
                m2 = int((seg22 == TRAIN_NORMAL_2).sum())
                if max(m1, m2) >= 21:
                    kind = "normale 1" if m1 >= m2 else "normale 2"
                else:
                    kind = None
            if kind:
                self.counts[kind] += 1
                if not self.locked and sum(self.counts.values()) >= 3:
                    self.locked = True
                    self._emit("Porteuse TETRA verrouillée (%d salves)" % sum(self.counts.values()))
                if kind == "synchro":
                    bs = p - SYNC_TRAIN_START + BSCH_START
                    if bs >= 0 and bs + BSCH_LEN <= n:
                        info = decode_bsch(b[bs:bs + BSCH_LEN])
                        if info is None:
                            self.bsch_bad += 1
                        else:
                            self.bsch_ok += 1
                            key = (info["mcc"], info["mnc"], info["colour"])
                            if key != self._cell_sent:
                                self._cell_sent = key
                                self.cell = info
                                self._emit("Cellule TETRA identifiée : pays %d, réseau %d, code couleur %d"
                                           % (info["mcc"], info["mnc"], info["colour"]))
            self._next_train = p + BURST_BITS

    def _emit(self, text):
        line = text + "\n"
        if line not in self.messages[-10:]:
            self.messages.append(line)
            if self.callback:
                self.callback(line)

    # ---------------------------------------------------------------- état
    def status(self):
        if self.locked:
            c = self.cell
            if c:
                cell = "cellule pays %d, réseau %d, code couleur %d, " % (c.get("mcc", 0), c.get("mnc", 0), c.get("colour", 0))
            else:
                cell = "cellule inconnue, "
            return ("Porteuse TETRA verrouillée : %s%d salves (trame %d bits), qualité %.0f%%, "
                    "erreur de fréquence %.0f Hz" % (cell, sum(self.counts.values()), BURST_BITS,
                                                      100 * self.quality, self.freq_error_hz))
        return "Pas de porteuse TETRA"

    def burst_rates(self):
        sec = max(len(self.y) / self.fs, 1e-9)
        return {k: v / sec for k, v in self.counts.items()}


TETRADecoder = TetraDecoder
