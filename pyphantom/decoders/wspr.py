"""Décodeur WSPR (Weak Signal Propagation Reporter).

Signal : 162 symboles 4-FSK de SYMBOL_S secondes, tonalités espacées de TONE_HZ
autour de la fréquence audio de la balise (1400-1600 Hz typiquement).  Le bit de
poids faible de chaque symbole est la séquence de synchronisation fixe SYNC ;
le bit de poids fort est un bit du message codé en convolutif (taux 1/2) puis
entrelacé (INTERLEAVE).  Les messages font 50 bits : indicatif (28), locator
Maidenhead 4 caractères (15), puissance dBm (7).

Les slots WSPR durent 120 s (minutes paires) ; à chaque changement de slot, le
contenu du slot écoulé est analysé (recherche de la synchro en temps/fréquence,
puis décodage de Viterbi et validation du message).
"""
import collections
import time

import numpy as np

SYMBOLS = 162
SYMBOL_S = 8192.0 / 12000.0            # 0,6827 s
TONE_HZ = 12000.0 / 8192.0             # 1,4648 Hz
SLOT_S = 120.0
FS_DIV = 33                            # le décodeur travaille à fs/33 (~378,8 Hz à 12,5 kHz)
F_CENTER = 1500.0                      # la bande des balises (1400-1600 Hz) est ramenée ici
F_SEARCH = (-100.0, 100.0)             # plage cherchée autour de F_CENTER, en relatif
SYNC_MIN = 120                         # seuil de corrélation avec SYNC (sur 162)

_CALL_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"   # base 36

# Séquence de synchronisation (162 bits, fixe) — le LSB de chaque symbole.
def _sync_seq():
    state = 0x1ACE
    out = np.zeros(SYMBOLS, dtype=np.int8)
    for i in range(SYMBOLS):
        bit = bin(state & 0xFFFF).count("1") & 1
        out[i] = bit
        state = ((state << 1) | bit) & 0xFFFF
    return out


SYNC = _sync_seq()

# Entrelacement des 162 bits codés (permutation fixe).
INTERLEAVE = np.array([(61 * i) % SYMBOLS for i in range(SYMBOLS)], dtype=np.int64)

_G1, _G2 = 0x6D, 0x4F                   # polynômes du code convolutif taux 1/2, K=7


# ------------------------------------------------------------------ message 50 bits
def _call_value(call):
    v = 0
    for c in call:
        v = v * 36 + _CALL_CHARS.index(c)
    return v


def _put(value, n):
    return [(value >> (n - 1 - k)) & 1 for k in range(n)]


def pack(call, loc, dbm):
    """(indicatif, locator 4 car., dBm) -> 50 bits (28 + 15 + 7)."""
    call = str(call).upper().strip()
    if not 1 <= len(call) <= 6 or any(c not in _CALL_CHARS for c in call):
        raise ValueError("indicatif invalide : %r" % call)
    v = _call_value(call)
    if v >= 1 << 28:
        raise ValueError("indicatif trop long pour 28 bits : %r" % call)
    loc = str(loc).upper().strip()
    if len(loc) != 4 or loc[0] not in "ABCDEFGHIJKLMNOPR" or loc[1] not in "ABCDEFGHIJKLMNOPR" \
            or not loc[2:].isdigit():
        raise ValueError("locator invalide : %r" % loc)
    g = ((ord(loc[0]) - 65) * 18 + (ord(loc[1]) - 65)) * 100 + int(loc[2]) * 10 + int(loc[3])
    return _put(v, 28) + _put(g, 15) + _put(int(dbm), 7)


def unpack(bits):
    """50 bits -> (indicatif, locator, dBm), ou None si le contenu est invalide."""
    b = np.asarray(bits).flatten()
    if len(b) != 50:
        return None
    v = 0
    for x in b[:28]:
        v = (v << 1) | int(x)
    if v <= 0:
        return None
    call = ""
    vv = v
    while vv:
        vv, r = divmod(vv, 36)
        call = _CALL_CHARS[r] + call
    if _call_value(call) != v:
        return None
    g = 0
    for x in b[28:43]:
        g = (g << 1) | int(x)
    if g >= 18 * 18 * 100:
        return None
    d2 = g % 10
    g //= 10
    d1 = g % 10
    g //= 10
    l2 = g % 18
    l1 = g // 18
    loc = chr(65 + l1) + chr(65 + l2) + str(d1) + str(d2)
    p = 0
    for x in b[43:50]:
        p = (p << 1) | int(x)
    if p > 99:
        return None
    return call, loc, p


# ------------------------------------------------------------------ code convolutif
def convolve(bits):
    """Bits -> 2 bits codés par bit d'entrée (taux 1/2, K=7, G1=0x6D, G2=0x4F)."""
    reg = 0
    out = []
    for b in np.asarray(bits).flatten():
        reg = ((reg << 1) | int(b)) & 0x7F
        out.append(bin(reg & _G1).count("1") & 1)
        out.append(bin(reg & _G2).count("1") & 1)
    return out


def _viterbi(soft, n_out):
    soft = np.asarray(soft, dtype=np.float64).flatten()
    n_in = len(soft) // 2
    if n_in < 1:
        return []
    INF = 1e18
    pm = np.full(128, INF)
    pm[0] = 0.0
    decisions = np.zeros((n_in, 128), dtype=np.int16)
    for t in range(n_in):
        r1, r2 = soft[2 * t], soft[2 * t + 1]
        npm = np.full(128, INF)
        dec = np.zeros(128, dtype=np.int16)
        for state in range(128):
            base = pm[state]
            if base >= INF:
                continue
            for b in (0, 1):
                reg = ((state << 1) | b) & 0x7F
                e1 = 1.0 if (bin(reg & _G1).count("1") & 1) else -1.0
                e2 = 1.0 if (bin(reg & _G2).count("1") & 1) else -1.0
                cost = base - (r1 * e1 + r2 * e2)
                if cost < npm[reg]:
                    npm[reg] = cost
                    dec[reg] = state
        pm = npm
        decisions[t] = dec
    end = int(np.argmin(pm))
    s = end
    bits = []
    for t in range(n_in - 1, -1, -1):
        bits.append(s & 1)
        s = int(decisions[t][s])
    bits.reverse()
    return bits[:n_out], -pm[end]          # bits + métrique de confiance (corrélation du chemin)


MIN_PATH_METRIC = 200.0                   # en dessous : ce n'est pas un signal codé (hasard)


def sequential_decode(soft_bits):
    """Valeurs molles (±) des 162 bits codés -> 50 bits du message, ou None si invalide."""
    soft = np.asarray(soft_bits, dtype=np.float64).flatten()
    if len(soft) < SYMBOLS:
        return None
    soft = soft[:SYMBOLS]
    peak = float(np.max(np.abs(soft)))
    if peak > 0:
        soft = soft * (3.0 / peak)             # normalise à ±3 comme un signal propre
    bits, metric = _viterbi(soft, 50)
    if len(bits) < 50 or metric < MIN_PATH_METRIC:
        return None
    if unpack(bits) is None:
        return None
    return [int(b) for b in bits]


def encode(call, loc, dbm):
    """Message -> 162 symboles 4-FSK (0..3), LSB = SYNC."""
    bits = pack(call, loc, dbm)
    coded = np.array(convolve(bits + [0] * 31), dtype=np.int8)      # 162 bits
    inter = coded[INTERLEAVE]
    return (2 * inter + SYNC).astype(np.int8)


# ------------------------------------------------------------------ décodeur de slots
def _analytic(x):
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[1:n // 2] = 2.0
        h[n // 2] = 1.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(X * h)


class WSPRDecoder:
    """Décode les balises WSPR slot par slot (120 s).

    `callback` reçoit chaque ligne décodée ; `clock()` donne l'heure courante en
    secondes (horloge externe pour les tests).  `process` retourne la liste des
    nouvelles lignes.
    """

    def __init__(self, fs, callback=None, clock=None):
        self.fs = float(fs)
        self.callback = callback
        self.clock = clock or time.time
        self.threaded = True
        self.fs_d = self.fs / FS_DIV
        self.sps_d = SYMBOL_S * self.fs_d
        self._raw = np.zeros(0, dtype=np.complex64)
        self._t_abs = 0.0                  # heure absolue du premier échantillon reçu
        self._n_in = 0                     # échantillons d'entrée consommés
        self._slot = None
        self._slot_audio = []
        self.decodes = 0
        self.messages = []
        self.history = collections.deque(maxlen=1500)
        self.thr_db = None

    def process(self, audio):
        out = []
        if audio is not None and len(audio):
            x = np.asarray(audio, dtype=np.float32).flatten()
            self.history.extend((20 * np.log10(np.abs(x[::8]) + 1e-12)).tolist())
            now = float(self.clock())
            t0 = now - len(x) / self.fs
            # mélange la bande des balises (autour de F_CENTER) en bande de base, puis décime
            tt = (self._t_abs + self._n_in + np.arange(len(x))) / self.fs
            bb = x.astype(np.complex64) * np.exp(-2j * np.pi * F_CENTER * tt)
            self._n_in += len(x)
            buf = np.concatenate([self._raw, bb])
            nd = len(buf) // FS_DIV
            dec = buf[:nd * FS_DIV].reshape(nd, FS_DIV).mean(axis=1).astype(np.complex64)
            self._raw = buf[nd * FS_DIV:]
            t_dec = t0 + (np.arange(nd) + 0.5) * FS_DIV / self.fs
            for j in range(nd):
                s = int(t_dec[j] // SLOT_S)
                if self._slot is None:
                    self._slot = s
                elif s != self._slot:
                    self._finish_slot(out)
                    self._slot = s
                self._slot_audio.append(dec[j])
            if self._slot is not None and int(now // SLOT_S) > self._slot:
                self._finish_slot(out)
                self._slot = int(now // SLOT_S)
                self._slot_audio = []
        return out

    def _finish_slot(self, out):
        audio = np.asarray(self._slot_audio, dtype=np.complex64)
        self._slot_audio = []
        slot = self._slot
        if slot is None or len(audio) < int(SYMBOLS * SYMBOL_S * self.fs_d):
            return
        for f0, call, loc, dbm in self._decode_slot(slot, audio):
            self.decodes += 1
            hhmm = time.strftime("%H:%M", time.gmtime(slot * SLOT_S))
            line = "%s  %.1f Hz  %-6s %s %d dBm" % (hhmm, f0 + F_CENTER, call, loc, dbm)
            self.messages.append(line)
            if self.callback:
                self.callback(line + "\n")
            out.append(line)

    def _decode_slot(self, slot, x):
        """Cherche les balises du slot : synchro (temps, fréquence) puis Viterbi."""
        found = []
        z = x                            # déjà en bande de base complexe (mélangé à -F_CENTER)
        n = len(x)
        W = max(int(round(self.sps_d)), 8)
        t_max = SLOT_S - SYMBOLS * SYMBOL_S
        if t_max < 0 or n < int(SYMBOLS * SYMBOL_S * self.fs_d) + W:
            return found
        t = np.arange(W) / self.fs_d
        tone_off = (np.arange(4) - 1.5) * TONE_HZ
        # recherche grossière : t0 pas de SYMBOL_S/4, f0 pas de 2 Hz
        cands = []
        f_grid = np.arange(F_SEARCH[0], F_SEARCH[1], 2.0)
        all_f = (f_grid[:, None] + tone_off[None, :]).ravel()          # (F*4,)
        E = np.exp(-2j * np.pi * all_f[:, None] * t[None, :])          # (F*4, W)
        for t0 in np.arange(0.0, t_max + 1e-9, SYMBOL_S / 4):
            starts = np.round((t0 + np.arange(SYMBOLS) * SYMBOL_S) * self.fs_d).astype(np.int64)
            if starts[-1] + W > n:
                break
            idx = starts[:, None] + np.arange(W)[None, :]
            Wm = z[idx]                                                 # (162, W)
            mags = np.abs(Wm @ E.T)                                     # (162, F*4)
            mags = mags.reshape(SYMBOLS, len(f_grid), 4)
            s = mags.argmax(axis=2)                                     # (162, F)
            score = ((s & 1) == SYNC[:, None]).sum(axis=0)              # (F,)
            j = int(np.argmax(score))
            if score[j] >= SYNC_MIN - 30:
                cands.append((int(score[j]), float(t0), float(f_grid[j])))
        cands.sort(reverse=True)
        placed = []                       # (t0, f0) des balises déjà décodées
        for _, t0, f0 in cands[:8]:
            if any(abs(t0 - ft) < 0.7 and abs(f0 - ff) < 4.0 for ft, ff in placed):
                continue
            r = self._try_decode(x, z, t0, f0, n, W)
            if r is not None:
                ff, call, loc, dbm, tt = r
                found.append((ff, call, loc, dbm))
                placed.append((tt, ff))
                if len(found) >= 5:
                    break
        return found

    def _try_decode(self, x, z, t0, f0, n, W):
        """Affine (t0, f0) autour du candidat, puis décode le message. Retourne
        (f0, call, loc, dbm, t0) ou None."""
        best = None
        for dt in np.arange(-SYMBOL_S / 4, SYMBOL_S / 4 + 1e-9, SYMBOL_S / 16):
            for df in np.arange(-3.0, 3.0 + 1e-9, 0.5):
                tt, ff = t0 + dt, f0 + df
                if tt < 0 or tt > SLOT_S - SYMBOLS * SYMBOL_S:
                    continue
                starts = np.round((tt + np.arange(SYMBOLS) * SYMBOL_S) * self.fs_d).astype(np.int64)
                if starts[-1] + W > n:
                    continue
                idx = starts[:, None] + np.arange(W)[None, :]
                Wm = z[idx]
                E = np.exp(-2j * np.pi * (ff + (np.arange(4) - 1.5) * TONE_HZ)[:, None]
                           * (np.arange(W) / self.fs_d)[None, :])
                mags = np.abs(Wm @ E.T)                                 # (162, 4)
                s = mags.argmax(axis=1)
                score = int(((s & 1) == SYNC).sum())
                if best is None or score > best[0]:
                    best = (score, tt, ff, s, mags)
        if best is None or best[0] < SYNC_MIN:
            return None
        score, tt, ff, s, mags = best
        # bits mous : marge entre la meilleure tonalité et la seconde, signée par le bit codé
        order = np.argsort(mags, axis=1)
        top = mags[np.arange(SYMBOLS), s]
        second = mags[np.arange(SYMBOLS), order[:, -2]]
        margin = 3.0 * (top - second) / (top + second + 1e-9)
        coded_soft = margin * (2 * (s >> 1) - 1)
        inter = np.empty(SYMBOLS, dtype=np.float64)
        inter[:] = coded_soft
        coded = np.empty(SYMBOLS, dtype=np.float64)
        coded[INTERLEAVE] = inter
        bits = sequential_decode(coded)
        if bits is None:
            return None
        info = unpack(bits)
        if info is None:
            return None
        call, loc, dbm = info
        return ff, call, loc, dbm, tt

    def status(self):
        return "WSPR : %d balises décodées" % self.decodes


WSPRDecoderAlias = WSPRDecoder
