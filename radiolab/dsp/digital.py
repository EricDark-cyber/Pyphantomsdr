"""Décodages numériques simples sur la FM étroite : horloge de bits, DCS, POCSAG.

Tous partent du signal en sortie du discriminateur FM (avant désaccentuation) :
un « 1 » et un « 0 » y sont simplement deux niveaux de tension.
"""
import collections
import time
import numpy as np
from .filters import StreamFIR, lowpass_taps, taps_for_transition


class BitSlicer:
    """Transforme un signal à deux niveaux en suite de bits, à un débit connu.

    Les changements de niveau tombent toujours aux frontières entre bits.  On
    repère donc l'instant des passages par zéro, on en déduit le rythme (leur
    position moyenne modulo la durée d'un bit), et on lit le signal au milieu de
    chaque bit.
    """

    def __init__(self, fs, baud):
        self.T = fs / float(baud)             # échantillons par bit (pas forcément entier)
        self.n = 0
        self.E = 0j
        self.prev = 0.0
        self.t_last = None
        self.dc = 0.0
        self.quality = 0.0

    def process(self, x):
        """Retourne les bits (booléens, True = niveau positif) lus dans ce bloc."""
        if len(x) == 0:
            return np.zeros(0, dtype=bool)
        self.dc += 0.05 * (float(np.mean(x)) - self.dc)
        buf = np.concatenate([[self.prev], x.astype(np.float64) - self.dc])   # buf[0] : indice absolu n - 1
        base = self.n - 1
        pos = buf > 0
        k = np.nonzero(pos[1:] != pos[:-1])[0]
        if len(k):
            t = base + k + buf[k] / (buf[k] - buf[k + 1])                     # passage par zéro interpolé
            e = np.sum(np.exp(-2j * np.pi * t / self.T))
            self.E = 0.9 * self.E + e
            self.quality = 0.9 * self.quality + 0.1 * abs(e) / len(k)         # 1 = passages bien alignés
        edge = -np.angle(self.E) * self.T / (2 * np.pi)                       # frontière de bit (modulo T)
        mid = edge + self.T / 2
        last = base + len(buf) - 1
        t_exp = base + self.T if self.t_last is None else self.t_last + self.T
        t0 = max(mid + np.round((t_exp - mid) / self.T) * self.T, base)
        count = int(np.floor((last - t0) / self.T)) + 1
        bits = np.zeros(0, dtype=bool)
        if count > 0:
            ts = t0 + self.T * np.arange(count)
            bits = pos[np.clip(np.round(ts - base).astype(np.int64), 0, len(buf) - 1)]
            self.t_last = float(ts[-1])
        self.prev = float(buf[-1])
        self.n += len(x)
        return bits


# ------------------------------------------------------------------------ DCS
DCS_CODES = [0o023, 0o025, 0o026, 0o031, 0o032, 0o043, 0o047, 0o051, 0o054, 0o065, 0o071, 0o072, 0o073, 0o074,
             0o114, 0o115, 0o116, 0o125, 0o131, 0o132, 0o134, 0o143, 0o152, 0o155, 0o156, 0o162, 0o165, 0o172,
             0o174, 0o205, 0o223, 0o226, 0o243, 0o244, 0o245, 0o251, 0o261, 0o263, 0o265, 0o271, 0o306, 0o311,
             0o315, 0o331, 0o343, 0o346, 0o351, 0o364, 0o365, 0o371, 0o411, 0o412, 0o413, 0o423, 0o431, 0o432,
             0o445, 0o464, 0o465, 0o466, 0o503, 0o506, 0o516, 0o532, 0o546, 0o565, 0o606, 0o612, 0o624, 0o627,
             0o631, 0o632, 0o654, 0o662, 0o664, 0o703, 0o712, 0o723, 0o731, 0o732, 0o734, 0o743, 0o754]


def dcs_codeword(code):
    """Mot de 23 bits d'un code DCS : 11 bits de parité (Golay 23,12), puis « 100 », puis le code."""
    data = 0x800 | (code & 0x1FF)
    cw = data
    for _ in range(12):
        if cw & 1:
            cw ^= 0xC75
        cw >>= 1
    return (cw << 12) | data


_DCS_WORDS = {dcs_codeword(c): c for c in DCS_CODES}
_DCS_BY_DATA = {0x800 | c: c for c in DCS_CODES}          # 12 bits de données (« 100 » + code) -> code


def _golay_parity(data12):
    """11 bits de parité Golay(23,12) d'un mot de 12 bits (même générateur 0xC75 que l'encodeur DCS)."""
    cw = data12 & 0xFFF
    for _ in range(12):
        if cw & 1:
            cw ^= 0xC75
        cw >>= 1
    return cw & 0x7FF


def golay_syndrome(word23):
    """Syndrome 11 bits : nul si le mot de 23 bits est un mot de code Golay valide."""
    return ((word23 >> 12) & 0x7FF) ^ _golay_parity(word23 & 0xFFF)


def _build_golay_table():
    """syndrome -> (motif d'erreur, poids). Le code Golay(23,12) étant parfait, les 2048
    syndromes correspondent exactement aux 2048 motifs d'erreur de poids <= 3."""
    import itertools
    table = {0: (0, 0)}
    for w in (1, 2, 3):
        for combo in itertools.combinations(range(23), w):
            e = 0
            for b in combo:
                e |= 1 << b
            table.setdefault(golay_syndrome(e), (e, w))
    return table


_GOLAY_TABLE = _build_golay_table()


def golay_decode(word23):
    """Corrige un mot Golay(23,12). Retourne (données 12 bits corrigées, nombre de bits corrigés)."""
    e, w = _GOLAY_TABLE.get(golay_syndrome(word23), (0, 0))
    return (word23 ^ e) & 0xFFF, w


class DCSDetector:
    """Reconnaît le code DCS (134,4 bit/s, mot Golay(23,12) répété sans fin), AVEC correction d'erreurs.

    Chaque mot de 23 bits est décodé par syndrome Golay (jusqu'à 3 bits faux corrigés, ici
    limité à 2 pour écarter le bruit) au lieu d'une simple recherche exacte : bien plus robuste
    sur signal faible. La confirmation demande trois lectures cohérentes du même code.
    """
    MAX_CORRECT = 2            # on n'accepte un mot que si <= 2 bits ont dû être corrigés
    CONFIRM = 3               # lectures cohérentes avant d'afficher

    def __init__(self, fs):
        self.lp = StreamFIR(lowpass_taps(250.0, fs, taps_for_transition(fs, 300.0)))
        self.slicer = BitSlicer(fs, 134.4)
        self.reg = 0
        self.code = None                      # (code, inversé) ou None
        self.hits = collections.deque(maxlen=20)
        self.since = 0
        self.nbits = 0
        self.first_seen = {}                    # verrou de phase du mot de 23 bits

    def process(self, audio):
        for bit in self.slicer.process(self.lp.process(audio.astype(np.float32, copy=False))):
            self.reg = ((self.reg >> 1) | (int(bit) << 22)) & 0x7FFFFF  # le premier bit reçu finit en position 0
            self.since += 1
            self.nbits += 1
            if self.nbits < 23:                   # avant un mot complet, la fenetre est trompeuse
                continue
            hit = None
            best = None
            for inverted in (False, True):          # on teste les deux polarités, on garde la meilleure
                word = self.reg ^ 0x7FFFFF if inverted else self.reg
                data, nerr = golay_decode(word)
                if nerr > self.MAX_CORRECT:
                    continue
                code = _DCS_BY_DATA.get(data)
                inv = inverted
                if code is None:
                    # le complémentaire d'un mot de code Golay est un mot de code : un signal inversé
                    # se lit aussi en polarité normale, avec les 12 bits de données complémentés
                    code = _DCS_BY_DATA.get(data ^ 0xFFF)
                    if code is not None:
                        inv = not inverted
                if code is not None and (best is None or nerr < best[0]):
                    best = (nerr, code, inv)
            if best is not None:
                hit = (best[1], best[2], best[0])
                cand = (hit[0], hit[1])
                if cand not in self.first_seen:
                    self.first_seen[cand] = self.nbits   # le vrai mot (phase 0) arrive en premier dans chaque période
                self.hits.append(hit)
                self.since = 0
                self._decide()
            if self.since > 23 * 6:                                 # plus rien de valable depuis une seconde
                self.code = None
                self.hits.clear()
                self.first_seen.clear()

    def _decide(self):
        # Un code DCS normal et son inverse peuvent paraitre identiques (rotation du mot de 23 bits) :
        # on departage au nombre de mots decodes SANS erreur (nerr == 0), puis au nombre total de
        # lectures, puis a la somme des erreurs (la vraie phase ne corrige jamais rien), puis on
        # prefere la polarite normale a egalite complete.
        counts = collections.Counter()
        zeros = collections.Counter()
        errs = collections.Counter()
        for code, inv, nerr in self.hits:
            h = (code, inv)
            counts[h] += 1
            errs[h] += nerr
            if nerr == 0:
                zeros[h] += 1

        def key(h):
            return (zeros[h], counts[h], -errs[h], -self.first_seen.get(h, 1 << 60), not h[1])

        if not counts:
            return
        best = max(counts, key=key)
        if counts[best] >= self.CONFIRM:
            self.code = best

    def label(self):
        if self.code is None:
            return None
        return "D%03o%s" % (self.code[0], "I" if self.code[1] else "N")


# --------------------------------------------------------------------- POCSAG
POCSAG_SYNC = 0x7CD215D8
POCSAG_IDLE = 0x7A89C197
_NUMERIC = "0123456789*U -)("


def pocsag_syndrome(cw):
    """0 si le mot de 32 bits est valide : code BCH(31,21) sur les 31 premiers bits + parité paire."""
    reg = cw >> 1
    for bit in range(30, 9, -1):
        if reg & (1 << bit):
            reg ^= 0x769 << (bit - 10)
    return (reg & 0x3FF) | ((bin(cw).count("1") & 1) << 10)


def pocsag_encode(data21):
    """Fabrique un mot de 32 bits à partir de 21 bits utiles (pour les essais)."""
    reg = data21 << 10
    for bit in range(30, 9, -1):
        if reg & (1 << bit):
            reg ^= 0x769 << (bit - 10)
    cw = ((data21 << 10) | (reg & 0x3FF)) << 1
    return cw | (bin(cw).count("1") & 1)


def pocsag_correct(cw):
    """Mot corrigé (au plus un bit faux), ou None si l'erreur est plus grave."""
    if pocsag_syndrome(cw) == 0:
        return cw
    for k in range(32):
        if pocsag_syndrome(cw ^ (1 << k)) == 0:
            return cw ^ (1 << k)
    return None


class PocsagDecoder:
    """Décode les messages de radiomessagerie POCSAG à un débit donné (512, 1200 ou 2400 bit/s)."""

    def __init__(self, fs, baud, on_message):
        self.baud, self.on_message = baud, on_message
        self.slicer = BitSlicer(fs, baud)
        self.reg = 0
        self.invert = 0
        self.state = "search"
        self.nbits = 0
        self.word = 0
        self.msg = None

    def process(self, audio):
        for bit in self.slicer.process(audio):
            # en POCSAG la fréquence haute (niveau positif) représente un 0
            self._push((0 if bit else 1) ^ self.invert)

    def _push(self, bit):
        self.reg = ((self.reg << 1) | bit) & 0xFFFFFFFF
        if self.state == "search":
            for inv in (0, 1):
                if bin(self.reg ^ POCSAG_SYNC ^ (0xFFFFFFFF * inv)).count("1") <= 2:
                    self.invert ^= inv
                    self.state, self.nbits, self.word = "batch", 0, 0
            return
        self.nbits += 1
        if self.nbits % 32:
            return
        if self.state == "sync":
            if bin(self.reg ^ POCSAG_SYNC).count("1") <= 2:
                self.state, self.nbits, self.word = "batch", 0, 0
            else:
                self._flush()
                self.state = "search"
            return
        self._codeword(self.reg, self.word // 2)
        self.word += 1
        if self.word == 16:
            self.state, self.nbits = "sync", 0

    def _codeword(self, raw, frame):
        cw = pocsag_correct(raw)
        if cw is None:
            if self.msg is not None:
                self.msg["errors"] += 1
                self.msg["chunks"].append(None)
            return
        if cw == POCSAG_IDLE:
            self._flush()
        elif cw >> 31 == 0:                                          # mot d'adresse : début d'un message
            self._flush()
            self.msg = {"ric": (((cw >> 13) & 0x3FFFF) << 3) | frame, "function": (cw >> 11) & 3,
                        "chunks": [], "errors": 0}
        elif self.msg is not None:                                   # mot de message : 20 bits utiles
            self.msg["chunks"].append((cw >> 11) & 0xFFFFF)

    def _flush(self):
        m, self.msg = self.msg, None
        if m is None:
            return
        bits = []
        for chunk in m["chunks"]:
            bits += [None] * 20 if chunk is None else [(chunk >> k) & 1 for k in range(19, -1, -1)]
        if m["function"] == 0 and bits:                              # numérique : 4 bits par chiffre
            text = "".join("?" if None in bits[i:i + 4] else
                           _NUMERIC[bits[i] | bits[i + 1] << 1 | bits[i + 2] << 2 | bits[i + 3] << 3]
                           for i in range(0, len(bits) - 3, 4)).rstrip()
        else:                                                        # alphanumérique : 7 bits par caractère
            chars = []
            for i in range(0, len(bits) - 6, 7):
                seg = bits[i:i + 7]
                v = None if None in seg else sum(b << k for k, b in enumerate(seg))
                chars.append("?" if v is None else (chr(v) if 32 <= v < 127 else ("" if v in (0, 3, 4, 23) else "·")))
            text = "".join(chars).rstrip()
        self.on_message({"time": time.strftime("%H:%M:%S"), "baud": self.baud, "ric": m["ric"],
                         "function": m["function"], "text": text, "errors": m["errors"]})


class Pocsag:
    """Écoute les trois débits POCSAG à la fois."""

    def __init__(self, fs, on_message):
        self.decoders = [PocsagDecoder(fs, baud, on_message) for baud in (512, 1200, 2400)]

    def process(self, audio):
        for d in self.decoders:
            d.process(audio)
