"""Décodeur NAVTEX avec synchronisation SITOR-B (CCIR 476, code 3/4).

NAVTEX : messages maritimes sur 518/490 kHz.  SITOR-B : 100 bauds, deux tonalités
à 170 Hz d'écart autour de la fréquence audio `center` (1700 Hz par défaut, soit
1615/1785 Hz) ; chaque caractère de 7 bits (poids 3 ou 4) est émis DEUX FOIS et le
récepteur fait la majorité (FEC) entre les deux exemplaires.

La polarité ne se déduit pas du poids 3/4 (le complément d'un mot valide est valide),
mais du contenu : le texte NAVTEX est en lettres/chiffres/espace, donc la polarité
qui décode le plus de caractères "communs" est la bonne.

`encode(text)` produit la suite de mots de 7 bits (chacun deux fois) — signaux de test.
"""
import numpy as np

BAUD = 100
SHIFT = 170.0


def _build_code_table():
    """Caractère -> mot de 7 bits de poids 3 ou 4 (règle CCIR 476).

    C(7,3)+C(7,4) = 70 mots possibles : on couvre l'alphabet utile dans un ordre fixe.
    """
    chars = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,-/\r\n:;()?'\"!@#$%&*+=<>[]{}|\\^~`"
    words = [w for w in range(1 << 7) if bin(w).count("1") in (3, 4)]
    return {ch: w for ch, w in zip(chars, words)}


CODE_TABLE = _build_code_table()
INV_CODE = {v: k for k, v in CODE_TABLE.items()}

POW7 = (1 << np.arange(6, -1, -1, dtype=np.int64))
_WT = np.array([bin(i).count("1") in (3, 4) for i in range(128)], dtype=bool)
_INVT = np.zeros(128, dtype=np.int32)
_COMMON_CHARS = frozenset(" ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\r\n")
_COMMON_W = np.zeros(128, dtype=bool)
for _ch, _w in CODE_TABLE.items():
    _INVT[_w] = ord(_ch)
    _COMMON_W[_w] = _ch in _COMMON_CHARS
del _ch, _w


def encode(text):
    """Texte -> liste de mots de 7 bits, chacun émis deux fois (SITOR-B)."""
    out = []
    for ch in text.upper():
        w = CODE_TABLE.get(ch, CODE_TABLE["?"])
        out += [w, w]
    return out


def _weight_ok(v):
    return bin(v).count("1") in (3, 4)


def _words_of(bits, off):
    """Liste de bits -> mots de 7 bits (MSB en tête) à partir de la position off."""
    bb = np.asarray(bits, dtype=np.uint8)[off:]
    nw = len(bb) // 7
    if nw == 0:
        return np.zeros(0, dtype=np.int64)
    return bb[:nw * 7].reshape(nw, 7) @ POW7


def _combine_pairs(w):
    """Mots -> (mot_fusionne, valide, accord) par paire SITOR-B (majorité bit à bit)."""
    np_ = len(w) // 2
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool), np.zeros(0, dtype=bool))
    if np_ == 0:
        return empty
    p = w[:np_ * 2].reshape(np_, 2)
    v1 = _WT[p[:, 0]]
    v2 = _WT[p[:, 1]]
    b1 = (p[:, 0][:, None] >> np.arange(6, -1, -1)) & 1
    b2 = (p[:, 1][:, None] >> np.arange(6, -1, -1)) & 1
    wmaj = ((b1 + b2) >= 2) @ POW7
    both = v1 & v2
    eq = p[:, 0] == p[:, 1]
    wm = np.where(both, np.where(eq, p[:, 0], np.where(_WT[wmaj], wmaj, p[:, 0])),
                  np.where(v1, p[:, 0], p[:, 1]))
    agree = both & (eq | _WT[wmaj])
    return wm, v1 | v2, agree


class NAVTEXDecoder:
    def __init__(self, fs=8000, center=None, emit=None):
        self.fs = float(fs)
        self.center = float(center) if center else 1700.0
        self.emit = emit
        self.baud = BAUD
        self.sps = max(int(round(self.fs / self.baud)), 1)
        self.f_mark = self.center + SHIFT / 2.0      # bit 1 = mark (tonalité haute)
        self.f_space = self.center - SHIFT / 2.0
        self.buf = np.zeros(0, dtype=np.float32)
        self.timing = list(range(0, self.sps, max(1, self.sps // 8)))       # offsets d'horloge testés (grille fine)
        self.bits = {d: [] for d in self.timing}     # bits démodulés par offset, persistants
        self._lock = None
        self.messages = []
        self.synced = False
        self.q = 0.0
        self._text = ""

    def process(self, audio):
        if audio is None or len(audio) == 0:
            return []
        x = np.asarray(audio, dtype=np.float32).flatten()
        self.buf = np.concatenate([self.buf, x])
        if len(self.buf) > self.fs * 30:             # on garde 30 s max
            trim = len(self.buf) - int(self.fs * 30)
            self.buf = self.buf[-int(self.fs * 30):]
            dropped = trim // self.sps
            for d in self.timing:
                self.bits[d] = self.bits[d][dropped:] if dropped < len(self.bits[d]) else []
        self._demod_sitor()
        return self._decode_sitor()

    def _demod_sitor(self):
        """Démodule les nouveaux échantillons pour chaque offset d'horloge (corrélation vectorisée)."""
        n = self.sps
        total = (len(self.buf) - n) // n + 1
        t = np.arange(n) / self.fs
        lo_m = np.exp(2j * np.pi * self.f_mark * t)
        lo_s = np.exp(2j * np.pi * self.f_space * t)
        for d in self.timing:
            have = len(self.bits[d])
            if total <= have:
                continue
            start = have * n + d
            if start >= len(self.buf):
                continue
            seg = self.buf[start:start + (total - have) * n]
            chunks = np.lib.stride_tricks.sliding_window_view(seg, n)[::n]
            cm = np.abs(chunks @ lo_m)
            cs = np.abs(chunks @ lo_s)
            self.bits[d].extend((cm > cs).astype(np.int64).tolist())

    def _find_lock(self):
        """Cherche (polarité, offset d'horloge, phase de bit) qui maximise les caractères
        communs décodés (la polarité ne se voit pas au poids 3/4 : le complément d'un mot
        valide est valide, mais décode des caractères exotiques)."""
        best = None
        for inv in (False, True):
            for d in self.timing:
                bits0 = self.bits[d]
                if len(bits0) < 140:
                    continue
                b = np.asarray(bits0, dtype=np.uint8)
                if inv:
                    b = 1 - b
                for off in range(7):
                    w = _words_of(b, off)
                    if len(w) < 20:
                        continue
                    wm, valid, agree = _combine_pairs(w)
                    np_ = len(wm)
                    if np_ < 10 or not valid.any():
                        continue
                    ncommon = int(_COMMON_W[wm[valid]].sum())
                    nagree = int(agree[valid].sum())
                    score = (ncommon, nagree)
                    if best is None or score > best[0]:
                        best = (score, inv, d, off, ncommon, np_)
        if best is None:
            return None
        _, inv, d, off, ncommon, np_ = best
        if ncommon < 8 or ncommon < 0.4 * np_:
            return None
        return (inv, d, off)

    def _decode_sitor(self):
        out = []
        if not self.synced:
            lock = self._find_lock()
            if lock is None:
                return out
            self._lock = lock
            self.synced = True
        inv, d, off = self._lock
        b = np.asarray(self.bits[d], dtype=np.uint8)
        if inv:
            b = 1 - b
        w = _words_of(b, off)
        wm, valid, agree = _combine_pairs(w)
        npairs = len(wm)
        if valid.any():
            self.q = float(agree[valid].sum() / valid.sum())
        if len(wm) > 40 and self.q < 0.5:            # on a perdu le verrou : on re-cherche
            self.synced = False
            self._lock = None
            self.q = 0.0
            self._text = ""
            return out
        # texte complet recalculé à chaque appel (les paires incomplètes sont révisées
        # quand leurs bits arrivent) ; le mot final seul (fin de trame) est décodé tel quel
        chars = [chr(int(_INVT[wm[i]])) for i in range(len(wm)) if valid[i]]
        if len(w) % 2 == 1 and _WT[w[-1]]:
            chars.append(chr(int(_INVT[w[-1]])))
        text = "".join(chars)
        self._text = text
        # un message NAVTEX est émis complet quand sa fin "NNNN" est décodée
        if "NNNN" in text:
            i = text.rfind("ZCZC")
            msg = text[i:] if i >= 0 else text
            if len(msg) > 3 and msg not in self.messages[-5:]:
                self.messages.append(msg)
                out.append(msg)
        return out


NavtexDecoder = NAVTEXDecoder  # alias utilisé par hfdigi.make_decoder("NAVTEX", ...)
