"""NAVTEX : bulletins de sécurité maritime (518 kHz en anglais, 490 kHz en langue locale).

Transmission « SITOR-B » (recommandation CCIR 476) : 100 bauds, deux fréquences écartées de
170 Hz, caractères de 7 bits dont EXACTEMENT 4 sont à 1 (35 mots possibles, ce qui permet de
repérer les erreurs).  Chaque caractère est émis deux fois, le second exemplaire CINQ
CARACTÈRES PLUS TARD (entrelacement « DX / RX ») : si l'un des deux est abîmé par un parasite,
on prend l'autre.

Comme en télétype, le même mot de 7 bits vaut une lettre ou un chiffre selon le dernier
signal « lettres » / « chiffres » reçu.

Interface :
    d = NAVTEXDecoder(fs, center, emit)
    d.process(audio)       # le texte sort caractère par caractère par emit(...)
    d.messages             # bulletins complets (de ZCZC à NNNN)
    d.synced, d.q          # accroché ? proportion de caractères valides (0 à 1)
Sans `emit`, process() retourne la liste des bulletins complets terminés pendant cet appel.

`encode(text)` fabrique la suite de mots de 7 bits d'une vraie émission (mise en phase puis
texte entrelacé) : elle sert aux essais.
"""
import collections

import numpy as np

from .filters import StreamFIR
from .hfdigi import Baseband

BAUD = 100.0
SHIFT = 170.0

# Code CCIR 476, premier bit émis à gauche (1 = fréquence haute) : lettre, chiffre/signe, mot.
_TABLE = """A - 1110001   B ? 0100111   C : 1011100   D $ 1100101   E 3 0110101   F ! 1101100   G & 1010110
H # 1001011   I 8 1011001   J * 1110100   K ( 0111100   L ) 1010011   M . 1001110   N , 1001101
O 9 1000111   P 0 1011010   Q 1 0111010   R 4 1010101   S ' 1101001   T 5 0010111   U 7 0111001
V = 0011110   W 2 1110010   X / 0101110   Y 6 1101010   Z + 1100011"""
LETTERS, FIGURES = {}, {}
_items = _TABLE.split()
for _i in range(0, len(_items), 3):
    _code = int(_items[_i + 2], 2)
    LETTERS[_code] = _items[_i]
    FIGURES[_code] = _items[_i + 1] if _items[_i + 1] not in "$*" else ""       # « qui est là ? » et sonnerie : ignorés
CR, LF, SPACE = 0b0001111, 0b0011011, 0b0011101
LTRS, FIGS = 0b0101101, 0b0110110
ALPHA, BETA, REP = 0b1111000, 0b1100110, 0b0110011           # signaux de service (attente et mise en phase)
for _table in (LETTERS, FIGURES):
    _table.update({CR: "\r", LF: "\n", SPACE: " "})
ENCODE_LETTERS = {v: k for k, v in LETTERS.items()}
ENCODE_FIGURES = {v: k for k, v in FIGURES.items() if v}
del _items, _i, _code, _table


def valid(code):
    """Un mot CCIR 476 a exactement 4 bits à 1."""
    return bin(code).count("1") == 4


class NAVTEXDecoder:
    def __init__(self, fs=8000, center=None, emit=None, baud=BAUD, shift=SHIFT):
        self.fs = float(fs)
        self.center = float(center) if center else 1700.0
        self.emit = emit
        self.bb = Baseband(self.fs, self.center, shift / 2 + 1.2 * baud, 150.0)
        fs_d = self.bb.fs_out
        self.spb = fs_d / baud                                 # échantillons par bit
        k = max(int(round(self.spb)), 1)
        tone = np.exp(2j * np.pi * (shift / 2) / fs_d * np.arange(k)) / k
        self.f_hi, self.f_lo = StreamFIR(tone), StreamFIR(np.conj(tone))    # un filtre adapté par fréquence
        self.phase = 0.0                                       # position dans le bit en cours (0 à 1)
        self.acc = 0.0
        self.prev = 0.0
        self.bits = []
        self.synced = False
        self.invert = False                                    # sens des deux fréquences, trouvé à l'accrochage
        self.codes = collections.deque(maxlen=12)              # derniers caractères reçus (valides ou non)
        self.count = 0                                         # nombre de caractères depuis l'accrochage
        self.score = [0.0, 0.0]                                # l'exemplaire répété tombe-t-il sur les rangs pairs ou impairs ?
        self.figures = False
        self.q = 0.0
        self.history = collections.deque(maxlen=1500)
        self.thr_db = None
        self.messages = []                                     # bulletins complets (ZCZC ... NNNN)
        self._text = ""                                        # texte en cours
        self._new = []                                         # bulletins terminés pendant l'appel en cours

    def status(self):
        return "NAVTEX : %s, caractères valides %.0f %%, %d bulletin(s)" % (
            "accroché" if self.synced else "recherche", 100 * self.q, len(self.messages))

    def process(self, audio):
        self._new = []
        if audio is None or len(audio) == 0:
            return []
        z = self.bb.process(np.asarray(audio, dtype=np.float32).ravel())
        if len(z) == 0:
            return []
        hi, lo = np.abs(self.f_hi.process(z)), np.abs(self.f_lo.process(z))
        self.history.extend(20 * np.log10(np.maximum(hi, lo)[::5] + 1e-12))
        x = (hi - lo) / (hi + lo + 1e-30)                      # +1 = fréquence haute, -1 = fréquence basse
        step = 1.0 / self.spb
        for v in x:
            if (v > 0) != (self.prev > 0):                     # changement de fréquence : il doit tomber entre deux bits
                err = self.phase if self.phase < 0.5 else self.phase - 1.0
                self.phase -= 0.15 * err
            self.prev = v
            self.acc += v
            self.phase += step
            if self.phase >= 1.0:
                self.phase -= 1.0
                self._bit(self.acc > 0)
                self.acc = 0.0
        return [] if self.emit is not None else list(self._new)

    def _bit(self, one):
        self.bits.append(int(one))
        if not self.synced:
            if len(self.bits) < 14 * 7 + 6:
                return
            best = None
            for offset in range(7):                            # où commencent les caractères, et dans quel sens ?
                words = [int("".join(map(str, self.bits[offset + 7 * i:offset + 7 * i + 7])), 2) for i in range(14)]
                for invert in (False, True):
                    good = sum(valid(w ^ 0x7F if invert else w) for w in words)
                    if best is None or good > best[0]:
                        best = (good, offset, invert)
            if best[0] >= 13:
                self.synced, self.invert = True, best[2]
                self.bits = self.bits[best[1]:]
                self.codes.clear()
                self.count, self.score, self.q = 0, [0.0, 0.0], 1.0
            else:
                self.bits = self.bits[7:]
                return
        while self.synced and len(self.bits) >= 7:
            word = int("".join(map(str, self.bits[:7])), 2)
            self.bits = self.bits[7:]
            self._code(word ^ 0x7F if self.invert else word)

    def _code(self, code):
        ok = valid(code)
        self.q += 0.05 * (float(ok) - self.q)
        if self.q < 0.5:                                       # trop d'erreurs : le signal est perdu, on recherche à nouveau
            self.synced = False
            return
        self.codes.append(code)
        self.count += 1
        if len(self.codes) < 6:
            return
        first = self.codes[-6]                                 # émis cinq caractères plus tôt
        parity = self.count % 2
        self.score[parity] *= 0.98
        self.score[1 - parity] *= 0.98
        if ok and code == first:
            self.score[parity] += 1.0
        if code == ALPHA:                                      # le signal d'attente n'occupe que les rangs « répétition »
            self.score[parity] += 0.25
        if self.score[parity] <= self.score[1 - parity] or self.score[parity] < 2.0:
            return
        chosen = first if valid(first) else (code if ok else None)
        if chosen is None:
            self._char("_")                                    # les deux exemplaires sont abîmés
        elif chosen == FIGS:
            self.figures = True
        elif chosen == LTRS:
            self.figures = False
        elif chosen not in (ALPHA, BETA, REP):
            ch = (FIGURES if self.figures else LETTERS).get(chosen, "")
            if ch:
                self._char(ch)

    def _char(self, ch):
        if self.emit is not None:
            self.emit(ch)
        self._text = (self._text + ch)[-20000:]
        if self._text.endswith("NNNN"):                        # fin de bulletin : on le range en entier
            i = self._text.rfind("ZCZC")
            msg = self._text[i:] if i >= 0 else self._text
            self._text = ""
            if len(msg) > 8 and msg not in self.messages[-5:]:
                self.messages.append(msg)
                del self.messages[:-50]
                self._new.append(msg)


NavtexDecoder = NAVTEXDecoder                                  # ancien nom


def encode(text):
    """Suite des mots de 7 bits à émettre pour `text` (sert aux essais) : mise en phase, puis
    chaque caractère une première fois et une seconde fois cinq caractères plus tard."""
    msg, figures = [], False
    for ch in text.upper():
        if ch in "\r\n ":
            msg.append({"\r": CR, "\n": LF, " ": SPACE}[ch])
        elif ch in ENCODE_LETTERS:
            if figures:
                msg.append(LTRS); figures = False
            msg.append(ENCODE_LETTERS[ch])
        elif ch in ENCODE_FIGURES:
            if not figures:
                msg.append(FIGS); figures = True
            msg.append(ENCODE_FIGURES[ch])
    slots = []
    for _ in range(20):
        slots += [REP, ALPHA]                                  # mise en phase
    start = len(slots)
    slots += [ALPHA] * (2 * len(msg) + 6)
    for i, code in enumerate(msg):
        slots[start + 2 * i] = code                            # premier exemplaire
        slots[start + 2 * i + 5] = code                        # second exemplaire, cinq caractères plus tard
    return slots
