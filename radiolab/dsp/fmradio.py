"""Compléments de la FM : stéréo et RDS (radiodiffusion), tonalité CTCSS (FM étroite).

Le signal démodulé d'un émetteur FM de radiodiffusion (« multiplex ») contient :

    0 - 15 kHz     G + D             ce qu'écoute un récepteur mono
    19 kHz         pilote            signale une émission stéréo et sert de référence de phase
    23 - 53 kHz    (G - D) modulant une porteuse supprimée à 38 kHz = 2 x pilote
    57 kHz         RDS, 1187,5 bit/s, sur une porteuse supprimée à 3 x pilote
"""
import numpy as np
from .filters import PolyDDC, StreamFIR

PILOT_HZ = 19000.0


class StereoPilot:
    """Suit la phase du pilote 19 kHz et reconstruit la porteuse 38 kHz du canal G - D.

    Le pilote est ramené à 0 Hz et filtré très étroit (PolyDDC) : il n'en reste
    qu'un nombre complexe qui tourne lentement, dont l'argument est la phase du
    pilote.  La porteuse 38 kHz vaut sin(2 x phase) puisque l'émetteur la
    fabrique en doublant le pilote.
    """

    def __init__(self, fs):
        self.fs = fs
        self.M = max(int(fs // 10000), 1)
        self.ddc = PolyDDC(fs, self.M, 500.0)
        self.ddc.set_offset(PILOT_HZ)
        self.L = len(self.ddc.g)
        self.w = 2 * np.pi * PILOT_HZ / fs
        self.b = self.L - self.M            # indice (dans le repère du DDC) du prochain échantillon
        self.m = 0                          # nombre de sorties du DDC déjà produites
        self.last_b, self.last_th = None, 0.0
        self.level, self.coherence = 0.0, 0.0
        self.locked = False

    def process(self, mpx):
        """Retourne 2.sin(2.phase) pour chaque échantillon du multiplex (ou des zéros sans pilote)."""
        n = len(mpx)
        p = self.ddc.process(mpx)
        if len(p):
            amp = float(np.mean(np.abs(p))) + 1e-12
            self.level += 0.3 * (amp - self.level)
            # un vrai pilote a une phase stable : sa moyenne complexe reste grande ; du bruit se moyenne à zéro
            self.coherence += 0.3 * (abs(complex(np.mean(p))) / amp - self.coherence)
            self.locked = self.coherence > 0.8 and self.level > 0.004
            bc = (self.m + np.arange(len(p))) * self.M + (self.L - 1) / 2.0
            th = np.angle(p)
            if self.last_b is not None:
                bc, th = np.concatenate([[self.last_b], bc]), np.concatenate([[self.last_th], th])
            th = np.unwrap(th)
            self.pts = (bc, th)
            self.last_b, self.last_th = float(bc[-1]), float(th[-1])
            self.m += len(p)
        idx = self.b + np.arange(n)
        self.b += n
        if not self.locked or self.last_b is None:
            return np.zeros(n, dtype=np.float32)
        theta = np.interp(idx, self.pts[0], self.pts[1])
        # le pilote émis est sin(wt) et la sous-porteuse sin(2wt) ; notre phase est celle d'un cosinus
        return (-2.0 * np.sin(2.0 * (self.w * idx + theta))).astype(np.float32)


# ------------------------------------------------------------------------ RDS
RDS_BITRATE = 1187.5
_POLY = 0x5B9                                    # x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1
_OFFSETS = {0x0FC: "A", 0x198: "B", 0x168: "C", 0x350: "C'", 0x1B4: "D"}
_NEXT = {"A": ("B",), "B": ("C", "C'"), "C": ("D",), "C'": ("D",), "D": ("A",)}
PTY_NAMES = ["", "Infos", "Magazine", "Service", "Sport", "Éducation", "Fiction", "Culture", "Sciences",
             "Divers", "Pop", "Rock", "Chansons", "Classique léger", "Classique", "Autre musique"]


# caractères 0x80 à 0x9B de la table RDS de base (les suivants sont affichés « ? »)
_RDS_ACCENTS = "áàéèíìóòúùÑÇŞß¡Ĳâäêëîïôöûüñç"


def rds_checkword(data16):
    """Les 10 bits de contrôle d'un mot de 16 bits (reste de la division polynomiale)."""
    reg = data16 << 10
    for bit in range(25, 9, -1):
        if reg & (1 << bit):
            reg ^= _POLY << (bit - 10)
    return reg & 0x3FF


class RDSDecoder:
    """Extrait du multiplex FM le nom de la station (PS), le texte (RT) et le code PI."""

    def __init__(self, fs):
        self.D = max(int(fs // 19000), 1)
        self.fs_r = fs / self.D
        self.ddc = PolyDDC(fs, self.D, 2400.0)
        self.ddc.set_offset(3 * PILOT_HZ)
        self.sps = self.fs_r / RDS_BITRATE                       # échantillons par bit (~16)
        half = int(round(self.sps / 2))
        # chaque bit est émis en deux moitiés de signes opposés (codage biphase) : filtre adapté
        self.mf = StreamFIR(np.concatenate([np.ones(half), -np.ones(half)]) / (2 * half))
        self.c = 0j
        self.phi = 0.0
        self.E = 0j
        self.n = 0                                               # indice absolu du prochain échantillon
        self.tail = np.zeros(1, dtype=np.float64)
        self.t_last = None
        self.prev_sym = False
        self.reg = 0
        self.nbits = 0
        self.synced = False
        self.expect = None
        self.count = 0
        self.bad = 0
        self.group = {}
        self.groups_ok = 0
        self.reset_text()

    def reset_text(self):
        self.pi = None
        self.pty = 0
        self.ps = [" "] * 8
        self.rt = [" "] * 64
        self.rt_flag = None

    # --------------------------------------------------- du signal aux bits
    def process(self, mpx):
        b = self.ddc.process(mpx)
        if len(b) == 0:
            return
        # phase de la porteuse : élever au carré efface la modulation +/-1 et laisse 2 x la phase
        c = complex(np.sum(b * b))
        self.c = 0.5 * self.c + 0.5 * c
        phi = 0.5 * np.angle(self.c)
        phi += np.pi * np.round((self.phi - phi) / np.pi)         # continuité (la phase n'est connue qu'à pi près)
        self.phi = float(phi)
        y = self.mf.process((b * np.exp(-1j * phi)).real.astype(np.float32)).astype(np.float64)
        idx = self.n + np.arange(len(y))
        # rythme des bits : l'énergie de y est maximale au milieu de chaque bit (méthode d'Oerder et Meyr)
        self.E = 0.9 * self.E + complex(np.sum(y * y * np.exp(-2j * np.pi * idx / self.sps)))
        n_peak = -np.angle(self.E) * self.sps / (2 * np.pi)
        buf = np.concatenate([self.tail, y])                      # buf[0] a l'indice absolu self.n - 1
        base, last = self.n - 1, self.n + len(y) - 1
        t_exp = base + self.sps if self.t_last is None else self.t_last + self.sps
        t0 = n_peak + np.round((t_exp - n_peak) / self.sps) * self.sps
        t0 = max(t0, base)
        count = int(np.floor((last - t0) / self.sps)) + 1
        if count > 0:
            t = t0 + self.sps * np.arange(count)
            i = np.minimum((t - base).astype(np.int64), len(buf) - 2)
            frac = t - base - i
            sym = (buf[i] * (1 - frac) + buf[i + 1] * frac) > 0
            prev = np.concatenate([[self.prev_sym], sym[:-1]])
            for bit in (sym ^ prev):                              # codage différentiel : bit = changement de signe
                self._push(int(bit))
            self.prev_sym = bool(sym[-1])
            self.t_last = float(t[-1])
        self.tail = buf[-1:]
        self.n += len(y)

    # ------------------------------------------------- des bits aux groupes
    def _push(self, bit):
        self.reg = ((self.reg << 1) | bit) & 0x3FFFFFF
        self.nbits += 1
        if self.nbits < 26:
            return
        data, check = self.reg >> 10, self.reg & 0x3FF
        kind = _OFFSETS.get(rds_checkword(data) ^ check)
        if not self.synced:
            # synchronisation : deux blocs valides de types consécutifs, à 26 bits d'écart
            if kind is not None:
                if self.expect and kind in self.expect[0] and self.nbits - self.expect[1] == 26:
                    self.synced, self.count, self.bad = True, 0, 0
                    self.cur = kind
                    self.group = {}
                    self._block(kind, data)
                    return
                self.expect = (_NEXT[kind], self.nbits)
            return
        self.count += 1
        if self.count < 26:
            return
        self.count = 0
        wanted = _NEXT[self.cur]
        if kind in wanted:
            self.bad = 0
            self.cur = kind
            self._block(kind, data)
        else:
            self.bad += 1
            self.cur = wanted[0]
            self.group = {}
            if self.bad > 12:
                self.synced, self.expect = False, None

    def _block(self, kind, data):
        if kind == "A":
            self.group = {"A": data}
            return
        self.group[kind] = data
        if kind == "D" and "A" in self.group and "B" in self.group:
            self._group(self.group)
            self.group = {}

    def _group(self, g):
        self.groups_ok += 1
        if g["A"] != self.pi:
            if self.pi is not None:
                self.reset_text()
            self.pi = g["A"]
        b = g["B"]
        gtype, version = b >> 12, (b >> 11) & 1
        self.pty = (b >> 5) & 0x1F
        d = g["D"]
        if gtype == 0:                                            # nom de la station, 2 caractères par groupe
            a = (b & 3) * 2
            self.ps[a], self.ps[a + 1] = self._char(d >> 8), self._char(d & 0xFF)
        elif gtype == 2:                                          # radiotexte
            flag = (b >> 4) & 1
            if flag != self.rt_flag:
                self.rt, self.rt_flag = [" "] * 64, flag
            a = b & 0xF
            if version == 0 and "C" in g:
                c = g["C"]
                for k, v in enumerate((c >> 8, c & 0xFF, d >> 8, d & 0xFF)):
                    self._rt_char(a * 4 + k, v)
            elif version == 1:
                for k, v in enumerate((d >> 8, d & 0xFF)):
                    self._rt_char(a * 2 + k, v)

    def _rt_char(self, pos, v):
        if v == 0x0D:                                             # fin du texte
            self.rt[pos:] = [" "] * (64 - pos)
        elif pos < 64:
            self.rt[pos] = self._char(v)

    @staticmethod
    def _char(v):
        if 0x20 <= v < 0x7F:
            return chr(v)
        return _RDS_ACCENTS[v - 0x80] if 0x80 <= v < 0x80 + len(_RDS_ACCENTS) else "?"

    def info(self):
        return {"synced": self.synced, "pi": self.pi, "ps": "".join(self.ps).strip(),
                "rt": "".join(self.rt).strip(), "pty": PTY_NAMES[self.pty] if self.pty < len(PTY_NAMES) else ""}


# ---------------------------------------------------------------------- CTCSS
CTCSS_TONES = [67.0, 69.3, 71.9, 74.4, 77.0, 79.7, 82.5, 85.4, 88.5, 91.5, 94.8, 97.4, 100.0, 103.5, 107.2,
               110.9, 114.8, 118.8, 123.0, 127.3, 131.8, 136.5, 141.3, 146.2, 151.4, 156.7, 159.8, 162.2,
               165.5, 167.9, 171.3, 173.8, 177.3, 179.9, 183.5, 186.2, 189.9, 192.8, 196.6, 199.5, 203.5,
               206.5, 210.7, 218.1, 225.7, 229.1, 233.6, 241.8, 250.3, 254.1]


class ToneDetector:
    """Reconnaît la tonalité subaudible CTCSS d'une émission FM étroite.

    Trois améliorations par rapport à la version d'origine (une mesure par seconde) :
      * fenêtres d'analyse courtes et CHEVAUCHANTES -> détection en ~0,5 s au lieu de ~2 s,
        et captures des émissions brèves ;
      * seuil lié au rapport pic/bruit en bande -> pas de fausse tonalité sur le souffle ;
      * HYSTÉRÉSIS : une tonalité verrouillée est lâchée plus lentement qu'acquise, ce qui
        supprime le clignotement « 88.5 -> — -> 88.5 ».
    """
    ACQUIRE_SNR, HOLD_SNR = 45.0, 18.0         # pic/médiane exigé (bruit ~12, signal ~75-190)
    ACQUIRE_AMP, HOLD_AMP = 0.02, 0.008        # amplitude de la tonalité (écarte le quasi-silence)
    ACQUIRE_RUN = 3                            # ~0,45 s de tonalité stable pour afficher
    HOLD_MISS = 4                              # ~0,6 s d'absence avant de lâcher

    def __init__(self, fs, window_s=0.5, hop_s=0.15):
        self.fs = float(fs)
        self.N = max(int(self.fs * window_s), 64)
        self.hop = max(int(self.fs * hop_s), 1)
        n = np.arange(self.N)
        win = np.hanning(self.N)
        self.win_gain = float(np.sum(win))
        self.table = (np.exp(-2j * np.pi * np.outer(CTCSS_TONES, n) / self.fs) * win).astype(np.complex64)
        self.buf = np.zeros(0, dtype=np.float32)
        self.tone = None
        self.candidate = None                 # conservé pour compatibilité
        self._cand = None
        self._run = 0                         # mesures consécutives concordantes (acquisition)
        self._miss = 0                        # mesures consécutives sans la tonalité verrouillée (maintien)

    def process(self, audio):
        self.buf = np.concatenate([self.buf, audio.astype(np.float32, copy=False)])
        while len(self.buf) >= self.N:
            seg = self.buf[:self.N]
            self.buf = self.buf[self.hop:]    # fenêtres chevauchantes : on n'avance que d'un saut
            self._analyze(seg)

    def _analyze(self, seg):
        power = np.abs(self.table @ seg) ** 2
        k = int(np.argmax(power))
        ref = float(np.median(power)) + 1e-20
        snr = power[k] / ref                                      # pic sur médiane : une raie se détache ?
        amp = 2.0 * float(np.sqrt(power[k])) / self.win_gain      # amplitude de la tonalité
        locked = self.tone is not None
        snr_thr = self.HOLD_SNR if locked else self.ACQUIRE_SNR   # hystérésis : maintien plus indulgent
        amp_thr = self.HOLD_AMP if locked else self.ACQUIRE_AMP
        found = CTCSS_TONES[k] if (snr > snr_thr and amp > amp_thr) else None
        self.candidate = found
        # --- acquisition
        if found is not None and found == self._cand:
            self._run += 1
        else:
            self._run = 1 if found is not None else 0
        self._cand = found
        if found is not None and self._run >= self.ACQUIRE_RUN:
            self.tone, self._miss = found, 0
            return
        # --- maintien (hystérésis) : ne lâcher qu'après plusieurs absences
        if locked:
            if found == self.tone:
                self._miss = 0
            else:
                self._miss += 1
                if self._miss >= self.HOLD_MISS:
                    self.tone, self._miss = None, 0
