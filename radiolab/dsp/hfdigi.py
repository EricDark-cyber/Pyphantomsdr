"""Modes numériques HF décodés à partir de l'audio BLU : RTTY, PSK31 / PSK63 et morse (CW).

Les trois décodeurs commencent de la même façon : la fréquence audio du signal
(par exemple 1000 Hz) est ramenée à 0 Hz et le débit réduit à 500 Hz environ,
bien assez pour des signaux de quelques dizaines de hertz de large.
"""
import collections
import numpy as np
from .filters import StreamFIR, lowpass_taps, taps_for_transition

# Varicode du PSK31 (spécification de G3PLX), caractères ASCII 0 à 127
VARICODE = """1010101011 1011011011 1011101101 1101110111 1011101011 1101011111 1011101111 1011111101
1011111111 11101111 11101 1101101111 1011011101 11111 1101110101 1110101011 1011110111 1011110101 1110101101
1110101111 1101011011 1101101011 1101101101 1101010111 1101111011 1101111101 1110110111 1101010101 1101011101
1110111011 1011111011 1101111111 1 111111111 101011111 111110101 111011011 1011010101 1010111011 101111111
11111011 11110111 101101111 111011111 1110101 110101 1010111 110101111 10110111 10111101 11101101 11111111
101110111 101011011 101101011 110101101 110101011 110110111 11110101 110111101 111101101 1010101 111010111
1010101111 1010111101 1111101 11101011 10101101 10110101 1110111 11011011 11111101 101010101 1111111
111111101 101111101 11010111 10111011 11011101 10101011 11010101 111011101 10101111 1101111 1101101
101010111 110110101 101011101 101110101 101111011 1010101101 111110111 111101111 111111011 1010111111
101101101 1011011111 1011 1011111 101111 101101 11 111101 1011011 101011 1101 111101011 10111111 11011
111011 1111 111 111111 110111111 10101 10111 101 110111 1111011 1101011 11011111 1011101 111010101
1010110111 110111011 1010110101 1011010111 1110110101""".split()
_VARICODE = {code: chr(i) for i, code in enumerate(VARICODE)}

# Baudot / ITA2 : l'indice est la valeur des 5 bits, premier bit émis = poids faible
_LETTERS = "\x00E\nA SIU\rDRJNFCKTZLWHYPQOBG\x0eMXV\x0f"
_FIGURES = "\x003\n- '87\r\x054\x07,!:(5+)2#6019?&\x0e./=\x0f"

MORSE = {".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F", "--.": "G", "....": "H",
         "..": "I", ".---": "J", "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O", ".--.": "P",
         "--.-": "Q", ".-.": "R", "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
         "-.--": "Y", "--..": "Z", "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
         ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9", ".-.-.-": ".", "--..--": ",",
         "..--..": "?", "-..-.": "/", "-...-": "=", ".-.-.": "+", "-....-": "-", ".--.-.": "@", "---...": ":",
         ".----.": "'", "-.--.": "(", "-.--.-": ")", "...-.-": "<SK>", "-.-.-": "<KA>", ".-...": "<AS>"}


class Baseband:
    """Ramène la fréquence audio `center` à 0 Hz et réduit le débit à ~500 Hz."""

    def __init__(self, fs, center, cutoff, transition):
        self.fs = fs
        self.D = max(int(round(fs / 500.0)), 1)
        self.fs_out = fs / self.D
        self.center = float(center)
        self.phi = 0.0
        self.fir = StreamFIR(lowpass_taps(cutoff, fs, taps_for_transition(fs, transition)), self.D)

    def process(self, audio):
        w = 2 * np.pi * self.center / self.fs
        n = len(audio)
        lo = np.exp(-1j * (self.phi + w * np.arange(n))).astype(np.complex64)
        self.phi = float((self.phi + w * n) % (2 * np.pi))
        # on garde le complexe : jeter la partie imaginaire ici (astype float32) casserait le signal
        return self.fir.process(np.asarray(audio) * lo)


class PSKDecoder:
    """PSK31 / PSK63 : la porteuse garde sa phase pour un 1 et l'inverse pour un 0."""

    def __init__(self, fs, center, baud, emit):
        self.emit, self.baud, self.user_center = emit, baud, float(center)
        self.bb = Baseband(fs, center, 1.3 * baud, 2.5 * baud)
        self.sps = self.bb.fs_out / baud                          # 16 échantillons par symbole en PSK31
        k = int(round(self.sps))
        self.mf = StreamFIR(np.ones(k) / k)                       # moyenne sur un symbole
        self.n = 0
        self.E = 0j
        self.t_last = None
        self.tail = np.zeros(1, dtype=np.complex128)
        self.prev = 0j
        self.R, self.P = 0j, 1e-30
        self.quality = 0.0
        self.code = ""
        self.zeros = 0
        self.points = collections.deque(maxlen=300)               # constellation : un point par symbole reçu
        self.history = collections.deque(maxlen=1500)             # niveau du signal en dB, pour l'affichage
        self.thr_db = None

    def process(self, audio):
        y = self.mf.process(self.bb.process(audio)).astype(np.complex128)
        if len(y) == 0:
            return
        self.history.extend(20 * np.log10(np.abs(y[::5]) + 1e-12))
        idx = self.n + np.arange(len(y))
        # rythme : l'amplitude est maximale au milieu de chaque symbole
        self.E = 0.95 * self.E + complex(np.sum(np.abs(y) ** 2 * np.exp(-2j * np.pi * idx / self.sps)))
        n_peak = -np.angle(self.E) * self.sps / (2 * np.pi)
        buf = np.concatenate([self.tail, y])
        base, last = self.n - 1, self.n + len(y) - 1
        t_exp = base + self.sps if self.t_last is None else self.t_last + self.sps
        t0 = max(n_peak + np.round((t_exp - n_peak) / self.sps) * self.sps, base)
        count = int(np.floor((last - t0) / self.sps)) + 1
        if count > 0:
            t = t0 + self.sps * np.arange(count)
            sym = buf[np.clip(np.round(t - base).astype(np.int64), 0, len(buf) - 1)]
            d = sym * np.conj(np.concatenate([[self.prev], sym[:-1]]))     # +A² (phase gardée) ou -A² (inversée)
            self.prev, self.t_last = sym[-1], float(t[-1])
            # un écart de fréquence fait tourner d ; d² tourne deux fois plus mais ne dépend plus du bit
            self.R = 0.9 * self.R + complex(np.sum(d * d))
            self.P = 0.9 * self.P + float(np.sum(np.abs(d) ** 2))
            self.quality = abs(self.R) / self.P
            rot = 0.5 * np.angle(self.R)
            # constellation : +1 (phase gardée) ou -1 (phase inversée) quand le signal est propre
            self.points.extend(d * np.exp(-1j * rot) / (np.mean(np.abs(d)) + 1e-30))
            if self.quality > 0.5:                                         # accord fin automatique (AFC)
                df = rot / (2 * np.pi) * self.baud
                new = self.bb.center + 0.1 * df
                if abs(new - self.user_center) < 0.9 * self.baud:
                    self.bb.center = new
                    self.R *= np.exp(-2j * 0.1 * rot)
                for bit in (d * np.exp(-1j * rot)).real > 0:
                    self._bit(bit)
        self.tail = buf[-1:]
        self.n += len(y)

    def _bit(self, one):
        if one:
            self.code = "1" if self.zeros >= 2 else self.code + "0" * self.zeros + "1"
            self.zeros = 0
            if len(self.code) > 12:
                self.code = ""
        else:
            self.zeros += 1
            if self.zeros == 2:                                            # « 00 » sépare les caractères
                ch = _VARICODE.get(self.code)
                if ch is not None and (ch >= " " or ch in "\r\n"):
                    self.emit(ch)
                self.code = ""


class RTTYDecoder:
    """RTTY : deux fréquences écartées de 170 Hz (repos = « mark », travail = « space »), code Baudot."""

    def __init__(self, fs, center, baud, emit, mark_high=True, shift=170.0):
        self.emit = emit
        self.bb = Baseband(fs, center, shift / 2 + 1.2 * baud, 150.0)
        fs_d = self.bb.fs_out
        self.bit = fs_d / baud                                             # échantillons par bit (~11)
        k = int(round(self.bit))
        tone = np.exp(2j * np.pi * (shift / 2) / fs_d * np.arange(k)) / k
        sign = 1 if mark_high else -1
        self.f_mark = StreamFIR(tone if sign > 0 else np.conj(tone))       # énergie autour de +85 Hz
        self.f_space = StreamFIR(np.conj(tone) if sign > 0 else tone)      # énergie autour de -85 Hz
        self.state = "idle"
        self.t = 0.0
        self.bits = []
        self.figures = False
        self.q = 0.5
        self.was_mark = True
        self.history = collections.deque(maxlen=1500)
        self.thr_db = None

    def process(self, audio):
        z = self.bb.process(audio)
        if len(z) == 0:
            return
        m, s = np.abs(self.f_mark.process(z)), np.abs(self.f_space.process(z))
        self.history.extend(20 * np.log10(np.maximum(m, s)[::5] + 1e-12))
        x = (m - s) / (m + s + 1e-30)                                      # +1 = mark, -1 = space
        for v in x:
            mark = v > 0
            if self.state == "idle":
                if self.was_mark and not mark:                             # front descendant : bit de départ
                    self.state, self.t, self.bits = "char", 0.0, []
            else:
                self.t += 1.0
                k = len(self.bits)
                if k == 0 and abs(self.t - 0.5 * self.bit) < 0.5 and mark:
                    self.state = "idle"                                    # fausse alerte : ce n'était pas un départ
                elif k < 5 and self.t >= (1.5 + k) * self.bit:
                    self.bits.append(mark)
                    self.q += 0.1 * (abs(v) - self.q)                      # au milieu d'un bit, un vrai signal donne |x| ~ 1
                elif k == 5 and self.t >= 6.4 * self.bit:
                    if mark and self.q > 0.6:                              # bit d'arrêt correct
                        self._char(sum(int(b) << i for i, b in enumerate(self.bits)))
                    self.state = "idle"
            self.was_mark = mark

    def _char(self, code):
        if code == 27:
            self.figures = True
        elif code == 31:
            self.figures = False
        else:
            ch = (_FIGURES if self.figures else _LETTERS)[code]
            if ch == " ":
                self.figures = False                                       # retour aux lettres après une espace
            if ch >= " " or ch in "\r\n":
                self.emit(ch)


class CWDecoder:
    """Morse : mesure la durée des signaux et des silences, et s'adapte à la vitesse de l'opérateur."""

    def __init__(self, fs, center, emit):
        self.emit = emit
        self.bb = Baseband(fs, center, 60.0, 120.0)
        self.dt = 1000.0 / self.bb.fs_out                                  # millisecondes par échantillon
        self.smooth = StreamFIR(np.ones(4) / 4)
        self.peak, self.floor = 0.0, None
        self.warm = []
        self.key = False
        self.run = 0.0                                                     # durée de l'état courant (ms)
        self.dot = 60.0                                                    # durée d'un point (60 ms = 20 mots/min)
        self.recent = collections.deque(maxlen=16)
        self.marks = []
        self.spaced = True
        self.auto_speed = True
        self.manual_thr = None                                             # seuil imposé (amplitude), ou None = automatique
        self.history = collections.deque(maxlen=1500)                      # enveloppe en dB (100 points par seconde)
        self.thr_db = None                                                 # seuil en vigueur, pour l'affichage
        self._count = 0

    def set_speed(self, wpm):
        """Vitesse imposée en mots par minute, ou None pour l'estimation automatique."""
        self.auto_speed = wpm is None
        if wpm is not None:
            self.dot = 1200.0 / float(wpm)

    def set_threshold_db(self, db):
        """Seuil imposé (dB, même échelle que le tracé), ou None pour le seuil automatique."""
        self.manual_thr = None if db is None else 10.0 ** (db / 20.0)

    def wpm(self):
        return 1200.0 / self.dot

    def process(self, audio):
        env = np.abs(self.smooth.process(self.bb.process(audio)))
        for v in env:
            # niveau du bruit : quartile inférieur de l'enveloppe, suivi pas à pas (le manipulateur est
            # levé plus de la moitié du temps, ce quartile tombe donc toujours dans les silences)
            self.peak = max(self.peak * 0.9995, v)
            if self.floor is None:                                         # mise en route : première estimation
                self.warm.append(float(v))
                if len(self.warm) < 300:
                    continue
                self.floor = float(np.percentile(self.warm[50:], 25)) + 1e-30
            self.floor *= (1 - 0.003) if v < self.floor else (1 + 0.001)
            self._count += 1
            if self._count % 5 == 0:
                self.history.append(20 * np.log10(v + 1e-12))
            if self.manual_thr is not None:
                thr = self.manual_thr
                on = v > thr * (1.0 if not self.key else 0.85)
            else:
                # le seuil suit le signal le plus fort récent, mais ne descend jamais dans le bruit :
                # entre deux émissions, les pointes du bruit ne doivent pas être prises pour des points
                thr = max(self.floor + 0.45 * (self.peak - self.floor), 8.0 * self.floor)   # au moins 18 dB au-dessus du bruit
                on = v > thr * (1.0 if not self.key else 0.75)
            self.thr_db = 20 * np.log10(thr + 1e-12)
            self.run += self.dt
            if on != self.key:
                if self.key:
                    if self.run >= 12.0:                                   # fin d'un signal
                        self.marks.append(self.run)
                        self.recent.append(self.run)
                        self._update_speed()
                else:
                    if self.run > 2.0 * self.dot:                          # silence long : le caractère est fini
                        self._flush(self.run)
                self.key, self.run = on, 0.0
            elif not self.key and self.marks and self.run > 5.0 * self.dot:
                self._flush(self.run)
            elif not self.key and not self.spaced and self.run > 5.0 * self.dot:
                self.emit(" ")
                self.spaced = True

    def _update_speed(self):
        s = sorted(self.recent)
        if len(s) < 3 or not self.auto_speed:
            return
        ratios = [s[i + 1] / s[i] for i in range(len(s) - 1)]
        i = int(np.argmax(ratios))
        if ratios[i] >= 1.7:                                               # deux familles : points et traits
            dot = float(np.mean(s[:i + 1]))
        else:                                                              # une seule famille : laquelle ?
            mean = float(np.mean(s))
            dot = mean / 3.0 if 2.0 * self.dot < mean < 4.5 * self.dot else mean
        self.dot = float(np.clip(0.5 * self.dot + 0.5 * dot, 20.0, 240.0))  # 5 à 60 mots par minute

    def _flush(self, gap):
        if self.marks:
            code = "".join("-" if d > 2.0 * self.dot else "." for d in self.marks)
            self.emit(MORSE.get(code, "?"))
            self.marks = []
            self.spaced = False
        if gap > 5.0 * self.dot and not self.spaced:
            self.emit(" ")
            self.spaced = True


KINDS = ("RTTY45", "RTTY50", "PSK31", "PSK63", "CW", "PACKET300", "APRS1200", "NAVTEX", "FAX", "SSTV", "WSPR", "FT8", "FT4", "GOLAY", "HELL", "DCF77", "ALS162")


def _audio_decoder(dec, emit):
    """Adapte un décodeur qui RETOURNE une liste de messages en décodeur à callback `emit`
    (le receiver appelle digi.process() sans lire la valeur de retour)."""
    orig = dec.process

    def process(audio):
        try:
            r = orig(audio)
        except Exception:
            return
        if r:
            for m in r:
                emit(str(m))

    dec.process = process
    return dec


def make_decoder(kind, fs, center, emit, mark_high=True):
    if kind == "PSK31":
        return PSKDecoder(fs, center, 31.25, emit)
    if kind == "PSK63":
        return PSKDecoder(fs, center, 62.5, emit)
    if kind == "RTTY45":
        return RTTYDecoder(fs, center, 45.45, emit, mark_high)
    if kind == "RTTY50":
        return RTTYDecoder(fs, center, 50.0, emit, mark_high)
    if kind == "CW":
        return CWDecoder(fs, center, emit)
    if kind == "NAVTEX":                          # le sens des deux fréquences est trouvé automatiquement
        from .navtex import NAVTEXDecoder
        return _audio_decoder(NAVTEXDecoder(fs, center, emit), emit)
    if kind == "FAX":                             # images : la note audio va de 1500 Hz (noir) à 2300 Hz (blanc)
        from .pictures import FaxDecoder
        return FaxDecoder(fs, emit)
    if kind == "SSTV":
        from .pictures import SSTVDecoder
        return SSTVDecoder(fs, emit)
    if kind == "HELL":
        from .pictures import HellDecoder
        return HellDecoder(fs, center, emit)
    if kind == "DCF77":                           # signaux horaires : la porteuse doit donner une note à `center`
        from .timesignals import DCF77Decoder
        return DCF77Decoder(fs, center, emit)
    if kind == "ALS162":
        from .timesignals import ALS162Decoder
        return ALS162Decoder(fs, center, emit)
    if kind == "WSPR":                            # balises de faible puissance, autour de 1500 Hz audio
        from .wspr import WSPRDecoder
        return WSPRDecoder(fs, emit)              # le décodeur WSPR émet directement via le callback
    if kind in ("FT8", "FT4"):
        from .ft8 import FT8Decoder, FT4Decoder
        return FT4Decoder(fs, emit, center) if kind == "FT4" else FT8Decoder(fs, emit, center=center)
    if kind == "GOLAY":
        from .golay import GolayDecoder
        return _audio_decoder(GolayDecoder(), emit)
    if kind == "PACKET300":                       # packet HF : deux tonalités à 200 Hz d'écart, autour de `center`
        from .packet import AFSKDecoder
        return AFSKDecoder(fs, 300.0, center - 100.0, center + 100.0, emit)
    if kind == "APRS1200":                        # APRS / packet VHF (Bell 202), reçu en FM étroite
        from .packet import AFSKDecoder
        return AFSKDecoder(fs, 1200.0, 1200.0, 2200.0, emit)
    # décodeurs "audio" du menu qui retournent leurs messages (liste) plutôt que par callback
    if kind == "JS8":
        from .js8 import JS8Decoder
        return _audio_decoder(JS8Decoder(fs), emit)
    if kind == "JT65":
        from .jt65 import JT65Decoder
        return _audio_decoder(JT65Decoder(fs), emit)
    if kind == "M17":
        from .m17 import M17Decoder
        return _audio_decoder(M17Decoder(fs), emit)
    if kind == "DSTAR":
        from .dstar import DStarDecoder
        return _audio_decoder(DStarDecoder(fs), emit)
    if kind == "DMR":
        from .dmr import DMRDecoder
        return _audio_decoder(DMRDecoder(fs), emit)
    if kind == "C4FM":
        from .c4fm import C4FMDecoder
        return _audio_decoder(C4FMDecoder(fs), emit)
    if kind == "OLIVIA":
        from .olivia import OliviaDecoder
        return _audio_decoder(OliviaDecoder(fs), emit)
    if kind == "CONTESTIA":
        from .contestia import ContestiaDecoder
        return _audio_decoder(ContestiaDecoder(fs), emit)
    return None
