"""Signaux horaires en grandes ondes : DCF77 (Allemagne, 77,5 kHz) et l'émetteur d'Allouis (France, 162 kHz).

Les deux envoient un bit par seconde ; une minute complète donne l'heure et la date de la
minute qui commence.  La seconde 59 ne porte aucun bit : ce « trou » repère le début de la minute.

- DCF77 abaisse la puissance de sa porteuse au début de chaque seconde : 0,1 s pour un 0, 0,2 s pour un 1.
- Allouis garde une puissance constante et fait osciller la PHASE de sa porteuse : un motif en
  dents de scie de 0,1 s pour un 0, répété deux fois pour un 1.

Ici la porteuse est reçue en BLU ou en CW : elle donne une note fixe, dont on suit
l'amplitude (DCF77) ou la phase (Allouis).
"""
import collections
import numpy as np
from .hfdigi import Baseband

DAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")


def _bcd(bits, weights):
    return sum(w for b, w in zip(bits, weights) if b)


def decode_minute(bits):
    """59 bits (secondes 0 à 58) -> dictionnaire heure/date, ou None si un contrôle de parité échoue."""
    if len(bits) != 59 or not bits[20]:
        return None
    if sum(bits[21:29]) % 2 or sum(bits[29:36]) % 2 or sum(bits[36:59]) % 2:
        return None
    minute = _bcd(bits[21:28], (1, 2, 4, 8, 10, 20, 40))
    hour = _bcd(bits[29:35], (1, 2, 4, 8, 10, 20))
    day = _bcd(bits[36:42], (1, 2, 4, 8, 10, 20))
    weekday = _bcd(bits[42:45], (1, 2, 4))
    month = _bcd(bits[45:50], (1, 2, 4, 8, 10))
    year = _bcd(bits[50:58], (1, 2, 4, 8, 10, 20, 40, 80))
    if minute > 59 or hour > 23 or not 1 <= day <= 31 or not 1 <= month <= 12 or not 1 <= weekday <= 7:
        return None
    return dict(minute=minute, hour=hour, day=day, weekday=weekday, month=month, year=2000 + year,
                summer=bool(bits[17]) and not bits[18])


def encode_minute(minute, hour, day, weekday, month, year, summer=False):
    """Les 59 bits d'une minute (sert aux essais)."""
    def bcd(value, weights):
        out = []
        for w in reversed(weights):
            out.append(1 if value >= w else 0)
            value -= w * out[-1]
        return out[::-1]
    bits = [0] * 59
    bits[17], bits[18], bits[20] = int(summer), int(not summer), 1
    bits[21:28] = bcd(minute, (1, 2, 4, 8, 10, 20, 40)); bits[28] = sum(bits[21:28]) % 2
    bits[29:35] = bcd(hour, (1, 2, 4, 8, 10, 20)); bits[35] = sum(bits[29:35]) % 2
    bits[36:42] = bcd(day, (1, 2, 4, 8, 10, 20)); bits[42:45] = bcd(weekday, (1, 2, 4))
    bits[45:50] = bcd(month, (1, 2, 4, 8, 10)); bits[50:58] = bcd(year % 100, (1, 2, 4, 8, 10, 20, 40, 80))
    bits[58] = sum(bits[36:58]) % 2
    return bits


class _TimeSignal:
    NAME = ""

    def _setup(self, emit):
        self.emit = emit
        self.bits = []
        self.started = False                                # vrai dès qu'un début de minute a été repéré
        self.minutes = 0
        self.last = "en attente d'un début de minute (jusqu'à une minute)"
        self.history = collections.deque(maxlen=1500)
        self.thr_db = None

    def status(self):
        return "%s : seconde %d — %s" % (self.NAME, len(self.bits) if self.started else 0, self.last)

    def _bit(self, bit):
        if self.started:
            self.bits.append(int(bit))
            self.emit(str(int(bit)))
            if len(self.bits) > 59:                         # une minute ne compte que 59 bits : le repère a été manqué
                self.started, self.bits = False, []
                self.emit(" ?\n")

    def _minute_mark(self):
        """Le trou de la seconde 59 vient de passer : les bits accumulés forment (peut-être) une minute complète."""
        if self.started:
            t = decode_minute(self.bits)
            if t is None:
                self.last = "minute reçue avec des erreurs (%d bits)" % len(self.bits)
                self.emit("  -> minute incomplète ou erronée\n")
            else:
                self.minutes += 1
                self.last = "%s %02d/%02d/%d %02d:%02d (heure %s)" % (
                    DAYS[t["weekday"] - 1], t["day"], t["month"], t["year"], t["hour"], t["minute"],
                    "d'été" if t["summer"] else "d'hiver")
                self.emit("  -> %s\n" % self.last)
        self.started, self.bits = True, []


class DCF77Decoder(_TimeSignal):
    NAME = "DCF77"

    def __init__(self, fs, center, emit):
        self._setup(emit)
        self.bb = Baseband(fs, center, 40.0, 60.0)
        self.fs_d = self.bb.fs_out
        self.smooth = max(int(0.02 * self.fs_d), 1)
        self.tail = np.zeros(0)
        self.ref = 0.0                                      # amplitude de la porteuse à pleine puissance
        self.n = 0
        self.low = False
        self.low_start = 0
        self.last_start = None

    def process(self, audio):
        z = self.bb.process(audio)
        if len(z) == 0:
            return
        env = np.concatenate([self.tail, np.abs(z)])
        self.tail = env[-(self.smooth - 1):] if self.smooth > 1 else np.zeros(0)
        env = np.convolve(env, np.ones(self.smooth) / self.smooth, "valid")
        self.history.extend(20 * np.log10(env[::5] + 1e-12))
        for v in env:
            self.ref = v if v > self.ref else self.ref + 0.0005 * (v - self.ref)     # monte vite, redescend lentement
            low = v < (0.65 if self.low else 0.5) * self.ref                         # avec une marge contre les rebonds
            if low and not self.low:
                self.low_start = self.n
            elif self.low and not low:
                self._pulse(self.low_start, (self.n - self.low_start) / self.fs_d)
            self.low = low
            self.n += 1
        self.thr_db = 20 * np.log10(0.5 * self.ref + 1e-12)

    def _pulse(self, start, seconds):
        if not 0.04 <= seconds <= 0.32:
            return                                          # parasite ou évanouissement : ce n'est pas un bit
        if self.last_start is not None and 1.5 < (start - self.last_start) / self.fs_d < 2.5:
            self._minute_mark()                             # deux secondes sans abaissement : début de minute
        self.last_start = start
        self._bit(seconds > 0.15)


class ALS162Decoder(_TimeSignal):
    NAME = "Allouis 162 kHz"
    RAMP_S = 0.025                                          # la phase monte de 1 radian en 25 ms, descend de 2 en 50 ms, remonte de 1

    def __init__(self, fs, center, emit):
        self._setup(emit)
        self.bb = Baseband(fs, center, 45.0, 60.0)
        self.fs_d = self.bb.fs_out
        n = int(round(4 * self.RAMP_S * self.fs_d))
        t = (np.arange(n) + 0.5) / n
        self.template = np.where(t < 0.25, 4 * t, np.where(t < 0.75, 2 - 4 * t, 4 * t - 4))   # dents de scie, de -1 à +1
        self.template -= self.template.mean()
        self.prev = 0j
        self.phase = 0.0
        self.buf = np.zeros(0)                              # phase de la porteuse (radians, déroulée)
        self.n0 = 0                                         # numéro absolu du premier échantillon de buf
        self.profile = np.zeros(50)                         # où tombe le motif dans la seconde (cases de 20 ms)
        self.next_s = None                                  # prochaine seconde à lire (temps absolu, en secondes)
        self.typical = 0.0
        self.gap = False
        self._seen = 0.0                                    # jusqu'où le repérage des secondes a déjà été fait

    def process(self, audio):
        z = self.bb.process(audio).astype(np.complex128)
        if len(z) == 0:
            return
        self.history.extend(20 * np.log10(np.abs(z[::5]) + 1e-12))
        d = np.angle(z * np.conj(np.concatenate([[self.prev], z[:-1]])))
        self.prev = z[-1]
        ph = self.phase + np.cumsum(d)
        self.phase = float(ph[-1])
        self.buf = np.concatenate([self.buf, ph])
        fs = self.fs_d
        half = int(0.5 * fs)
        if len(self.buf) < 4 * half:
            return
        # une erreur d'accord fait dériver la phase en ligne droite : on retranche la moyenne sur une seconde
        csum = np.concatenate([[0.0], np.cumsum(self.buf)])
        mid = np.arange(half, len(self.buf) - half)
        flat = self.buf[mid] - (csum[mid + half] - csum[mid - half]) / (2 * half)
        corr = np.correlate(flat, self.template, "valid") / np.sum(self.template ** 2)
        t = (self.n0 + half + np.arange(len(corr))) / fs    # instant du début du motif cherché
        fresh = t >= self._seen
        if fresh.any():                                     # repérage des secondes : le motif revient à heure fixe
            self.profile *= 0.97 ** (np.count_nonzero(fresh) / fs)
            np.add.at(self.profile, ((t[fresh] % 1.0) * 50).astype(np.int64) % 50, np.maximum(corr[fresh], 0.0) ** 2)
            self._seen = t[-1] + 0.5 / fs
        best = int(np.argmax(self.profile))
        tau = (best + 0.5) / 50.0
        if self.next_s is None and t[-1] > 4.0:           # quatre secondes d'écoute suffisent à repérer le rythme
            self.next_s = np.ceil(t[0] - tau) + tau
        while self.next_s is not None and self.next_s + 0.35 < t[-1]:
            if abs(((self.next_s - tau + 0.5) % 1.0) - 0.5) > 0.03:
                self.next_s = np.round(self.next_s - tau) + tau              # le repère s'est précisé : on se recale
            if self.next_s < t[0] + 0.02:
                self.next_s += 1.0
                continue

            def strength(at):
                i = int(round((at - t[0]) * fs))
                w = max(int(0.012 * fs), 1)
                return float(corr[max(i - w, 0):i + w + 1].max())
            first, second = strength(self.next_s), strength(self.next_s + 4 * self.RAMP_S)
            if first < 0.4 * self.typical:                  # aucun motif : c'est la seconde 59
                self.gap = True
            else:
                self.typical += 0.2 * (first - self.typical) if self.typical else first
                if self.gap:
                    self._minute_mark()
                    self.gap = False
                self._bit(second > 0.5 * first)
            self.next_s += 1.0
        drop = len(self.buf) - int(4.0 * fs)                # on garde quatre secondes de recul
        if drop > 0:
            self.buf = self.buf[drop:]
            self.n0 += drop
