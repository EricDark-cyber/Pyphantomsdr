"""Corrections audio : graves, aigus et filtres coupe-bande (notch) placés à la main."""
import numpy as np
from ..filters import StreamFIR, lowpass_taps, taps_for_transition


class AudioFX:
    """Un seul filtre RIF réunit tous les réglages.

    On part d'une impulsion (filtre « qui ne fait rien ») et on lui ajoute ou
    retranche des morceaux : un passe-bas à 250 Hz pour doser les graves, le
    complément d'un passe-bas à 3 kHz pour les aigus, et un passe-bande étroit
    retranché pour chaque notch.
    """

    def __init__(self, fs, sharp=True):
        self.fs = fs
        self.ntaps = taps_for_transition(fs, 150.0 if sharp else 400.0)
        self.fir = None
        self.settings = ((), 0.0, 0.0)

    def configure(self, notches=(), bass_db=0.0, treble_db=0.0):
        """notches : liste de (fréquence centrale en Hz, largeur en Hz)."""
        notches = tuple((float(f), float(w)) for f, w in notches if 0 < f < 0.45 * self.fs)
        settings = (notches, float(bass_db), float(treble_db))
        if settings == self.settings:
            return
        self.settings = settings
        if not notches and bass_db == 0 and treble_db == 0:
            self.fir = None
            return
        n = np.arange(self.ntaps) - (self.ntaps - 1) / 2
        gb, gt = 10 ** (bass_db / 20.0), 10 ** (treble_db / 20.0)
        h = np.zeros(self.ntaps)
        h[(self.ntaps - 1) // 2] = gt
        h += (gb - 1.0) * lowpass_taps(250.0, self.fs, self.ntaps)
        h -= (gt - 1.0) * lowpass_taps(min(3000.0, 0.4 * self.fs), self.fs, self.ntaps)
        for f0, width in notches:
            h -= 2.0 * lowpass_taps(max(width, 40.0) / 2, self.fs, self.ntaps) * np.cos(2 * np.pi * f0 / self.fs * n)
        self.fir = StreamFIR(h)

    @property
    def active(self):
        return self.fir is not None

    def process(self, x):
        return x if self.fir is None else self.fir.process(x.astype(np.float32, copy=False))


class Biquad:
    """Filtre biquad simple (peaking / notch / etc.) en flux."""

    def __init__(self):
        self.b0 = self.b1 = self.b2 = 1.0
        self.a1 = self.a2 = 0.0
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0
        self.enabled = False

    def set_peaking(self, fs, f0, gain_db, q):
        """Filtre peaking (boost ou cut)."""
        if abs(gain_db) < 0.1 or f0 <= 0 or f0 >= 0.49 * fs or q <= 0:
            self.enabled = False
            return
        A = 10 ** (gain_db / 40.0)
        w0 = 2 * np.pi * f0 / fs
        alpha = np.sin(w0) / (2 * q)
        cosw = np.cos(w0)
        b0 = 1 + alpha * A
        b1 = -2 * cosw
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cosw
        a2 = 1 - alpha / A
        self.b0, self.b1, self.b2 = b0 / a0, b1 / a0, b2 / a0
        self.a1, self.a2 = a1 / a0, a2 / a0
        self.enabled = True

    def process(self, x):
        if not self.enabled or len(x) == 0:
            return x
        y = np.empty_like(x, dtype=np.float32)
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        for i, xi in enumerate(x):
            yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            y[i] = yi
            x2, x1 = x1, xi
            y2, y1 = y1, yi
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        return y


class Contour:
    """Façonnage doux de la bande passante audio (style FT-710 Contour).

    Accentue ou atténue une zone de fréquences pour améliorer l'intelligibilité.
    """

    def __init__(self, fs):
        self.fs = fs
        self.bq = Biquad()
        self.enabled = False
        self.freq = 1500.0
        self.gain_db = 0.0
        self.q = 1.0

    def configure(self, enabled, freq=1500.0, gain_db=0.0, width_hz=800.0):
        self.enabled = bool(enabled)
        self.freq = float(freq)
        self.gain_db = float(gain_db)
        # Convertir largeur en Q approximatif
        self.q = max(self.freq / max(width_hz, 50.0), 0.3)
        if self.enabled:
            self.bq.set_peaking(self.fs, self.freq, self.gain_db, self.q)
        else:
            self.bq.enabled = False

    def process(self, x):
        return self.bq.process(x) if self.enabled else x


class APF:
    """Audio Peak Filter pour CW : pic étroit centré sur la note CW."""

    def __init__(self, fs):
        self.fs = fs
        self.bq = Biquad()
        self.enabled = False
        self.center = 700.0
        self.width = 100.0

    def configure(self, enabled, center=700.0, width_hz=100.0, gain_db=12.0):
        self.enabled = bool(enabled)
        self.center = float(center)
        self.width = float(width_hz)
        q = max(self.center / max(width_hz, 20.0), 2.0)
        if self.enabled:
            self.bq.set_peaking(self.fs, self.center, gain_db, q)
        else:
            self.bq.enabled = False

    def process(self, x):
        return self.bq.process(x) if self.enabled else x


class DeepNotch:
    """Notch IIR très profond destiné à protéger l'AGC.

    Placé juste avant l'AGC, il empêche une porteuse parasite de faire baisser
    le gain et d'affaiblir le signal utile.
    """

    def __init__(self, fs):
        self.fs = fs
        self.bq = Biquad()
        self.enabled = False
        self.freq = 1000.0
        self.q = 30.0

    def configure(self, enabled, freq=1000.0, q=30.0):
        self.enabled = bool(enabled)
        self.freq = float(freq)
        self.q = float(q)
        if self.enabled and 50 < freq < 0.45 * self.fs:
            # Notch = peaking avec gain très négatif
            self.bq.set_peaking(self.fs, self.freq, -40.0, self.q)
        else:
            self.bq.enabled = False

    def process(self, x):
        return self.bq.process(x) if self.enabled else x
