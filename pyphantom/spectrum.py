"""Spectre pour l'affichage (courbe et cascade), avec zoom et choix de la fenêtre d'analyse.

Finesse d'analyse : une FFT de N points sur un signal échantillonné à fs sépare
des fréquences distantes de fs / N.  À 2,5 MS/s, 4096 points donnent 610 Hz par
point ; un million de points donnent 2,4 Hz, mais il faut alors observer le
signal pendant 0,42 s : on ne peut pas être fin à la fois en fréquence et en temps.

La fenêtre (pondération appliquée aux échantillons avant la FFT) règle un autre
compromis : des raies fines mais entourées de « bavures » qui masquent les
signaux faibles voisins, ou des raies plus larges mais un fond bien propre.
"""
import numpy as np

# nom -> (description courte, fonction donnant la fenêtre de n points)
def _cosine_sum(coeffs):
    def make(n):
        x = 2 * np.pi * np.arange(n) / n
        return sum(((-1) ** k) * a * np.cos(k * x) for k, a in enumerate(coeffs))
    return make


WINDOWS = {
    "Blackman-Harris": ("raies larges, fond très propre (-92 dB) : le meilleur choix général",
                        _cosine_sum([0.35875, 0.48829, 0.14128, 0.01168])),
    "Hann": ("bon compromis finesse / propreté", _cosine_sum([0.5, 0.5])),
    "Hamming": ("raies fines, premières bavures à -43 dB", _cosine_sum([0.54, 0.46])),
    "Blackman": ("intermédiaire entre Hann et Blackman-Harris", _cosine_sum([0.42, 0.5, 0.08])),
    "Kaiser": ("réglage classique (bêta = 8), proche de Blackman", lambda n: np.kaiser(n, 8.0)),
    "Flat-top": ("raies très larges mais niveau exact : pour mesurer une amplitude",
                 _cosine_sum([0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368])),
    "Poisson": ("exponentielle (alpha = 2) : bavures qui décroissent sans rebonds",
                lambda n: np.exp(-2.0 * np.abs(np.arange(n) - n / 2) / (n / 2))),
    "Rectangulaire": ("aucune pondération : raies les plus fines, bavures très fortes (-13 dB)",
                      lambda n: np.ones(n)),
}
DEFAULT_WINDOW = "Blackman-Harris"
_cache = {}


def get_window(name, n):
    """Fenêtre normalisée : une sinusoïde pleine échelle donne 0 dB quelle que soit la fenêtre."""
    key = (name, n)
    if key not in _cache:
        if len(_cache) > 6:
            _cache.clear()
        w = WINDOWS.get(name, WINDOWS[DEFAULT_WINDOW])[1](n)
        _cache[key] = (w / w.sum()).astype(np.float32)
    return _cache[key]


def power_spectrum_db(iq, nfft=4096, max_avg=8, window=DEFAULT_WINDOW):
    """Spectre moyen en dB (0 dB = sinusoïde pleine échelle), fréquences croissantes.

    On moyenne jusqu'à `max_avg` FFT prises à la fin du signal : la moyenne lisse
    le bruit et fait ressortir les signaux faibles.
    """
    if len(iq) < nfft:
        return np.full(nfft, -140.0, dtype=np.float32)
    k = max(min(len(iq) // nfft, max_avg), 1)
    frames = iq[len(iq) - k * nfft:].reshape(k, nfft) * get_window(window, nfft)
    spec = np.fft.fft(frames, axis=1)
    p = np.mean(spec.real ** 2 + spec.imag ** 2, axis=0)
    return (10 * np.log10(np.fft.fftshift(p) + 1e-20)).astype(np.float32)


def auto_nfft(fs, span_hz, npoints, available, lo=4096, hi=1 << 20):
    """Taille de FFT donnant au moins un point par pixel sur la largeur affichée."""
    want = fs / max(span_hz, 1.0) * npoints
    n = 1 << int(np.ceil(np.log2(max(want, lo))))
    n = min(n, hi)
    while n > lo and n > available:
        n >>= 1
    return n


def zoom_spectrum(iq, fs, center_hz, f_lo, f_hi, npoints=1536, nfft=4096, window=DEFAULT_WINDOW):
    """Spectre de la seule bande [f_lo, f_hi], ramené à `npoints` points au plus.

    Quand la FFT a plus de points que l'écran n'a de pixels, on garde le maximum
    de chaque groupe de points : une raie fine reste visible au lieu d'être moyennée.
    Retourne (fréquences, niveaux en dB).
    """
    db = power_spectrum_db(iq, nfft, max_avg=max((1 << 18) // nfft, 1), window=window)
    df = fs / nfft
    f0 = center_hz - fs / 2
    i0 = int(np.clip(np.floor((f_lo - f0) / df), 0, nfft - 2))
    i1 = int(np.clip(np.ceil((f_hi - f0) / df), i0 + 2, nfft))
    part = db[i0:i1]
    freqs = f0 + (i0 + np.arange(len(part))) * df
    if len(part) > npoints:
        edges = np.linspace(0, len(part), npoints + 1).astype(np.int64)[:-1]
        part = np.maximum.reduceat(part, edges)
        freqs = freqs[edges] + 0.5 * df * len(db[i0:i1]) / npoints
    return freqs, part


class TraceMemory:
    """Mémoire de la courbe du spectre : moyenne glissante (lisse le bruit) et maintien des crêtes.

    Tout est remis à zéro quand la bande affichée change, puisque les points ne désignent plus
    les mêmes fréquences.
    """

    def __init__(self):
        self.key = None
        self.avg = None                                    # puissances moyennes (échelle linéaire)
        self.peak = None                                   # plus forts niveaux vus, en dB

    def reset(self):
        self.key = None

    def update(self, freqs, db, average=1, hold=False):
        """Retourne (courbe moyennée en dB, courbe des crêtes en dB ou None)."""
        key = (len(freqs), float(freqs[0]), float(freqs[-1]))
        if key != self.key:
            self.key, self.avg, self.peak = key, None, None
        power = 10.0 ** (np.asarray(db, dtype=np.float64) / 10.0)
        if average <= 1 or self.avg is None:
            self.avg = power
        else:
            self.avg = self.avg + (power - self.avg) / float(average)
        shown = 10.0 * np.log10(self.avg + 1e-30)
        if not hold:
            self.peak = None
            return shown, None
        self.peak = shown.copy() if self.peak is None else np.maximum(self.peak, shown)
        return shown, self.peak
