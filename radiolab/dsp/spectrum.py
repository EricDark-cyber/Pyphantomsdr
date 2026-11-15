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


# ---------------------------------------------------------------------------------------------
# Loupe (« zoom FFT ») : finesse du spectre quand on regarde une petite partie de la bande
# ---------------------------------------------------------------------------------------------
# Une FFT de N points prise sur toute la bande reçue donne une finesse de fs / N : pour voir
# 1 Hz à 2,5 MHz d'échantillonnage il faudrait 2,5 millions de points, recalculés à chaque image.
# La loupe fait autrement, comme un récepteur :
#   1. elle ramène le milieu de la bande regardée à 0 Hz (multiplication par une exponentielle) ;
#   2. elle filtre et ne garde qu'un échantillon sur D : le débit tombe à fs / D, juste assez
#      pour la bande regardée ;
#   3. elle fait une PETITE FFT sur ce signal réduit.
# La finesse ne dépend plus que de la durée de signal analysée (finesse = 1 / durée), et le calcul
# coûte quelques passes sur les échantillons au lieu d'une FFT géante.
ZOOM_MARGIN = 2.5              # débit réduit = 2,5 fois la bande regardée (place pour la pente du filtre)
ZOOM_T_MAX = 2.0               # durée de signal analysée au plus, en secondes (finesse de 0,5 Hz)
ZOOM_CELLS = 2.0               # durée visée : deux « grains » de finesse par pixel de l'écran
_gain_cache = {}
_KAISER_BETA = 8.6             # fenêtre du filtre de réduction : environ 85 dB d'atténuation hors bande
_plan_cache = {}


def zoom_plan(fs, span_hz):
    """Réduction de débit en deux étages (D1, D2) pour regarder `span_hz`, ou None si la bande
    regardée est trop large pour que la loupe apporte quelque chose."""
    d = int(fs // (ZOOM_MARGIN * max(span_hz, 1e-3)))
    if d < 4:
        return None
    if d >= 32:
        return d // 16, 16
    return 1, d


def _zoom_filters(d1, d2):
    """Coefficients des deux filtres de réduction (somme = 1)."""
    key = (d1, d2)
    if key not in _plan_cache:
        if len(_plan_cache) > 32:
            _plan_cache.clear()
        box = np.ones(d1) / d1                                # premier étage : trois moyennes glissantes en cascade,
        h1 = np.convolve(np.convolve(box, box), box)          # dont les zéros tombent pile sur les bandes repliées
        h1 = np.concatenate([h1, np.zeros(3 * d1 - len(h1))])
        n = 10 * d2                                           # second étage : passe-bas en sinus cardinal fenêtré
        k = np.arange(n) - (n - 1) / 2.0
        h2 = np.sinc(k / d2) * np.kaiser(n, _KAISER_BETA)     # coupure à la moitié du débit réduit
        _plan_cache[key] = (h1, h2 / h2.sum())
    return _plan_cache[key]


def _shift_decimate(x, d, taps, w=0.0):
    """y[m] = somme sur j de taps[j] . x[m.d + j] . exp(-i.w.(m.d + j)) : décalage en fréquence,
    filtrage et réduction de débit en une seule opération, calculée seulement aux instants gardés."""
    k = len(taps) // d
    m = len(x) // d - (k - 1)
    if m <= 0:
        return np.zeros(0, dtype=np.complex64)
    blocks = x[len(x) - (m + k - 1) * d:].reshape(-1, d)      # on travaille sur la fin du signal
    g = taps * np.exp(-1j * w * np.arange(k * d)) if w else taps
    g = g.reshape(k, d).astype(np.complex64)
    y = blocks[:m] @ g[0]
    for j in range(1, k):
        y += blocks[j:j + m] @ g[j]
    if w:
        y *= np.exp(-1j * ((w * d * np.arange(m)) % (2 * np.pi))).astype(np.complex64)
    return y


def _zoom_gain(f_rel, fs, d1, d2):
    """Gain en puissance des deux filtres à la fréquence f_rel (comptée depuis le milieu de la bande regardée)."""
    key = (fs, d1, d2, len(f_rel), float(f_rel[0]), float(f_rel[-1]))
    if key not in _gain_cache:
        if len(_gain_cache) > 8:
            _gain_cache.clear()
        _gain_cache[key] = _zoom_gain_compute(f_rel, fs, d1, d2)
    return _gain_cache[key]


def _zoom_gain_compute(f_rel, fs, d1, d2):
    h1, h2 = _zoom_filters(d1, d2)
    g = np.ones(len(f_rel))
    if d1 > 1:
        x = np.pi * f_rel / fs
        small = np.abs(x) < 1e-9
        num, den = np.sin(d1 * np.where(small, 1.0, x)), d1 * np.sin(np.where(small, 1.0, x))
        g *= np.where(small, 1.0, (num / den) ** 2) ** 3
    resp = np.exp(-2j * np.pi * np.outer(f_rel * d1 / fs, np.arange(len(h2)))) @ h2
    return g * (resp.real ** 2 + resp.imag ** 2)


def zoom_raw_samples(fs, span_hz, npoints, available):
    """Nombre d'échantillons à fournir à zoom_fft : assez pour deux grains de finesse par pixel
    (durée = 2 x npoints / bande), sans dépasser ZOOM_T_MAX ni ce qui est en mémoire."""
    plan = zoom_plan(fs, span_hz)
    if plan is None:
        return 0
    d = plan[0] * plan[1]
    want = min(ZOOM_CELLS * npoints / max(span_hz, 1e-3), ZOOM_T_MAX) * fs + 16 * d
    return int(min(max(want, 1 << 18), available))


def zoom_fft(iq, fs, center_hz, f_lo, f_hi, npoints=1536, window=DEFAULT_WINDOW):
    """Spectre fin de la seule bande [f_lo, f_hi] par la méthode de la loupe.

    Retourne (fréquences, niveaux en dB, nombre d'échantillons d'origine par analyse) ; la finesse
    réelle vaut fs divisé par ce nombre.  Retourne None si la bande est trop large (voir zoom_plan)
    ou s'il n'y a pas assez de signal."""
    span = f_hi - f_lo
    plan = zoom_plan(fs, span)
    if plan is None:
        return None
    d1, d2 = plan
    h1, h2 = _zoom_filters(d1, d2)
    mid = 0.5 * (f_lo + f_hi)
    w = 2 * np.pi * (mid - center_hz) / fs
    if d1 > 1:
        y = _shift_decimate(_shift_decimate(iq, d1, h1, w), d2, h2)
    else:
        y = _shift_decimate(iq, d2, h2, w)
    fs_d = fs / (d1 * d2)
    n = int(min(len(y), max(min(ZOOM_CELLS * npoints / span, ZOOM_T_MAX) * fs_d, 64)))
    if n < 64:
        return None
    nfft = 1 << int(np.ceil(np.log2(2 * n)))                  # complétée par des zéros : courbe lisse entre les points
    win = get_window(window, n)
    hop = max(n // 2, 1)
    count = min((len(y) - n) // hop + 1, 8)                   # moyenne de plusieurs analyses qui se chevauchent
    p = np.zeros(nfft)
    for i in range(count):
        seg = y[len(y) - n - i * hop:len(y) - i * hop]
        spec = np.fft.fft(seg * win, nfft)
        p += spec.real ** 2 + spec.imag ** 2
    p = np.fft.fftshift(p / count)
    f_rel = (np.arange(nfft) - nfft // 2) * (fs_d / nfft)
    keep = np.flatnonzero((f_rel >= -span / 2) & (f_rel <= span / 2))
    f_rel, p = f_rel[keep], p[keep]
    db = (10 * np.log10(p / _zoom_gain(f_rel, fs, d1, d2) + 1e-20)).astype(np.float32)
    freqs = mid + f_rel
    if len(db) > npoints:                                     # plus de points que de pixels : on garde les crêtes
        edges = np.linspace(0, len(db), npoints + 1).astype(np.int64)[:-1]
        step = (freqs[-1] - freqs[0]) / max(len(db) - 1, 1) * len(db) / npoints
        db = np.maximum.reduceat(db, edges)
        freqs = freqs[edges] + 0.5 * step
    return freqs, db, n * d1 * d2


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
