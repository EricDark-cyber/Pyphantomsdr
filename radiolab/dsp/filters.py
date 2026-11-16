"""Filtres, décimation et rééchantillonnage « en flux ».

Tous les blocs de ce fichier gardent un état interne entre deux appels : on peut
leur donner le signal par morceaux de taille quelconque et obtenir exactement
la même sortie que si on l'avait donné d'un seul coup.
"""
import numpy as np


# ----------------------------------------------------------------- conception
def lowpass_taps(cutoff_hz, fs, ntaps, beta=8.0, window=None):
    """Passe-bas RIF par la méthode de la fenêtre (sinus cardinal x Kaiser).

    cutoff_hz : fréquence de coupure à -6 dB.  beta=8 donne ~80 dB d'atténuation.
    Largeur de transition ~ 5 * fs / ntaps : plus il y a de coefficients, plus
    le flanc du filtre est raide.
    """
    ntaps = int(ntaps) | 1                      # impair : retard entier
    n = np.arange(ntaps) - (ntaps - 1) / 2
    if window in (None, "Kaiser"):
        w = np.kaiser(ntaps, beta)
    else:                                       # fenêtres de l'analyse spectrale, réutilisées pour le filtre
        try:
            from .spectrum import WINDOWS
            w = WINDOWS[window][1](ntaps + 1)[1:]   # forme symétrique
        except Exception:
            w = np.kaiser(ntaps, beta)              # repli sur Kaiser si pas de module spectrum
    h = 2 * cutoff_hz / fs * np.sinc(2 * cutoff_hz / fs * n) * w
    return (h / h.sum()).astype(np.float64)


def bandpass_taps(lo_hz, hi_hz, fs, ntaps, beta=8.0, window=None):
    """Passe-bande COMPLEXE laissant passer [lo, hi] (fréquences signées).

    C'est un passe-bas de largeur (hi-lo) décalé en fréquence : on le multiplie
    par exp(j.2pi.fc.n).  Il n'est pas symétrique autour de 0 Hz, ce qui permet
    de garder une seule bande latérale (BLU).
    """
    h = lowpass_taps((hi_hz - lo_hz) / 2, fs, ntaps, beta, window)
    n = np.arange(len(h)) - (len(h) - 1) / 2
    return (h * np.exp(2j * np.pi * (hi_hz + lo_hz) / 2 / fs * n)).astype(np.complex64)


def taps_for_transition(fs, transition_hz, lo=31, hi=1501):
    """Nombre de coefficients nécessaire pour une largeur de transition donnée."""
    return int(np.clip(5 * fs / max(transition_hz, 1.0), lo, hi)) | 1


# ------------------------------------------------------------------ blocs RIF
class StreamFIR:
    """Filtre RIF avec décimation optionnelle (on garde 1 échantillon sur M)."""

    def __init__(self, taps, decim=1):
        taps = np.asarray(taps)
        self.taps = taps.astype(np.complex64 if np.iscomplexobj(taps) else np.float32)
        self.decim = int(decim)
        self.hist = np.zeros(len(taps) - 1, dtype=np.float32)
        self.phase = 0

    def process(self, x):
        if len(x) == 0:
            return x[:0]
        buf = np.concatenate([self.hist.astype(x.dtype, copy=False), x])
        y = np.convolve(buf, self.taps, "valid")          # len(y) == len(x)
        self.hist = buf[len(buf) - (len(self.taps) - 1):]
        if self.decim == 1:
            return y
        out = y[self.phase::self.decim]
        self.phase = (self.phase - len(y)) % self.decim
        return out


class PolyDDC:
    """Changement de fréquence + filtrage + décimation par M, en une passe.

    C'est le bloc qui travaille au débit du récepteur (2,5 MS/s) : il doit être
    économe.  Deux idées :

    1. On ne calcule que les sorties utiles.  Un filtre de L = K.M coefficients
       suivi d'une décimation par M coûte K multiplications par échantillon
       d'entrée au lieu de L (forme « polyphase »).
    2. Au lieu de décaler le signal (une exponentielle complexe par échantillon
       d'entrée), on décale le FILTRE : ses coefficients sont multipliés une
       fois pour toutes par exp(-j.w.i).  Il reste une rotation de phase à
       appliquer en sortie, au débit réduit.
    """

    def __init__(self, fs_in, decim, pass_hz):
        self.fs_in, self.M = float(fs_in), int(decim)
        fs_out = self.fs_in / self.M
        stop_hz = fs_out - pass_hz                  # début de la zone qui se replie sur la bande utile
        if stop_hz <= pass_hz:
            raise ValueError("bande utile trop large pour ce taux de décimation")
        ntaps = int(np.ceil(5 * self.fs_in / (stop_hz - pass_hz) / self.M)) * self.M
        self.K = ntaps // self.M
        n = np.arange(ntaps) - (ntaps - 1) / 2
        fc = (pass_hz + stop_hz) / 2
        h = 2 * fc / self.fs_in * np.sinc(2 * fc / self.fs_in * n) * np.kaiser(ntaps, 8.0)
        self.h = h / h.sum()
        self.left = np.zeros(ntaps - self.M, dtype=np.complex64)
        self.set_offset(0.0)

    def set_offset(self, offset_hz):
        """Fréquence (relative au centre) à ramener à 0 Hz."""
        self.w = 2 * np.pi * offset_hz / self.fs_in
        i = np.arange(len(self.h))
        self.g = (self.h[::-1] * np.exp(-1j * self.w * i)).astype(np.complex64)
        self.phi = 0.0

    def process(self, x):
        buf = np.concatenate([self.left, x.astype(np.complex64, copy=False)])
        M, L = self.M, len(self.g)
        n_out = (len(buf) - L) // M + 1 if len(buf) >= L else 0
        if n_out <= 0:
            self.left = buf
            return np.zeros(0, dtype=np.complex64)
        acc = np.zeros(n_out, dtype=np.complex64)
        for k in range(self.K):                      # K produits matrice-vecteur
            acc += buf[k * M:k * M + n_out * M].reshape(n_out, M) @ self.g[k * M:(k + 1) * M]
        self.left = buf[n_out * M:]
        step = -self.w * M                           # rotation de phase résiduelle
        rot = np.exp(1j * (self.phi + step * np.arange(n_out))).astype(np.complex64)
        self.phi = float((self.phi + step * n_out) % (2 * np.pi))
        return acc * rot


class Mixer:
    """Décalage en fréquence simple (oscillateur numérique à phase continue)."""

    def __init__(self, fs):
        self.fs, self.w, self.phi = float(fs), 0.0, 0.0

    def set_offset(self, offset_hz):
        self.w = 2 * np.pi * offset_hz / self.fs

    def process(self, x):
        rot = np.exp(-1j * (self.phi + self.w * np.arange(len(x)))).astype(np.complex64)
        self.phi = float((self.phi + self.w * len(x)) % (2 * np.pi))
        return x.astype(np.complex64, copy=False) * rot


class LinearResampler:
    """Rééchantillonnage de rapport quelconque par interpolation linéaire.

    Suffisant pour amener l'audio (déjà filtré bien en dessous de la moitié du
    débit) vers les 48 kHz de la carte son.
    """

    def __init__(self, fs_in, fs_out):
        self.step = float(fs_in) / float(fs_out)
        self.pos = 0.0
        self.last = np.zeros(1, dtype=np.float32)

    def process(self, x):
        if len(x) == 0:
            return x
        buf = np.concatenate([self.last, x.astype(np.float32, copy=False)])
        span = len(buf) - 1
        n = max(int(np.ceil((span - self.pos) / self.step)), 0)
        t = self.pos + self.step * np.arange(n)
        i = np.minimum(t.astype(np.int64), span - 1)
        frac = (t - i).astype(np.float32)
        out = buf[i] * (1 - frac) + buf[i + 1] * frac
        self.pos = self.pos + n * self.step - span
        self.last = buf[-1:]
        return out


def exp_taps(tau_s, fs):
    """Filtre RC du premier ordre (désaccentuation FM) approché par un RIF."""
    a = np.exp(-1.0 / (tau_s * fs))
    n = np.arange(max(int(6 * tau_s * fs), 2))
    h = (1 - a) * a ** n
    return h / h.sum()
