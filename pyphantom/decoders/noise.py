"""Réduction de bruit : noise blanker, filtre adaptatif (notch / LMS), NR spectrale.

Chaque classe vise un type de bruit :
  NoiseBlanker         impulsions brèves (clôtures électriques, allumage, CPL)
  LineEnhancer         porteuses (mode "notch") ou souffle (mode "enhance")
  SpectralNR           souffle large bande, par gain spectral
"""
import numpy as np


class NoiseBlanker:
    """Efface les impulsions sur le signal I/Q, AVANT le filtre étroit.

    Une impulsion est brève et beaucoup plus forte que le reste : on compare le
    module de chaque échantillon à `threshold` fois le niveau médian.  Les
    échantillons touchés (plus `guard` voisins de chaque côté) sont remplacés
    par une interpolation linéaire entre les échantillons sains.

    Il faut agir avant le filtre étroit : un filtre de 2,4 kHz étale une
    impulsion de quelques microsecondes sur une milliseconde, et elle devient
    alors impossible à séparer du signal.
    """

    def __init__(self, threshold=4.0, guard=2):
        self.threshold, self.guard = float(threshold), int(guard)
        self.enabled = False
        self.ref = None
        self.tail = np.zeros(4 * self.guard, dtype=np.complex64)   # retard d'anticipation
        self.blanked = 0

    def process(self, x):
        buf = np.concatenate([self.tail, x])
        d = len(self.tail)
        if self.enabled and len(buf) > 2 * d:
            mag = np.abs(buf)
            med = float(np.median(mag))
            self.ref = med if self.ref is None else self.ref + 0.2 * (med - self.ref)
            mask = mag > self.threshold * self.ref
            if mask.any():
                mask = np.convolve(mask, np.ones(2 * self.guard + 1), "same") > 0.5
                good = ~mask
                if good.sum() >= 2:
                    idx = np.arange(len(buf))
                    buf = buf.copy()
                    buf[mask] = (np.interp(idx[mask], idx[good], buf.real[good])
                                 + 1j * np.interp(idx[mask], idx[good], buf.imag[good]))
                    self.blanked += int(mask.sum())
        self.tail = buf[len(buf) - d:]
        return buf[:len(buf) - d]


class LineEnhancer:
    """Filtre adaptatif NLMS en prédicteur linéaire.

    Le filtre essaie de prédire l'échantillon courant à partir d'échantillons
    plus anciens (retardés de `delay`).  Seul ce qui est PRÉVISIBLE sur cette
    durée peut être prédit :

      * avec un long retard, seules les porteuses pures restent prévisibles ;
        la sortie « erreur » (signal - prédiction) est donc le signal débarrassé
        des porteuses : c'est le notch automatique (mode "notch") ;
      * avec un retard court, la voix et la télégraphie restent prévisibles mais
        pas le souffle ; on garde alors la prédiction (mode "enhance").

    Les coefficients sont mis à jour par blocs de `block` échantillons (LMS par
    blocs normalisé), ce qui évite une boucle Python par échantillon.
    """

    def __init__(self, ntaps=64, delay=64, mu=0.5, leak=1e-4, mode="notch", block=8):
        self.ntaps, self.delay, self.mu, self.leak = ntaps, delay, mu, leak
        self.mode, self.block = mode, block
        self.enabled = False
        self.reset()

    def reset(self):
        self.w = np.zeros(self.ntaps, dtype=np.float64)
        self.hist = np.zeros(self.ntaps + self.delay - 1, dtype=np.float64)

    def process(self, x):
        if not self.enabled or len(x) == 0:
            return x
        x64 = x.astype(np.float64)
        buf = np.concatenate([self.hist, x64])
        # U[t] = les ntaps échantillons qui précèdent x[t] d'au moins `delay`
        U = np.lib.stride_tricks.sliding_window_view(buf, self.ntaps)[:len(x64)]
        y = np.empty_like(x64)
        w, B = self.w, self.block
        for i in range(0, len(x64), B):
            Ub = U[i:i + B]
            yb = Ub @ w                                   # prédiction
            eb = x64[i:i + B] - yb                        # erreur
            y[i:i + B] = yb
            power = np.einsum("ij,ij->", Ub, Ub) + 1e-9
            w = (1 - self.leak) * w + self.mu * (Ub.T @ eb) / power   # pas normalisé : stable pour mu < 2
        self.w = w
        self.hist = buf[len(buf) - len(self.hist):]
        out = (x64 - y) if self.mode == "notch" else y
        return out.astype(np.float32)


def expint_e1(x):
    """Exponentielle intégrale E1(x), x > 0 (Abramowitz & Stegun 5.1.53 et 5.1.56)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    s = x <= 1.0
    xs = np.maximum(x[s], 1e-300)
    out[s] = (-np.log(xs) - 0.57721566 + xs * (0.99999193 + xs * (-0.24991055 + xs * (
        0.05519968 + xs * (-0.00976004 + xs * 0.00107857)))))
    xl = x[~s]
    num = xl ** 4 + 8.5733287401 * xl ** 3 + 18.0590169730 * xl ** 2 + 8.6347608925 * xl + 0.2677737343
    den = xl ** 4 + 9.5733223454 * xl ** 3 + 25.6329561486 * xl ** 2 + 21.0996530827 * xl + 3.9584969228
    out[~s] = num / den * np.exp(-xl) / xl
    return out


class SpectralNR:
    """Réduction de bruit par gain spectral (soustraction spectrale, Wiener, MMSE-LSA).

    Principe : on découpe l'audio en trames de 20 ms qui se recouvrent, on passe
    chaque trame en fréquence (FFT), on multiplie chaque case par un gain entre
    `floor` et 1 selon qu'elle contient surtout du bruit ou surtout du signal,
    puis on revient en temps et on recolle les trames.

    Deux estimations sont nécessaires à chaque trame et pour chaque case :
      * la puissance du BRUIT, suivie en continu par la méthode MCRA
        (Cohen & Berdugo, 2002) : le minimum récent du spectre lissé indique
        le niveau de bruit, et on ne met à jour l'estimation que là où la
        parole semble absente ;
      * le rapport signal/bruit « a priori », par l'approche « décision
        dirigée » d'Ephraim et Malah (1984), qui lisse le gain dans le temps
        et évite le « bruit musical » de la soustraction spectrale simple.
    """

    METHODS = ("lsa", "wiener", "specsub")

    def __init__(self, fs, nfft=256, method="lsa", reduction_db=18.0):
        self.fs, self.nfft, self.hop = fs, nfft, nfft // 4
        self.method, self.reduction_db = method, reduction_db
        self.enabled = False
        n = np.arange(nfft)
        self.win = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * n / nfft))   # racine de Hann, périodique
        self.norm = 1.0 / 2.0                    # somme des fenêtres de Hann à 75 % de recouvrement
        self.reset()

    def reset(self):
        nb = self.nfft // 2 + 1
        self.inbuf = np.zeros(self.nfft - self.hop, dtype=np.float64)
        self.ola = np.zeros(self.nfft, dtype=np.float64)
        self.pending = np.zeros(0, dtype=np.float64)
        self.noise = None
        self.S = self.Smin = self.Stmp = None
        self.p = np.zeros(nb)
        self.G_prev = np.ones(nb)
        self.gamma_prev = np.ones(nb)
        self.frame = 0
        self.sub = max(int(0.4 * self.fs / self.hop), 4)     # fenêtre de recherche du minimum ~0,8 s

    def _gain(self, P):
        if self.noise is None:                                # première trame : tout est du bruit
            self.noise = P.copy(); self.S = P.copy(); self.Smin = P.copy(); self.Stmp = P.copy()
        # --- estimation du bruit (MCRA)
        self.S = 0.8 * self.S + 0.2 * P
        self.Smin = np.minimum(self.Smin, self.S)
        self.Stmp = np.minimum(self.Stmp, self.S)
        self.frame += 1
        if self.frame % self.sub == 0:
            self.Smin = np.minimum(self.Stmp, self.S)
            self.Stmp = self.S.copy()
        # parole probablement présente : spectre lissé OU trame courante bien au-dessus du minimum
        present = (self.S > 5.0 * self.Smin) | (P > 12.0 * self.Smin)
        self.p = np.maximum(present, 0.8 * self.p)            # détection immédiate, relâchement lent
        a = 0.95 + 0.05 * self.p                              # parole présente -> on fige l'estimation
        self.noise = a * self.noise + (1 - a) * P
        # --- gain
        floor = 10.0 ** (-self.reduction_db / 20.0)
        gamma = np.minimum(P / (self.noise + 1e-20), 1e3)     # SNR a posteriori
        if self.method == "specsub":
            G = np.sqrt(np.maximum(1.0 - 2.0 / np.maximum(gamma, 1e-6), 0.0))
        else:
            xi = 0.98 * self.G_prev ** 2 * self.gamma_prev + 0.02 * np.maximum(gamma - 1.0, 0.0)
            xi = np.maximum(xi, 10.0 ** (-25 / 10.0))         # SNR a priori (décision dirigée)
            G = xi / (1.0 + xi)                               # filtre de Wiener
            if self.method == "lsa":
                v = np.maximum(G * gamma, 1e-8)
                G = G * np.exp(0.5 * expint_e1(v))            # MMSE-LSA (Ephraim & Malah, 1985)
        G = np.clip(G, floor, 1.0)
        self.G_prev, self.gamma_prev = G, gamma
        return G

    def process(self, x):
        if not self.enabled or len(x) == 0:
            return x
        data = np.concatenate([self.pending, x.astype(np.float64)])
        nframes = len(data) // self.hop
        out = np.empty(nframes * self.hop, dtype=np.float64)
        for f in range(nframes):
            chunk = data[f * self.hop:(f + 1) * self.hop]
            frame = np.concatenate([self.inbuf, chunk])
            self.inbuf = frame[self.hop:]
            X = np.fft.rfft(frame * self.win)
            G = self._gain(X.real ** 2 + X.imag ** 2)
            self.ola += np.fft.irfft(X * G, self.nfft) * self.win
            out[f * self.hop:(f + 1) * self.hop] = self.ola[:self.hop] * self.norm
            self.ola = np.concatenate([self.ola[self.hop:], np.zeros(self.hop)])
        self.pending = data[nframes * self.hop:]
        return out.astype(np.float32)
