"""DSP optimisé avec Numba pour performances proches de SDR#.

Utilise la compilation JIT pour accélérer les calculs.
"""
import numpy as np

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@jit(nopython=True, cache=True, fastmath=True)
def fft_power_fast(iq, n):
    """Calcul de puissance FFT optimisé."""
    # Fenêtre de Hann
    window = np.empty(n, dtype=np.float64)
    for i in range(n):
        window[i] = 0.5 * (1 - np.cos(2 * np.pi * i / n))
    # Appliquer la fenêtre
    windowed = iq[:n] * window
    # FFT
    spec = np.fft.fft(windowed)
    # Puissance en dB
    power = np.empty(n, dtype=np.float64)
    for i in range(n):
        p = spec[i].real ** 2 + spec[i].imag ** 2
        power[i] = 10 * np.log10(p + 1e-10)
    return power


@jit(nopython=True, cache=True, fastmath=True)
def demod_fm_fast(iq, prev_phase):
    """Démodulation FM optimisée."""
    n = len(iq)
    audio = np.empty(n, dtype=np.float64)
    phase = prev_phase
    for i in range(n):
        new_phase = np.arctan2(iq[i].imag, iq[i].real)
        diff = new_phase - phase
        # Unwrap
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
        audio[i] = diff
        phase = new_phase
    return audio, phase


@jit(nopython=True, cache=True, fastmath=True)
def demod_am_fast(iq):
    """Démodulation AM optimisée."""
    n = len(iq)
    audio = np.empty(n, dtype=np.float64)
    for i in range(n):
        audio[i] = np.sqrt(iq[i].real ** 2 + iq[i].imag ** 2)
    # Retirer la composante DC
    mean = np.mean(audio)
    for i in range(n):
        audio[i] -= mean
    return audio


@jit(nopython=True, cache=True, fastmath=True)
def filter_fir_fast(signal, coeffs):
    """Filtre FIR optimisé."""
    n = len(signal)
    m = len(coeffs)
    output = np.zeros(n, dtype=np.float64)
    for i in range(n):
        acc = 0.0
        for j in range(min(i + 1, m)):
            acc += signal[i - j] * coeffs[j]
        output[i] = acc
    return output


@jit(nopython=True, cache=True, parallel=True, fastmath=True)
def waterfall_update_fast(wf, new_line):
    """Mise à jour waterfall optimisée."""
    rows, cols = wf.shape
    # Décaler vers le bas
    for i in range(rows - 1, 0, -1):
        for j in prange(cols):
            wf[i, j] = wf[i - 1, j]
    # Nouvelle ligne
    for j in prange(cols):
        wf[0, j] = new_line[j]
    return wf


class OptimizedDSP:
    """Interface DSP optimisée."""

    def __init__(self):
        self.prev_phase = 0.0
        self.has_numba = HAS_NUMBA

    def spectrum(self, iq, n=1024):
        """Calcul de spectre optimisé."""
        if len(iq) < n:
            return np.zeros(n)
        return fft_power_fast(iq[:n], n)

    def demod_fm(self, iq):
        """Démodulation FM optimisée."""
        audio, self.prev_phase = demod_fm_fast(iq, self.prev_phase)
        return audio

    def demod_am(self, iq):
        """Démodulation AM optimisée."""
        return demod_am_fast(iq)

    def filter(self, signal, coeffs):
        """Filtrage FIR optimisé."""
        return filter_fir_fast(signal, coeffs)
