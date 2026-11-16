"""Pont minimal vers les modules hfdigi (non fournis) : Baseband + make_decoder.

Baseband : decalage en frequence (Mixer) + passe-bas (StreamFIR) + decimation,
utilise par timesignals.py (DCF77 / Allouis). make_decoder : aucun decodeur
externe branche pour l'instant (retourne None) — les decodeurs RTTY/PSK31/SSTV
de ce package sont utilises directement.
"""
import numpy as np

from ..filters import Mixer, StreamFIR, lowpass_taps, taps_for_transition


class Baseband:
    """Ramene la bande [center-lo, center+hi] autour de 0 Hz et decime."""

    def __init__(self, fs, center, lo=40.0, hi=60.0):
        self.fs = float(fs)
        self.center = float(center)
        half = float(max(hi, lo, 1.0))
        self.M = max(1, int(self.fs // (8.0 * half)))
        self.fs_out = self.fs / self.M
        self.mix = Mixer(self.fs)
        self.mix.set_offset(-self.center)
        ntaps = taps_for_transition(self.fs, max(half, 10.0))
        self.filt = StreamFIR(lowpass_taps(half, self.fs, ntaps), decim=self.M)

    def process(self, x):
        y = self.mix.process(np.asarray(x, dtype=np.complex64))
        return self.filt.process(y)


def make_decoder(kind, fs, center, emit, mark_high=True):
    """Decodeurs externes hfdigi non fournis : aucun pour l'instant."""
    return None
