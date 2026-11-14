"""Génère demo.cf32 : 10 s à 1 MS/s avec stations AM / NFM / USB synthétiques.
Usage : python tools/make_demo_iq.py [sortie.cf32] [duree_s] [rate]
Puis testez l'appli sans récepteur : python -m pyphantom --iq demo.cf32 --rate 1000000
"""
from __future__ import annotations

import sys

import numpy as np


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "demo.cf32"
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    fs = float(sys.argv[3]) if len(sys.argv) > 3 else 1_000_000.0
    n = int(fs * dur)
    t = np.arange(n) / fs
    rng = np.random.default_rng(42)
    iq = 0.02 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))

    am = (1.0 + 0.6 * np.cos(2 * np.pi * 1000 * t)) * np.exp(2j * np.pi * (-150_000) * t)
    iq += 0.3 * am

    inst = 5000.0 * np.cos(2 * np.pi * 440 * t)
    phase = 2 * np.pi * np.cumsum(inst) / fs + 2 * np.pi * 200_000 * t
    iq += 0.3 * np.exp(1j * phase)

    usb = (np.cos(2 * np.pi * 700 * t) + 0.5 * np.cos(2 * np.pi * 1400 * t)) * np.exp(2j * np.pi * 60_000 * t)
    iq += 0.2 * usb

    iq = (iq / np.max(np.abs(iq)) * 0.8).astype(np.complex64)
    iq.tofile(out)
    print(f"{out} : {n} échantillons ({dur:g} s à {fs/1e6:g} MS/s), cf32")


if __name__ == "__main__":
    main()
