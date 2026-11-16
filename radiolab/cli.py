"""Démodule un fichier I/Q sans interface graphique.

    python -m radiolab.cli enregistrement_IQ.wav  (dans PyPhantom SDR : python -m pyphantom) --freq 7100000 --mode LSB --nr -o audio.wav

Utile pour essayer les traitements anti-bruit sur un même enregistrement, avec
et sans, et comparer à l'oreille.
"""
import argparse
import time
import numpy as np
from .dsp.receiver import Receiver, MODES
from .sources.iqfile import IQFileSource
from .audio import WavWriter


def main(argv=None):
    p = argparse.ArgumentParser(description="PyPhantom SDR : démodulation d'un fichier I/Q vers un fichier audio")
    p.add_argument("iq_file")
    p.add_argument("-o", "--out", default="audio.wav")
    p.add_argument("--freq", type=float, help="fréquence à écouter, en Hz")
    p.add_argument("--offset", type=float, default=0.0, help="ou : écart par rapport au centre, en Hz")
    p.add_argument("--mode", default="USB", choices=sorted(MODES))
    p.add_argument("--nb", action="store_true", help="noise blanker")
    p.add_argument("--nr", action="store_true", help="réduction de bruit")
    p.add_argument("--nr-method", default="lsa", choices=["lsa", "wiener", "specsub", "lms"])
    p.add_argument("--nr-level", type=float, default=18.0, help="réduction maximale en dB")
    p.add_argument("--anf", action="store_true", help="notch automatique")
    p.add_argument("--rate", type=float, default=None,
                   help="débit (S/s) d'un fichier .cf32, qui n'a pas d'en-tête (défaut 1000000)")
    a = p.parse_args(argv)

    src = IQFileSource(a.iq_file, loop=False, realtime=False, sample_rate=a.rate)
    rx = Receiver(src.sample_rate)
    rx.set_mode(a.mode)
    rx.set_offset(a.freq - src.center_hz if a.freq is not None else a.offset)
    rx.set_nb(a.nb)
    rx.set_nr(a.nr, a.nr_method, a.nr_level)
    rx.set_anf(a.anf)
    out = WavWriter(a.out, rx.audio_rate)
    src.rewind()
    t0, n_in = time.perf_counter(), 0
    while True:
        blk = src.read_block()
        if blk is None:
            break
        n_in += len(blk)
        out.write(rx.process(blk))
    out.close()
    dur = n_in / src.sample_rate
    print("%s : %.1f s à %d Hz, centre %.0f Hz -> %s (%s) en %.1f s de calcul"
          % (a.iq_file, dur, src.sample_rate, src.center_hz, a.out, a.mode, time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
