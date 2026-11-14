"""Point d'entrée : python -m pyphantom [--iq fichier.cf32] [--rate 1000000]."""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="PyPhantom SDR")
    parser.add_argument("--iq", help="Fichier IQ .cf32 à lire en boucle (test sans récepteur)")
    parser.add_argument("--rate", type=float, default=1_000_000.0, help="Sample rate du fichier IQ (S/s)")
    parser.add_argument("--audio-rate", type=int, default=48_000, help="Sample rate audio (Hz)")
    args = parser.parse_args()

    from PySide6.QtWidgets import QApplication

    from .gui.main_window import MainWindow

    app = QApplication(sys.argv)
    win = MainWindow(iq_file=args.iq, iq_rate=args.rate)
    if args.audio_rate != 48_000:
        win.cfg["audio_rate"] = args.audio_rate
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
