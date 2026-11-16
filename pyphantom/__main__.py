"""Point d'entrée de PyPhantom SDR : lance l'interface (port complet de RadioLab 0.15).

Lancer depuis la racine du dépôt :  python -m pyphantom [--iq fichier.wav|.cf32] [--rate S/s]
Le récepteur Airspy R2 est piloté en ctypes sur airspy.dll (fourni à la racine du dépôt,
avec libusb-1.0.dll et pthreadVC2.dll) : aucune installation de PothosSDR/SoapySDR
n'est requise. Un fichier I/Q (.cf32 de SDR# ou WAV) peut être ouvert au lancement
avec --iq, ou depuis le menu « Ouvrir un fichier I/Q… ».
"""
import sys


def main() -> int:
    from radiolab.gui import main as radiolab_main

    return radiolab_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
