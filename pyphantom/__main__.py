"""Point d'entree de PyPhantom SDR : lance l'interface (port complet de RadioLab 0.15).

Lancer depuis la racine du depot :  python -m pyphantom
Le recepteur Airspy R2 est pilote en ctypes sur airspy.dll (fourni a la racine du depot,
avec libusb-1.0.dll et pthreadVC2.dll) : aucune installation de PothosSDR/SoapySDR n'est
requise. Un fichier I/Q peut etre ouvert depuis le menu (ou en ligne de commande :
python -m radiolab.cli fichier.wav --freq 7100000 --mode USB -o audio.wav).
"""
import sys


def main() -> int:
    from radiolab.gui import main as radiolab_main

    return radiolab_main()


if __name__ == "__main__":
    sys.exit(main())
