"""Point d'entrée pour fabriquer PyPhantom.exe (voir fabriquer_exe.bat).

Pour un lancement normal depuis les sources :  python -m pyphantom
"""
import sys

from pyphantom.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
