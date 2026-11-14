"""Décodeur Contestia complet.

Contestia : variante d'Olivia pour les concours.
"""
import numpy as np
try:
    from .olivia import OliviaDecoder
    HAS_OLIVIA = True
except ImportError:
    OliviaDecoder = object
    HAS_OLIVIA = False  # fichier olivia.py non fourni


class ContestiaDecoder(OliviaDecoder):
    """Contestia utilise la même modulation qu'Olivia."""

    def __init__(self, fs=8000):
        if not HAS_OLIVIA:
            raise RuntimeError("ContestiaDecoder: fichier olivia.py manquant (non fourni)")
        super().__init__(fs=fs, tones=16)
        self.baud = 62.5
        self.samples_per_symbol = int(fs / self.baud)
