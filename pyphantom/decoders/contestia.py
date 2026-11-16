"""Décodeur Contestia complet.

Contestia : variante d'Olivia pour les concours.
"""
import numpy as np
from .olivia import OliviaDecoder


class ContestiaDecoder(OliviaDecoder):
    """Contestia utilise la même modulation qu'Olivia."""

    def __init__(self, fs=8000):
        super().__init__(fs=fs, tones=16)
        self.baud = 62.5
        self.samples_per_symbol = int(fs / self.baud)
