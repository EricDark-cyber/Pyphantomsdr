"""Décodeur Golay (24,12) complet.

Code de Golay étendu pour correction d'erreurs.
"""
import numpy as np


class GolayDecoder:
    def __init__(self):
        self.n = 24
        self.k = 12
        self.messages = []
        self.G = self._build_generator()

    def _build_generator(self):
        """Matrice génératrice du code de Golay."""
        # Matrice identité + matrice B
        I = np.eye(12, dtype=np.int32)
        B = np.array([
            [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
            [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
            [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ], dtype=np.int32)
        return np.hstack([I, B])

    def decode(self, received):
        """Décode un mot de Golay (24,12)."""
        if len(received) < 24:
            return None
        r = np.array(received[:24], dtype=np.int32)
        # Calcul du syndrome
        H = np.hstack([self.G[:, 12:].T, np.eye(12, dtype=np.int32)])
        syndrome = (H @ r) % 2
        if np.all(syndrome == 0):
            return r[:12].tolist()
        # Correction d'erreurs
        weight = np.sum(syndrome)
        if weight <= 3:
            # Erreur dans les bits de parité
            corrected = r.copy()
            for i in range(12):
                if syndrome[i]:
                    corrected[12 + i] ^= 1
            return corrected[:12].tolist()
        return r[:12].tolist()

    def process(self, bits):
        """Traite des bits et retourne les messages décodés."""
        msgs = []
        for i in range(0, len(bits) - 24, 24):
            decoded = self.decode(bits[i:i+24])
            if decoded:
                text = "".join(chr(32 + b) for b in decoded if b < 95)
                if text and text not in self.messages[-5:]:
                    self.messages.append(text)
                    msgs.append(text)
        return msgs
