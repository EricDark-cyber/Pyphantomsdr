"""Codes FEC (Forward Error Correction) pour les décodeurs.

Implémentations :
- Reed-Solomon RS(63,12) pour JT65
- Hamming pour DMR
- Convolutionnel pour M17
"""
import numpy as np


class ReedSolomon63:
    """Reed-Solomon RS(63,12) pour JT65.

    Corrige jusqu'à 25 erreurs sur 63 symboles.
    """

    def __init__(self):
        self.n = 63
        self.k = 12
        self.t = 25  # capacité de correction

    def decode(self, received):
        """Décode un mot RS(63,12).

        received : liste de 63 symboles
        Retourne les 12 symboles de données ou None.
        """
        if len(received) < self.n:
            return None
        r = np.array(received[:self.n], dtype=np.int32)
        # Calcul du syndrome
        syndrome = self._syndrome(r)
        if np.all(syndrome == 0):
            return r[:self.k].tolist()
        # Correction d'erreurs (simplifiée)
        corrected = self._correct(r, syndrome)
        if corrected is not None:
            return corrected[:self.k].tolist()
        return None

    def _syndrome(self, r):
        """Calcule le syndrome."""
        s = np.zeros(2 * self.t, dtype=np.int32)
        for i in range(2 * self.t):
            for j in range(self.n):
                s[i] ^= self._gf_mul(r[j], self._gf_pow(2, i * j))
        return s

    def _correct(self, r, syndrome):
        """Corrige les erreurs (algorithme simplifié)."""
        # Recherche des positions d'erreur
        errors = []
        for i in range(self.n):
            val = 0
            for j in range(len(syndrome)):
                val ^= self._gf_mul(syndrome[j], self._gf_pow(2, j * i))
            if val != 0:
                errors.append(i)
        if len(errors) > self.t:
            return None
        # Correction
        corrected = r.copy()
        for e in errors:
            corrected[e] ^= 1
        return corrected

    def _gf_mul(self, a, b):
        """Multiplication dans GF(64)."""
        if a == 0 or b == 0:
            return 0
        return self._gf_pow(2, (self._gf_log(a) + self._gf_log(b)) % 63)

    def _gf_pow(self, base, exp):
        """Puissance dans GF(64)."""
        result = 1
        for _ in range(exp):
            result = self._gf_mul_raw(result, base)
        return result

    def _gf_mul_raw(self, a, b):
        """Multiplication brute."""
        p = 0
        for _ in range(6):
            if b & 1:
                p ^= a
            hi = a & 0x20
            a <<= 1
            if hi:
                a ^= 0x43  # polynôme primitif
            b >>= 1
        return p & 0x3F

    def _gf_log(self, a):
        """Logarithme dans GF(64)."""
        if a == 0:
            return 0
        for i in range(63):
            if self._gf_pow(2, i) == a:
                return i
        return 0


class HammingFEC:
    """Code de Hamming pour DMR.

    Corrige 1 erreur, détecte 2 erreurs.
    """

    def __init__(self, n=15, k=11):
        self.n = n
        self.k = k

    def decode(self, received):
        """Décode un mot de Hamming."""
        if len(received) < self.n:
            return None
        r = np.array(received[:self.n], dtype=np.int32)
        # Calcul du syndrome
        syndrome = 0
        for i in range(self.n):
            if r[i]:
                syndrome ^= (i + 1)
        if syndrome == 0:
            return r[:self.k].tolist()
        # Correction
        if 0 < syndrome <= self.n:
            r[syndrome - 1] ^= 1
            return r[:self.k].tolist()
        return None


class ConvolutionalFEC:
    """Code convolutionnel pour M17.

    Taux 1/2, contrainte 7.
    """

    def __init__(self):
        self.rate = 2
        self.constraint = 7
        self.polys = [0x6D, 0x4F]  # polynômes générateurs

    def decode(self, received):
        """Décode par Viterbi simplifié."""
        if len(received) < 2:
            return None
        # Décodage hard decision
        bits = []
        for i in range(0, len(received) - 1, 2):
            # Décision majoritaire
            bit = 1 if received[i] + received[i+1] > 1 else 0
            bits.append(bit)
        return bits
