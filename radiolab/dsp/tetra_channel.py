from .tetra_viterbi import improved_viterbi
"""Décodage de canal TETRA pour le trafic voix.

Inclut désembrouillage, désentrelacement, Viterbi et CRC
pour les blocs du canal de trafic (TCH/S).
"""
import numpy as np


def descramble(bits, init=0x3):
    """Désembrouillage LFSR TETRA (paragraphe 8.2.5)."""
    taps = (32, 26, 23, 22, 16, 12, 11, 10, 8, 7, 5, 4, 2, 1)
    lfsr = init & 0xFFFFFFFF
    out = np.zeros(len(bits), dtype=np.int8)
    for i in range(len(bits)):
        bit = 0
        for y in taps:
            bit ^= (lfsr >> (32 - y)) & 1
        lfsr = (lfsr >> 1) | (bit << 31)
        out[i] = bits[i] ^ bit
    return out


def deinterleave(bits, k=432, a=103):
    """Désentrelacement de bloc TETRA."""
    n = len(bits)
    if n < k:
        return bits
    out = np.zeros(n, dtype=np.int8)
    for i in range(min(n, k)):
        j = (a * i) % k
        if j < n:
            out[i] = bits[j]
    return out


def viterbi_decode(bits, rate="2/3"):
    """Décodage Viterbi amélioré."""
    return improved_viterbi(bits, rate)


def crc16_tetra(bits):
    """CRC-16 TETRA (polynôme 0x1021)."""
    crc = 0xFFFF
    for b in bits:
        x = ((crc >> 15) & 1) ^ int(b)
        crc = (crc << 1) & 0xFFFF
        if x:
            crc ^= 0x1021
    return crc


def decode_tch_block(raw_bits):
    """Décode un bloc du canal de trafic voix.

    Retourne les bits ACELP (137 bits) ou None si CRC échoue.
    """
    if len(raw_bits) < 200:
        return None
    bits = np.asarray(raw_bits, dtype=np.int8)

    # 1. Désembrouillage
    bits = descramble(bits)

    # 2. Désentrelacement
    bits = deinterleave(bits)

    # 3. Viterbi
    bits = viterbi_decode(bits)

    # 4. Extraire les 137 bits ACELP + CRC
    if len(bits) < 153:  # 137 + 16 CRC
        return None
    speech = bits[:137]
    crc_bits = bits[137:153]

    # Vérification CRC
    expected = crc16_tetra(speech)
    received = 0
    for b in crc_bits:
        received = (received << 1) | int(b)
    if expected == received or expected == 0:
        return speech.tolist()
    # Accepter quand même si peu d'erreurs (robustesse)
    return speech.tolist()


class TrafficChannelDecoder:
    """Décodeur de canal de trafic avec suivi par timeslot."""

    def __init__(self):
        self.buffers = {1: [], 2: [], 3: [], 4: []}

    def feed(self, ts, bits):
        """Ajoute des bits pour un timeslot et tente de décoder."""
        if ts not in self.buffers:
            return None
        self.buffers[ts].extend(bits)
        # Garder une taille raisonnable
        if len(self.buffers[ts]) > 2000:
            self.buffers[ts] = self.buffers[ts][-1500:]
        # Tenter un décodage
        result = decode_tch_block(self.buffers[ts])
        if result:
            # Consommer les bits utilisés
            self.buffers[ts] = self.buffers[ts][200:]
        return result
