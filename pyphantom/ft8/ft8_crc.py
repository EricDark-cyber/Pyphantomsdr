"""CRC-14 pour FT8 (polynôme WSJT-X).

Polynôme : x^14 + x^13 + x^10 + x^9 + x^8 + x^6 + x^4 + x^3 + x^2 + x + 1
(0x2757)
"""


def crc14(bits):
    """Calcule le CRC-14 sur une liste de bits.

    bits : liste de 0/1 (77 bits de message)
    Retourne les 14 bits de CRC.
    """
    poly = 0x2757
    crc = 0
    for b in bits:
        crc ^= (b << 13)
        for _ in range(14):
            if crc & 0x2000:
                crc = ((crc << 1) ^ poly) & 0x3FFF
            else:
                crc = (crc << 1) & 0x3FFF
    # Retourne les 14 bits
    return [(crc >> (13 - i)) & 1 for i in range(14)]


def check_crc14(bits):
    """Vérifie le CRC-14 d'un message de 91 bits (77 + 14).

    bits : liste de 91 bits (77 message + 14 CRC)
    Retourne True si le CRC est valide.
    """
    if len(bits) < 91:
        return False
    msg = bits[:77]
    received_crc = bits[77:91]
    computed = crc14(msg)
    return received_crc == computed


def add_crc14(bits):
    """Ajoute le CRC-14 à un message de 77 bits.

    Retourne 91 bits (77 + 14 CRC).
    """
    crc = crc14(bits[:77])
    return list(bits[:77]) + crc
