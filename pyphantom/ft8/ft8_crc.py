"""CRC-14 pour FT8, convention exacte de WSJT-X (chkcrc14a.f90 / get_crc14).

Polynome : x^14 + x^13 + x^10 + x^9 + x^8 + x^6 + x^4 + x^3 + x^2 + x + 1 (0x2757).
WSJT-X calcule le CRC sur les 77 bits du message suivis de 19 bits a zero
(12 octets lus en b8, le 10e octet masque aux 5 bits utiles), soit le reste
de msg * x^19 mod p. Vecteur de test officiel (weakmon/WSJT-X) verifie dans
tests/test_ft8.py.
"""

POLY = 0x2757


def crc14(bits):
    """CRC-14 sur les 77 premiers bits (liste de 0/1) -> 14 bits, MSB d'abord."""
    reg = 0
    for b in list(bits[:77]) + [0] * 19:
        reg = ((reg << 1) | int(b)) & 0x7FFF
        if reg & 0x4000:
            reg ^= POLY
    return [(reg >> (13 - i)) & 1 for i in range(14)]


def check_crc14(bits):
    """Verifie le CRC-14 d'un mot de 91 bits (77 message + 14 CRC)."""
    if len(bits) < 91:
        return False
    return list(bits[77:91]) == crc14(bits[:77])


def add_crc14(bits):
    """Ajoute le CRC-14 a un message de 77 bits -> 91 bits."""
    msg = list(bits[:77])
    return msg + crc14(msg)
