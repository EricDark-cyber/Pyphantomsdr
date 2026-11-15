"""CRC-24 pour ADS-B / Mode S (polynôme 0xFFF409)."""


def crc24(data):
    """Calcule le CRC-24 sur des données (bytes ou liste de bits).

    Polynôme : 0xFFF409 (Mode S)
    """
    poly = 0xFFF409
    if isinstance(data, (bytes, bytearray)):
        bits = []
        for b in data:
            for i in range(7, -1, -1):
                bits.append((b >> i) & 1)
    else:
        bits = list(data)

    crc = 0
    for b in bits:
        crc ^= (b << 23)
        if crc & 0x800000:                      # un decalage par bit (CRC MSB-first)
            crc = ((crc << 1) ^ poly) & 0xFFFFFF
        else:
            crc = (crc << 1) & 0xFFFFFF
    return crc


def check_crc24(data, received_crc):
    """Vérifie le CRC-24."""
    return crc24(data) == received_crc


_TABLE = []
for _i in range(256):
    _c = _i << 16
    for _ in range(8):
        _c = ((_c << 1) ^ 0xFFF409) & 0xFFFFFF if _c & 0x800000 else (_c << 1) & 0xFFFFFF
    _TABLE.append(_c)
del _i, _c


def crc24_bytes(data):
    """Reste de la division du message entier (octets) par le polynôme Mode S, calcul rapide par table.
    Pour un message ADS-B valide (contrôle compris), le reste est nul."""
    crc = 0
    for b in data:
        crc = ((crc << 8) & 0xFFFFFF) ^ _TABLE[((crc >> 16) ^ b) & 0xFF]
    return crc
