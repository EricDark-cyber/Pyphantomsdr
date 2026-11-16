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
