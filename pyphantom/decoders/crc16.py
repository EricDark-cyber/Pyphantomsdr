"""CRC-16 pour AIS (polynôme 0x1021, CRC-CCITT)."""


def crc16(data):
    """Calcule le CRC-16 CCITT sur des données."""
    poly = 0x1021
    if isinstance(data, (bytes, bytearray)):
        bits = []
        for b in data:
            for i in range(7, -1, -1):
                bits.append((b >> i) & 1)
    else:
        bits = list(data)

    crc = 0xFFFF
    for b in bits:
        crc ^= (b << 15)
        if crc & 0x8000:                        # un decalage par bit (CRC MSB-first)
            crc = ((crc << 1) ^ poly) & 0xFFFF
        else:
            crc = (crc << 1) & 0xFFFF
    return crc ^ 0xFFFF


def check_crc16(data, received_crc):
    """Vérifie le CRC-16."""
    return crc16(data) == received_crc
