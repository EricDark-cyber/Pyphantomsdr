"""Dépaquetage FT8 selon la norme WSJT-X.

Convertit 77 bits en message lisible (callsign, grid, report).
"""

# Alphabet pour les indicatifs (base 37)
CALLSIGN_CHARS = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/"


def unpack_callsign(n):
    """Décode un indicatif depuis un entier (28 bits)."""
    if n >= 37**6:
        return ""
    s = ""
    for _ in range(6):
        s = CALLSIGN_CHARS[n % 37] + s
        n //= 37
    return s.strip()


def unpack_grid(n):
    """Décode un locator depuis un entier (15 bits)."""
    if n >= 180 * 180:
        return ""
    lat = n // 180
    lon = n % 180
    field_lat = chr(ord('A') + lat // 10)
    field_lon = chr(ord('A') + lon // 10)
    sq_lat = str(lat % 10)
    sq_lon = str(lon % 10)
    return f"{field_lon}{field_lat}{sq_lon}{sq_lat}"


def unpack_ft8(bits):
    """Dépaquete un message FT8 de 77 bits.

    Retourne une chaîne lisible ou None.
    """
    if len(bits) < 77:
        return None

    # Extraire les champs
    val = 0
    for b in bits[:77]:
        val = (val << 1) | b

    # Type de message (3 bits de poids fort)
    msg_type = (val >> 74) & 0x7

    if msg_type == 0:
        # CQ + callsign + grid
        call_n = (val >> 46) & 0xFFFFFFF
        grid_n = (val >> 31) & 0x7FFF
        call = unpack_callsign(call_n)
        grid = unpack_grid(grid_n)
        if call:
            return f"CQ {call} {grid}"

    elif msg_type == 1:
        # Callsign + callsign + report
        call1_n = (val >> 46) & 0xFFFFFFF
        call2_n = (val >> 18) & 0xFFFFFFF
        report = (val >> 11) & 0x7F
        c1 = unpack_callsign(call1_n)
        c2 = unpack_callsign(call2_n)
        if c1 and c2:
            snr = report - 35  # offset
            return f"{c1} {c2} {snr:+d}"

    elif msg_type == 2:
        # Callsign + callsign + R + report
        call1_n = (val >> 46) & 0xFFFFFFF
        call2_n = (val >> 18) & 0xFFFFFFF
        report = (val >> 11) & 0x7F
        c1 = unpack_callsign(call1_n)
        c2 = unpack_callsign(call2_n)
        if c1 and c2:
            snr = report - 35
            return f"{c1} {c2} R{snr:+d}"

    elif msg_type == 3:
        # Callsign + callsign + RRR
        call1_n = (val >> 46) & 0xFFFFFFF
        call2_n = (val >> 18) & 0xFFFFFFF
        c1 = unpack_callsign(call1_n)
        c2 = unpack_callsign(call2_n)
        if c1 and c2:
            return f"{c1} {c2} RRR"

    elif msg_type == 4:
        # Callsign + callsign + 73
        call1_n = (val >> 46) & 0xFFFFFFF
        call2_n = (val >> 18) & 0xFFFFFFF
        c1 = unpack_callsign(call1_n)
        c2 = unpack_callsign(call2_n)
        if c1 and c2:
            return f"{c1} {c2} 73"

    return None
