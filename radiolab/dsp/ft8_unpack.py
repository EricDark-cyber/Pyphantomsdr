"""Depaquetage des messages FT8 selon WSJT-X (packjt77.f90 / unpk77).

Message standard (i3 = 1 ou 2), 77 bits MSB d'abord :
  bits  0-27 : n28a (indicatif 1, pack28)
  bit  28    : ipa (/R)
  bits 29-56 : n28b (indicatif 2)
  bit  57    : ipb (/P)
  bit  58    : ir (R)
  bits 59-73 : igrid4 (locator 4 caracteres, ou report si > 32400)
  bits 74-76 : i3 (type de message)
"""

A1 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
A2 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
A3 = "0123456789"
A4 = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NTOKENS = 2063592
MAX22 = 4194304
MAXGRID4 = 32400


def unpack28(n):
    """Decode un indicatif 28 bits (unpack28 WSJT-X). None si hash/non standard."""
    n = int(n)
    if n < NTOKENS:
        if n == 0:
            return "DE"
        if n == 1:
            return "QRZ"
        if n == 2:
            return "CQ"
        if n <= 1002:
            return f"CQ_{n - 3:03d}"
        if n <= 532443:
            v = n - 1003
            i1, v = divmod(v, 27 ** 3)
            i2, v = divmod(v, 27 ** 2)
            i3, i4 = divmod(v, 27)
            return "CQ_" + (A4[i1] + A4[i2] + A4[i3] + A4[i4]).strip()
        return None
    if n < NTOKENS + MAX22:
        return None
    v = n - NTOKENS - MAX22
    i1, v = divmod(v, 36 * 10 * 27 ** 3)
    i2, v = divmod(v, 10 * 27 ** 3)
    i3, v = divmod(v, 27 ** 3)
    i4, v = divmod(v, 27 ** 2)
    i5, i6 = divmod(v, 27)
    call = A1[i1] + A2[i2] + A3[i3] + A4[i4] + A4[i5] + A4[i6]
    return call.strip() or None


def pack28(call):
    """Encode un indicatif standard (pack28 WSJT-X). 'CQ' -> token 2."""
    c = call.upper().strip()
    if c == "CQ":
        return 2
    if c == "DE":
        return 0
    if c == "QRZ":
        return 1
    n = len(c)
    iarea = 0
    for i in range(n, 1, -1):
        if c[i - 1].isdigit():
            iarea = i
            break
    if iarea == 2:
        callsign = " " + c[:5]
    elif iarea == 3:
        callsign = c[:6]
    else:
        raise ValueError(f"indicatif non standard: {call}")
    i1 = A1.index(callsign[0])
    i2 = A2.index(callsign[1])
    i3 = A3.index(callsign[2])
    i4 = A4.index(callsign[3])
    i5 = A4.index(callsign[4])
    i6 = A4.index(callsign[5])
    return NTOKENS + MAX22 + 36 * 10 * 27 ** 3 * i1 + 10 * 27 ** 3 * i2 + 27 ** 3 * i3 + 27 ** 2 * i4 + 27 * i5 + i6


def to_grid4(n):
    """15 bits -> locator 4 caracteres (to_grid4 WSJT-X), ex. 17518 -> 'JN18'."""
    n = int(n)
    if not 0 <= n <= MAXGRID4:
        return None
    j1, r = divmod(n, 1800)
    j2, r = divmod(r, 100)
    j3, j4 = divmod(r, 10)
    if j1 > 17 or j2 > 17:
        return None
    return chr(65 + j1) + chr(65 + j2) + str(j3) + str(j4)


def grid4_to_int(g):
    """Locator 4 caracteres -> 15 bits (inverse de to_grid4)."""
    g = g.upper().strip()
    return (ord(g[0]) - 65) * 1800 + (ord(g[1]) - 65) * 100 + int(g[2]) * 10 + int(g[3])


def unpack_ft8(bits):
    """Depaquete un message FT8 de 77 bits -> texte lisible ou None."""
    if len(bits) < 77:
        return None
    val = 0
    for b in bits[:77]:
        val = (val << 1) | int(b)
    i3 = val & 7
    if i3 not in (1, 2):
        return None
    n28a = (val >> 49) & 0xFFFFFFF
    n28b = (val >> 20) & 0xFFFFFFF
    ir = (val >> 18) & 1
    igrid4 = (val >> 3) & 0x7FFF
    call1 = unpack28(n28a)
    call2 = unpack28(n28b)
    if not call1 or not call2:
        return None
    if igrid4 <= MAXGRID4:
        grid = to_grid4(igrid4)
        if grid is None:
            return None
        rpt = (" R " + grid) if ir else (" " + grid)
    else:
        irpt = igrid4 - MAXGRID4
        if irpt == 1:
            rpt = ""
        elif irpt == 2:
            rpt = " RRR"
        elif irpt == 3:
            rpt = " RR73"
        elif irpt == 4:
            rpt = " 73"
        else:
            isnr = irpt - 35
            if isnr > 50:
                isnr -= 101
            rpt = f" R{isnr:+d}" if ir else f" {isnr:+d}"
    return f"{call1} {call2}{rpt}"


def pack_standard(call1, call2, ir=0, grid=None, report=None, i3=1):
    """Construit les 77 bits d'un message standard (pour tests)."""
    n28a = pack28(call1)
    n28b = pack28(call2)
    if report is not None:
        r = report
        if r < -35:
            r += 101
        igrid4 = MAXGRID4 + 35 + r
    else:
        igrid4 = grid4_to_int(grid or "JN18")
    val = (n28a << 49) | (n28b << 20) | (ir << 18) | (igrid4 << 3) | i3
    return [(val >> (76 - i)) & 1 for i in range(77)]
