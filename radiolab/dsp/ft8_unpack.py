"""Dépaquetage des messages FT8 / FT4 de 77 bits (formats de WSJT-X, packjt77).

Les 3 derniers bits (i3) donnent le type du message :

  i3 = 1, 2  message standard :  n28a(28) ipa(1) n28b(28) ipb(1) ir(1) igrid4(15) i3(3)
             deux indicatifs (ou CQ, DE, QRZ), puis un carré locator ou un report.
             ipa / ipb ajoutent « /R » (i3 = 1) ou « /P » (i3 = 2) à l'indicatif.
  i3 = 4     indicatif non standard : n12(12) n58(58) iflip(1) nrpt(2) icq(1) i3(3)
             un indicatif en clair (11 caractères), l'autre remplacé par un condensé de 12 bits.
  i3 = 0     selon n3 (les 3 bits précédents) : 0 = texte libre de 13 caractères,
             1 = expédition (deux correspondants à la fois), 5 = télémesure (18 chiffres hexa).

Un indicatif trop long pour tenir en 28 bits est remplacé à l'émission par un « condensé »
(hash) de 22, 12 ou 10 bits.  Le récepteur ne peut le traduire que s'il a déjà entendu cet
indicatif en clair : on retient donc tous les indicatifs décodés.  Sinon on affiche <...>,
comme WSJT-X.

Les types plus rares (concours RTTY, Field Day, concours VHF européen) ne sont pas traduits :
unpack_ft8 retourne alors None.
"""

A1 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
A2 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
A3 = "0123456789"
A4 = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
A38 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/"                # indicatifs non standard et condensés
A42 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-./?"            # texte libre
NTOKENS = 2063592
MAX22 = 4194304
MAXGRID4 = 32400

# indicatifs déjà entendus, retrouvés par leur condensé de 22, 12 ou 10 bits
_HASH22, _HASH12, _HASH10 = {}, {}, {}


def hash_call(call):
    """Condensé de 22 bits d'un indicatif (les 12 et 10 bits en sont les bits de tête)."""
    n = 0
    for ch in (call.upper().strip() + " " * 11)[:11]:
        j = A38.find(ch)
        if j < 0:
            return None
        n = 38 * n + j
    return ((47055833459 * n) & 0xFFFFFFFFFFFFFFFF) >> 42


def remember(call):
    """Retient un indicatif entendu en clair, pour traduire plus tard ses condensés."""
    call = call.strip().strip("<>")
    if len(call) < 3 or call.startswith("CQ") or call in ("DE", "QRZ"):
        return
    h = hash_call(call)
    if h is None:
        return
    if len(_HASH22) > 5000:
        _HASH22.clear(); _HASH12.clear(); _HASH10.clear()
    _HASH22[h] = call
    _HASH12[h >> 10] = call
    _HASH10[h >> 12] = call


def forget_calls():
    _HASH22.clear(); _HASH12.clear(); _HASH10.clear()


def _hashed(table, h):
    call = table.get(h)
    return "<%s>" % call if call else "<...>"


def unpack28(n, ip=0, i3=1):
    """28 bits -> indicatif, « CQ », « CQ DX », « <...> » pour un condensé inconnu ; None si invalide."""
    n = int(n)
    if n < NTOKENS:
        if n == 0:
            return "DE"
        if n == 1:
            return "QRZ"
        if n == 2:
            return "CQ"
        if n <= 1002:
            return "CQ %03d" % (n - 3)
        if n <= 532443:
            v = n - 1003
            i1, v = divmod(v, 27 ** 3)
            i2, v = divmod(v, 27 ** 2)
            i3_, i4 = divmod(v, 27)
            word = (A4[i1] + A4[i2] + A4[i3_] + A4[i4]).strip()
            return "CQ " + word if word and " " not in word else None
        return None
    if n < NTOKENS + MAX22:
        return _hashed(_HASH22, n - NTOKENS)
    v = n - NTOKENS - MAX22
    i1, v = divmod(v, 36 * 10 * 27 ** 3)
    i2, v = divmod(v, 10 * 27 ** 3)
    i3_, v = divmod(v, 27 ** 3)
    i4, v = divmod(v, 27 ** 2)
    i5, i6 = divmod(v, 27)
    if i1 >= len(A1):
        return None
    call = (A1[i1] + A2[i2] + A3[i3_] + A4[i4] + A4[i5] + A4[i6]).strip()
    if len(call) < 3 or " " in call:
        return None
    remember(call)
    if ip:
        call += "/R" if i3 == 1 else "/P"
        remember(call)
    return call


def pack28(call):
    """Indicatif standard -> 28 bits (sert aux essais). « CQ » -> 2."""
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
        callsign = (" " + c + "     ")[:6]
    elif iarea == 3:
        callsign = (c + "     ")[:6]
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
    """15 bits -> carré locator de 4 caractères, par exemple 17518 -> « JN18 »."""
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
    """Carré locator de 4 caractères -> 15 bits (inverse de to_grid4)."""
    g = g.upper().strip()
    return (ord(g[0]) - 65) * 1800 + (ord(g[1]) - 65) * 100 + int(g[2]) * 10 + int(g[3])


def _report(isnr):
    return "%+03d" % isnr                                     # « -06 », « +12 », comme WSJT-X


def _standard(val, i3):
    n28a = (val >> 49) & 0xFFFFFFF
    ipa = (val >> 48) & 1
    n28b = (val >> 20) & 0xFFFFFFF
    ipb = (val >> 19) & 1
    ir = (val >> 18) & 1
    igrid4 = (val >> 3) & 0x7FFF
    call1 = unpack28(n28a, ipa, i3)
    call2 = unpack28(n28b, ipb, i3)
    if not call1 or not call2:
        return None
    if call2.startswith("CQ") or call2 in ("DE", "QRZ"):      # le second champ est toujours un indicatif
        return None
    is_cq = call1.startswith("CQ")
    if igrid4 <= MAXGRID4:
        grid = to_grid4(igrid4)
        if grid is None or (is_cq and ir):
            return None
        return "%s %s%s%s" % (call1, call2, " R " if ir else " ", grid)
    irpt = igrid4 - MAXGRID4
    if is_cq and irpt >= 2:
        return None
    if irpt == 1:
        return "%s %s" % (call1, call2)
    if irpt in (2, 3, 4):
        return "%s %s %s" % (call1, call2, {2: "RRR", 3: "RR73", 4: "73"}[irpt])
    isnr = irpt - 35
    if isnr > 50:
        isnr -= 101
    return "%s %s %s%s" % (call1, call2, "R" if ir else "", _report(isnr))


def _nonstandard(val):
    n12 = (val >> 65) & 0xFFF
    n58 = (val >> 7) & ((1 << 58) - 1)
    iflip = (val >> 6) & 1
    nrpt = (val >> 4) & 3
    icq = (val >> 3) & 1
    chars = []
    for _ in range(11):
        n58, j = divmod(n58, 38)
        chars.append(A38[j])
    call = "".join(reversed(chars)).strip()
    if len(call) < 3 or " " in call:
        return None
    remember(call)
    if icq:
        return "CQ " + call
    other = _hashed(_HASH12, n12)
    first, second = (call, other) if iflip else (other, call)
    return ("%s %s %s" % (first, second, ("", "RRR", "RR73", "73")[nrpt])).strip()


def _free_text(val):
    n = val >> 6                                              # 71 bits
    chars = []
    for _ in range(13):
        n, j = divmod(n, 42)
        chars.append(A42[j])
    return "".join(reversed(chars)).strip() or None


def _dxpedition(val):
    n28a = (val >> 49) & 0xFFFFFFF
    n28b = (val >> 21) & 0xFFFFFFF
    n10 = (val >> 11) & 0x3FF
    n5 = (val >> 6) & 0x1F
    call1, call2 = unpack28(n28a), unpack28(n28b)
    if not call1 or not call2 or call1.startswith("CQ") or call2.startswith("CQ"):
        return None
    return "%s RR73; %s %s %s" % (call1, call2, _hashed(_HASH10, n10), _report(2 * n5 - 30))


def unpack_ft8(bits):
    """77 bits (poids fort d'abord) -> texte du message, ou None si le type n'est pas traduit."""
    if len(bits) < 77:
        return None
    val = 0
    for b in bits[:77]:
        val = (val << 1) | int(b)
    i3 = val & 7
    if i3 in (1, 2):
        return _standard(val, i3)
    if i3 == 4:
        return _nonstandard(val)
    if i3 == 0:
        n3 = (val >> 3) & 7
        if n3 == 0:
            return _free_text(val)
        if n3 == 1:
            return _dxpedition(val)
        if n3 == 5:
            return "%018X" % (val >> 6)
    return None


def pack_standard(call1, call2, ir=0, grid=None, report=None, i3=1):
    """Construit les 77 bits d'un message standard (sert aux essais)."""
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
