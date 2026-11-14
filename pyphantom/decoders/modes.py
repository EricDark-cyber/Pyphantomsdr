"""Décodage étendu des messages Mode S (tous les principaux DF).

Complète le décodeur ADS-B avec les formats de surveillance et d'identité.
"""
import collections


DF_NAMES = {
    0: "Short Air-Air",
    4: "Surveillance Altitude",
    5: "Surveillance Identity",
    11: "All-Call Reply",
    16: "Long Air-Air",
    17: "ADS-B Extended Squitter",
    18: "ADS-B (non-transponder)",
    20: "Comm-B Altitude",
    21: "Comm-B Identity",
    24: "Comm-D",
}


def decode_mode_s(msg):
    """Enrichit un message Mode S décodé avec les champs selon son DF."""
    df = msg.get("df")
    raw = msg.get("raw", "")
    if not raw or df is None:
        return msg

    try:
        data = int(raw, 16)
    except ValueError:
        return msg

    msg["df_name"] = DF_NAMES.get(df, f"DF{df}")
    bits = msg.get("bits", 56)

    if df in (0, 4, 16, 20):
        # Altitude
        msg["altitude"] = _extract_altitude(data, bits)
    if df in (5, 21):
        # Identity / Squawk
        msg["squawk"] = _extract_squawk(data, bits)
    if df == 11:
        # All-call : adresse ICAO déjà extraite
        msg["ca"] = (data >> (bits - 8)) & 0x7 if bits >= 8 else None
    if df in (17, 18):
        # Déjà géré par decode_adsb_payload
        pass

    return msg


def _extract_altitude(data, bits):
    """Extrait l'altitude des messages de surveillance (AC field)."""
    if bits < 32:
        return None
    # AC field : 13 bits, position dépend du format
    ac = (data >> (bits - 32 + 8)) & 0x1FFF   # approximation courante
    if ac == 0:
        return None
    # Décodage Gillham / simple
    m = ac & 0x7F
    q = (ac >> 4) & 1
    if q:
        n = ((ac & 0x0FE0) >> 1) | (ac & 0x000F)
        return n * 25 - 1000
    return None


def _extract_squawk(data, bits):
    """Extrait le code squawk (Mode A) des messages d'identité."""
    if bits < 32:
        return None
    id_field = (data >> (bits - 32 + 8)) & 0x1FFF
    # Conversion en octal 4 chiffres
    a = (id_field >> 9) & 7
    b = (id_field >> 6) & 7
    c = (id_field >> 3) & 7
    d = id_field & 7
    return f"{a}{b}{c}{d}"


class ModeSDecoder:
    """Décodeur Mode S étendu (tous DF)."""

    def __init__(self):
        self.messages = collections.deque(maxlen=500)
        self.by_df = collections.defaultdict(int)

    def process(self, msg):
        """Enrichit et stocke un message."""
        msg = decode_mode_s(msg)
        self.messages.append(msg)
        df = msg.get("df")
        if df is not None:
            self.by_df[df] += 1
        return msg

    def stats(self):
        return dict(self.by_df)
