"""Champs des messages Mode S selon leur format (DF) : altitude et code transpondeur.

Position des champs (bits numérotés à partir de 1) : DF 1-5 ; pour les formats 0, 4, 5, 16, 20
et 21, le champ de 13 bits (altitude AC ou identité ID) occupe les bits 20 à 32.
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


def decode_ac13(ac):
    """Champ altitude de 13 bits -> pieds, ou None (pas d'altitude, mètres, ou ancien codage par 100 pieds)."""
    if ac == 0 or (ac >> 6) & 1 or not (ac >> 4) & 1:        # bit M (mètres) ou bit Q absent
        return None
    n = ((ac & 0x1F80) >> 2) | ((ac & 0x20) >> 1) | (ac & 0xF)
    return n * 25 - 1000


def decode_id13(idf):
    """Champ identité de 13 bits -> code transpondeur de 4 chiffres (0 à 7).
    Les bits sont entrelacés : C1 A1 C2 A2 C4 A4 X B1 D1 B2 D2 B4 D4."""
    a = ((idf >> 7) & 1) * 4 + ((idf >> 9) & 1) * 2 + ((idf >> 11) & 1)
    b = ((idf >> 1) & 1) * 4 + ((idf >> 3) & 1) * 2 + ((idf >> 5) & 1)
    c = ((idf >> 8) & 1) * 4 + ((idf >> 10) & 1) * 2 + ((idf >> 12) & 1)
    d = (idf & 1) * 4 + ((idf >> 2) & 1) * 2 + ((idf >> 4) & 1)
    return "%d%d%d%d" % (a, b, c, d)


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
    if bits < 32:
        return msg
    field = (data >> (bits - 32)) & 0x1FFF                   # bits 20 à 32
    if df in (0, 4, 16, 20):
        msg["altitude"] = decode_ac13(field)
    if df in (5, 21):
        msg["squawk"] = decode_id13(field)
    if df == 11:
        msg["ca"] = (data >> (bits - 8)) & 0x7
    return msg


def _extract_altitude(data, bits):
    return decode_ac13((data >> (bits - 32)) & 0x1FFF) if bits >= 32 else None


def _extract_squawk(data, bits):
    return decode_id13((data >> (bits - 32)) & 0x1FFF) if bits >= 32 else None


class ModeSDecoder:
    """Compte et garde les messages Mode S, tous formats."""

    def __init__(self):
        self.messages = collections.deque(maxlen=500)
        self.by_df = collections.defaultdict(int)

    def process(self, msg):
        msg = decode_mode_s(msg)
        self.messages.append(msg)
        df = msg.get("df")
        if df is not None:
            self.by_df[df] += 1
        return msg

    def stats(self):
        return dict(self.by_df)
