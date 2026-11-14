"""Analyse des messages SIGMET (Significant Meteorological Information).

Les SIGMET sont des avertissements météorologiques aéronautiques (orages, turbulence,
givrage, cendres volcaniques, etc.). Ils sont souvent transmis en texte clair sur HF.
"""
import re


_SIGMET_RE = re.compile(
    r'\b(SIGMET|AIRMET)\s+([A-Z0-9]{1,4})\s+VALID\s+(\d{6}/\d{6})\s+'
    r'([A-Z]{4})\s*-?\s*(.*?)(?=\s+(?:SIGMET|AIRMET|METAR|TAF|=)|$)',
    re.IGNORECASE | re.DOTALL
)

_HAZARD_KEYWORDS = {
    "TS": "Orages",
    "TURB": "Turbulence",
    "ICE": "Givrage",
    "MTW": "Ondes orographiques",
    "VA": "Cendres volcaniques",
    "TC": "Cyclone tropical",
    "DS": "Tempête de poussière",
    "SS": "Tempête de sable",
    "SQL": "Ligne de grains",
    "SEV": "Sévère",
    "EMBD": "Encastré",
    "OBSC": "Obscurci",
    "FRQ": "Fréquent",
    "ISOL": "Isolé",
}


def parse_sigmet(text):
    """Analyse un message SIGMET/AIRMET.

    Retourne un dict avec les informations principales, ou None.
    """
    text = text.strip().upper()
    m = _SIGMET_RE.search(text)
    if not m:
        return None

    kind, number, valid, fir, body = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5).strip()

    hazards = []
    for code, label in _HAZARD_KEYWORDS.items():
        if code in body:
            hazards.append(label)

    # Niveau de vol approximatif
    fl = None
    fl_m = re.search(r'FL(\d{3})', body)
    if fl_m:
        fl = f"FL{fl_m.group(1)}"

    return {
        "kind": kind,
        "number": number,
        "valid": valid,
        "fir": fir,
        "hazards": hazards,
        "flight_level": fl,
        "raw": text[:300],
        "body": body[:200],
    }


def format_sigmet(parsed):
    """Formate un SIGMET analysé."""
    if not parsed:
        return ""
    lines = [
        f"{parsed['kind']} {parsed['number']}  VALID {parsed['valid']}",
        f"  FIR : {parsed['fir']}",
        f"  Phénomènes : {', '.join(parsed['hazards']) or '—'}",
        f"  Niveau : {parsed['flight_level'] or '—'}",
        f"  Détail : {parsed['body']}",
    ]
    return "\n".join(lines)


class SigmetCollector:
    """Collecte les SIGMET à partir d'un flux de texte."""

    def __init__(self, maxlen=30):
        self.messages = []
        self.maxlen = maxlen
        self._buffer = ""

    def feed(self, text):
        self._buffer += text.upper()
        if len(self._buffer) > 3000:
            self._buffer = self._buffer[-2000:]

        found = []
        for m in _SIGMET_RE.finditer(self._buffer):
            raw = m.group(0)
            parsed = parse_sigmet(raw)
            if parsed and parsed not in self.messages[-3:]:
                self.messages.append(parsed)
                found.append(parsed)
                if len(self.messages) > self.maxlen:
                    self.messages.pop(0)
        return found

    def clear(self):
        self.messages.clear()
        self._buffer = ""
