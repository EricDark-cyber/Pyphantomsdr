"""Décodage et formatage des messages METAR / SPECI.

Les rapports METAR sont souvent transmis en texte clair sur HF (RTTY, PSK, CW…)
ou via d'autres modes. Ce module extrait les messages valides du flux de texte
et les formate de façon lisible.
"""
import re
import datetime


# Motifs courants dans un METAR
_METAR_RE = re.compile(
    r'\b(METAR|SPECI)\s+([A-Z]{4})\s+(\d{6}Z)\s+'
    r'([A-Z0-9/ ]+?)(?=\s+(?:METAR|SPECI|RMK|=)|$)',
    re.IGNORECASE
)

_WIND_RE = re.compile(r'(\d{3}|VRB)(\d{2,3})(G(\d{2,3}))?(KT|MPS)')
_VIS_RE = re.compile(r'(\d{4}|CAVOK|\d+SM)')
_TEMP_RE = re.compile(r'(M?\d{2})/(M?\d{2})')
_QNH_RE = re.compile(r'(Q|A)(\d{4})')


def parse_metar(text):
    """Extrait et analyse un message METAR/SPECI.

    Retourne un dict avec les champs principaux, ou None si ce n'est pas un METAR.
    """
    text = text.strip().upper()
    m = _METAR_RE.search(text)
    if not m:
        return None

    kind, icao, time_z, body = m.group(1), m.group(2), m.group(3), m.group(4).strip()

    result = {
        "kind": kind,
        "icao": icao,
        "time": time_z,
        "raw": text[:200],
        "wind": None,
        "visibility": None,
        "temperature": None,
        "dewpoint": None,
        "qnh": None,
        "weather": [],
    }

    wm = _WIND_RE.search(body)
    if wm:
        direction = wm.group(1)
        speed = wm.group(2)
        gust = wm.group(4)
        unit = wm.group(5)
        result["wind"] = f"{direction}° {speed}{unit}" + (f" G{gust}" if gust else "")

    vm = _VIS_RE.search(body)
    if vm:
        result["visibility"] = vm.group(1)

    tm = _TEMP_RE.search(body)
    if tm:
        t = tm.group(1).replace("M", "-")
        d = tm.group(2).replace("M", "-")
        result["temperature"] = t + "°C"
        result["dewpoint"] = d + "°C"

    qm = _QNH_RE.search(body)
    if qm:
        val = int(qm.group(2))
        if qm.group(1) == "Q":
            result["qnh"] = f"{val} hPa"
        else:
            result["qnh"] = f"{val/100:.2f} inHg"

    # Phénomènes simples
    for wx in ("RA", "SN", "FG", "BR", "HZ", "TS", "SH", "DZ", "GR"):
        if wx in body:
            result["weather"].append(wx)

    return result


def format_metar(parsed):
    """Formate un METAR analysé en texte lisible."""
    if not parsed:
        return ""
    lines = [
        f"{parsed['kind']} {parsed['icao']}  {parsed['time']}",
        f"  Vent : {parsed['wind'] or '—'}",
        f"  Visibilité : {parsed['visibility'] or '—'}",
        f"  Température : {parsed['temperature'] or '—'}  /  Point de rosée : {parsed['dewpoint'] or '—'}",
        f"  QNH : {parsed['qnh'] or '—'}",
    ]
    if parsed["weather"]:
        lines.append("  Phénomènes : " + ", ".join(parsed["weather"]))
    return "\n".join(lines)


class MetarCollector:
    """Collecte les METAR à partir d'un flux de texte décodé."""

    def __init__(self, maxlen=50):
        self.messages = []
        self.maxlen = maxlen
        self._buffer = ""

    def feed(self, text):
        """Ajoute du texte décodé et extrait les nouveaux METAR."""
        self._buffer += text.upper()
        # Garder un buffer raisonnable
        if len(self._buffer) > 2000:
            self._buffer = self._buffer[-1500:]

        found = []
        for m in _METAR_RE.finditer(self._buffer):
            raw = m.group(0)
            parsed = parse_metar(raw)
            if parsed and parsed not in self.messages[-5:]:  # éviter les doublons proches
                self.messages.append(parsed)
                found.append(parsed)
                if len(self.messages) > self.maxlen:
                    self.messages.pop(0)
        return found

    def clear(self):
        self.messages.clear()
        self._buffer = ""
