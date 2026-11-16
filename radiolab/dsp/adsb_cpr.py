"""Décodage CPR (Compact Position Reporting) pour ADS-B.

Implémentation basée sur la spécification ADS-B et les algorithmes de dump1090 / pyModeS.
Permet de résoudre la position à partir d'une paire de messages even/odd.
"""
import math


def cpr_nl(lat):
    """Nombre de zones de longitude (NL) pour une latitude donnée."""
    if lat == 0:
        return 59
    nz = 15
    a = 1 - math.cos(math.pi / (2 * nz))
    b = math.cos(math.pi / 180 * abs(lat)) ** 2
    if b < 1e-12:
        return 1
    arg = 1 - a / b
    if arg <= -1:
        return 1
    nl = 2 * math.pi / math.acos(arg)
    return max(1, int(math.floor(nl)))


def cpr_decode(even_lat, even_lon, odd_lat, odd_lon, even_time, odd_time, airborne=True):
    """Décode une position à partir d'une paire even/odd.

    even_lat/lon, odd_lat/lon : valeurs CPR brutes (0-131071)
    Retourne (lat, lon) en degrés, ou None si incohérent.
    """
    if None in (even_lat, even_lon, odd_lat, odd_lon):
        return None

    # Normalisation 17 bits
    even_lat /= 131072.0
    even_lon /= 131072.0
    odd_lat /= 131072.0
    odd_lon /= 131072.0

    dlat_even = 360.0 / 60.0 if airborne else 360.0 / 4.0
    dlat_odd = 360.0 / 59.0 if airborne else 360.0 / 4.0

    # Latitude
    j = math.floor(59 * even_lat - 60 * odd_lat + 0.5)
    lat_even = dlat_even * (j % 60 + even_lat)
    lat_odd = dlat_odd * (j % 59 + odd_lat)

    if lat_even >= 270:
        lat_even -= 360
    if lat_odd >= 270:
        lat_odd -= 360

    # les deux trames doivent tomber dans la même bande de latitude, sinon la paire est incohérente
    if cpr_nl(lat_even) != cpr_nl(lat_odd):
        return None

    # Choisir la plus récente
    if even_time >= odd_time:
        lat = lat_even
        nl = cpr_nl(lat)
        ni = max(nl - 0, 1) if airborne else 1
        dlon = 360.0 / ni
        m = math.floor(even_lon * (nl - 1) - odd_lon * nl + 0.5)
        lon = dlon * (m % ni + even_lon)
    else:
        lat = lat_odd
        nl = cpr_nl(lat)
        ni = max(nl - 1, 1) if airborne else 1
        dlon = 360.0 / ni
        m = math.floor(even_lon * (nl - 1) - odd_lon * nl + 0.5)
        lon = dlon * (m % ni + odd_lon)

    if lon >= 180:
        lon -= 360

    # Vérification de cohérence
    if abs(lat) > 90 or abs(lon) > 180:
        return None
    return lat, lon


class CPRTracker:
    """Suit les messages even/odd par avion pour résoudre les positions."""

    def __init__(self):
        self.tracks = {}  # icao -> {"even": ..., "odd": ..., "even_t": ..., "odd_t": ...}

    def update(self, icao, cpr_lat, cpr_lon, is_odd, timestamp, airborne=True):
        """Met à jour et tente de résoudre la position."""
        if icao not in self.tracks:
            self.tracks[icao] = {"even": None, "odd": None, "even_t": 0, "odd_t": 0}

        tr = self.tracks[icao]
        key = "odd" if is_odd else "even"
        tkey = "odd_t" if is_odd else "even_t"
        tr[key] = (cpr_lat, cpr_lon)
        tr[tkey] = timestamp

        # On a les deux ?
        if tr["even"] and tr["odd"] and abs(tr["even_t"] - tr["odd_t"]) < 10:
            pos = cpr_decode(
                tr["even"][0], tr["even"][1],
                tr["odd"][0], tr["odd"][1],
                tr["even_t"], tr["odd_t"],
                airborne
            )
            return pos
        return None
