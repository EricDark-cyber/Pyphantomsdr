"""Extraction de position AIS (messages de type 1/2/3)."""
import math


def decode_ais_position(bits):
    """Décode la position à partir d'un message AIS type 1/2/3 (168 bits).

    Retourne (lat, lon, mmsi) ou None.
    """
    if len(bits) < 168:
        return None
    try:
        # MMSI : bits 8-37
        mmsi = _bits_to_int(bits[8:38])
        if not (200000000 <= mmsi <= 799999999):
            return None

        # Longitude : bits 61-88 (28 bits, signé)
        lon_raw = _bits_to_int(bits[61:89])
        if lon_raw & 0x8000000:
            lon_raw -= 0x10000000
        lon = lon_raw / 600000.0

        # Latitude : bits 89-115 (27 bits, signé)
        lat_raw = _bits_to_int(bits[89:116])
        if lat_raw & 0x4000000:
            lat_raw -= 0x8000000
        lat = lat_raw / 600000.0

        if abs(lat) > 90 or abs(lon) > 180:
            return None
        return lat, lon, mmsi
    except Exception:
        return None


def _bits_to_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v
