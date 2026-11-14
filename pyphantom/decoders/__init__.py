"""Decodeurs numeriques et filtres audio pour PyPhantom SDR.

Source : bibliotheque RadioLab de l'auteur, reorganisee en package.
Certains modules optionnels (modes.py, gmsk.py, olivia.py) n'ont pas ete
fournis : les decoders concernes (ADSB ModeS avance, AIS GMSK, Contestia)
levent un RuntimeError explicite s'ils manquent.
"""
from . import adsb_cpr, ais_pos, crc16, crc24  # noqa: F401
from .adsb import ADSBDecoder, decode_adsb_payload  # noqa: F401
from .audiofx import APF, AudioFX, Biquad, Contour, DeepNotch  # noqa: F401
from .ctcss import CTCSSDecoder, DCSDecoder  # noqa: F401
from .digital import BitSlicer, DCSDetector, Pocsag, PocsagDecoder, golay_decode  # noqa: F401
from .dmr import DMRDecoder  # noqa: F401
from .c4fm import C4FMDecoder  # noqa: F401
from .acars import ACARSDecoder  # noqa: F401
from .ais import AISDecoder  # noqa: F401
from .contestia import ContestiaDecoder  # noqa: F401
from .ekf import EKFFilter, ExtendedKalmanFilter  # noqa: F401
from .fec import ConvolutionalFEC, HammingFEC, ReedSolomon63  # noqa: F401
from .fec_advanced import BerlekampMassey, TurboDecoder, ViterbiDecoder  # noqa: F401
from .fmradio import RDSDecoder, StereoPilot, ToneDetector  # noqa: F401
from .dsp_optimized import OptimizedDSP  # noqa: F401

__all__ = [
    "ADSBDecoder", "decode_adsb_payload", "adsb_cpr",
    "AISDecoder", "ais_pos",
    "ACARSDecoder", "C4FMDecoder", "ContestiaDecoder", "DMRDecoder",
    "CTCSSDecoder", "DCSDecoder", "BitSlicer", "DCSDetector", "Pocsag", "PocsagDecoder", "golay_decode",
    "AudioFX", "Biquad", "Contour", "APF", "DeepNotch",
    "RDSDecoder", "StereoPilot", "ToneDetector",
    "ExtendedKalmanFilter", "EKFFilter",
    "ReedSolomon63", "HammingFEC", "ConvolutionalFEC",
    "ViterbiDecoder", "BerlekampMassey", "TurboDecoder",
    "OptimizedDSP", "crc16", "crc24",
]
