"""Plan de bandes indicatif (Europe / France, Région 1 de l'UIT).

Sert à colorer le spectre et à nommer la bande écoutée.  Les limites exactes et
les usages varient selon les pays : à vérifier pour tout usage sérieux.
"""
# (début Hz, fin Hz, nom, famille)
BANDS = [
    (148_500, 283_500, "Grandes ondes", "radio"), (526_500, 1_606_500, "Petites ondes", "radio"),
    (1_810_000, 2_000_000, "160 m amateur", "amateur"), (3_500_000, 3_800_000, "80 m amateur", "amateur"),
    (5_351_500, 5_366_500, "60 m amateur", "amateur"), (5_900_000, 6_200_000, "49 m radiodiffusion", "radio"),
    (7_000_000, 7_200_000, "40 m amateur", "amateur"), (7_200_000, 7_450_000, "41 m radiodiffusion", "radio"),
    (9_400_000, 9_900_000, "31 m radiodiffusion", "radio"), (10_100_000, 10_150_000, "30 m amateur", "amateur"),
    (11_600_000, 12_100_000, "25 m radiodiffusion", "radio"), (13_570_000, 13_870_000, "22 m radiodiffusion", "radio"),
    (14_000_000, 14_350_000, "20 m amateur", "amateur"), (15_100_000, 15_800_000, "19 m radiodiffusion", "radio"),
    (17_480_000, 17_900_000, "16 m radiodiffusion", "radio"), (18_068_000, 18_168_000, "17 m amateur", "amateur"),
    (21_000_000, 21_450_000, "15 m amateur", "amateur"), (21_450_000, 21_850_000, "13 m radiodiffusion", "radio"),
    (24_890_000, 24_990_000, "12 m amateur", "amateur"), (26_965_000, 27_405_000, "CB 27 MHz", "divers"),
    (28_000_000, 29_700_000, "10 m amateur", "amateur"), (50_000_000, 52_000_000, "6 m amateur", "amateur"),
    (87_500_000, 108_000_000, "Radiodiffusion FM", "radio"), (108_000_000, 118_000_000, "Radionavigation aéro", "aero"),
    (118_000_000, 137_000_000, "Aviation (AM)", "aero"), (137_000_000, 138_000_000, "Satellites météo", "divers"),
    (144_000_000, 146_000_000, "2 m amateur", "amateur"), (156_000_000, 162_050_000, "Marine VHF", "divers"),
    (174_000_000, 240_000_000, "Radio numérique DAB", "radio"), (430_000_000, 440_000_000, "70 cm amateur", "amateur"),
    (446_000_000, 446_200_000, "PMR446", "divers"), (1_240_000_000, 1_300_000_000, "23 cm amateur", "amateur"),
]
COLORS = {"amateur": (0, 200, 90), "radio": (230, 170, 0), "aero": (80, 150, 255), "divers": (200, 90, 220)}


def band_at(hz):
    """Nom de la bande contenant cette fréquence, ou chaîne vide."""
    for lo, hi, name, _ in BANDS:
        if lo <= hz <= hi:
            return name
    return ""
