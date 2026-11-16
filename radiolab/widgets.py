"""Éléments de l'interface indépendants de la fenêtre principale : thème, pas d'accord,
afficheur de fréquence, panneau à sections repliables."""
import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

NFFT = 4096
WF_ROWS = 300
WF_COLS = 1536             # points affichés sur la largeur (courbe et cascade)
AXIS_WIDTH = 58
MEM_LABELS = 24            # étiquettes de mémoires affichées au plus sur le spectre
PIC_MAX_ROWS = 4000        # lignes d'image gardées (un fax dure parfois vingt minutes)
PIC_SHOWN_ROWS = 600       # lignes visibles à l'écran : les plus récentes
FFT_SIZES = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
# Pas d'accord en Hz (les mêmes familles que SDR# : multiples de 6,25 kHz, 8,33 kHz aviation, 9 kHz PO...)
STEP_VALUES = [1, 10, 100, 500, 1000, 2500, 5000, 6250, 7500, 25000 / 3, 9000, 10000, 12500, 15000, 20000,
               25000, 30000, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000,
               1000000]


def step_label(hz):
    if hz < 1000:
        return "%d Hz" % hz
    if hz < 1000000:
        return ("%.2f" % (hz / 1000)).rstrip("0").rstrip(".").replace(".", ",") + " kHz"
    return "%d MHz" % (hz / 1000000)


def snap(hz, grid, direction=0):
    """Fréquence sur la grille du pas : la plus proche (0), ou la suivante vers le haut (+n) / le bas (-n)."""
    k = hz / grid
    if direction == 0:
        return int(round(round(k) * grid))
    on_grid = abs(hz - round(k) * grid) <= 0.6        # à l'arrondi au hertz près (pas de 8,33 kHz)
    if on_grid:
        k = round(k) + direction
    else:
        k = (np.floor(k) + direction) if direction > 0 else (np.ceil(k) + direction)
    return int(round(k * grid))


class FreqSpin(QtWidgets.QSpinBox):
    """Champ de fréquence : flèches et molette avancent de pas en pas en restant sur la grille."""
    grid = 1000.0
    snap_enabled = True

    def stepBy(self, steps):
        if self.snap_enabled:
            self.setValue(snap(self.value(), self.grid, steps))
        else:
            self.setValue(int(round(self.value() + steps * self.grid)))
NR_METHODS = [("MMSE-LSA (recommandé)", "lsa"), ("Wiener", "wiener"),
              ("Soustraction spectrale", "specsub"), ("Adaptatif LMS", "lms")]


def waterfall_lut():
    """Palette de la cascade : bleu nuit -> bleu -> cyan -> jaune -> rouge -> blanc."""
    stops = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    colors = np.array([[0, 0, 45], [0, 40, 170], [0, 180, 220], [240, 240, 0], [230, 30, 0], [255, 255, 255]], float)
    x = np.linspace(0, 1, 256)
    return np.stack([np.interp(x, stops, colors[:, i]) for i in range(3)], axis=1).astype(np.ubyte)


class TunePlot(pg.PlotWidget):
    """Graphique où la molette change la fréquence d'un pas ; Ctrl + molette garde le zoom habituel."""
    on_steps = None                      # fonction appelée avec le nombre de crans (+ vers le haut)
    _rest = 0

    def wheelEvent(self, ev):
        if ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier or self.on_steps is None:
            super().wheelEvent(ev)
            return
        self._rest += ev.angleDelta().y()                      # 120 par cran ; les pavés tactiles envoient des fractions
        steps = int(self._rest / 120)
        if steps:
            self._rest -= steps * 120
            self.on_steps(steps)
        ev.accept()


# Thème sombre de l'application (feuille de style Qt).
STYLE = """
QWidget { background: #14181d; color: #d5dbe1; font-size: 10pt; }
QMainWindow, QScrollArea { background: #0d1014; border: none; }
QGroupBox { border: 1px solid #2a323b; border-top: none; margin: 0 0 6px 0; padding: 6px 4px 4px 4px; }
QPushButton { background: #232a32; border: 1px solid #36404a; border-radius: 3px; padding: 4px 9px; }
QPushButton:hover { background: #2c3540; }
QPushButton:checked { background: #1f6fb0; border-color: #3d95d8; color: white; }
QPushButton:disabled { color: #6b747d; }
QPushButton#sectionHeader { background: #1c2229; border: 1px solid #2a323b; border-radius: 0; text-align: left;
                            padding: 5px 8px; font-weight: bold; color: #e8edf2; }
QPushButton#sectionHeader:checked { background: #242c35; border-color: #2a323b; color: #e8edf2; }
QPushButton#startButton { font-size: 15pt; min-width: 42px; padding: 2px 8px; }
QPushButton#modeButton { padding: 4px 2px; min-width: 52px; }
QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QListWidget, QLineEdit {
    background: #0d1014; border: 1px solid #36404a; border-radius: 2px; padding: 2px 4px;
    selection-background-color: #1f6fb0; }
QComboBox QAbstractItemView { background: #14181d; selection-background-color: #1f6fb0; }
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled { color: #6b747d; }
QCheckBox:disabled, QLabel:disabled { color: #6b747d; }
QSlider::groove:horizontal { height: 4px; background: #36404a; border-radius: 2px; }
QSlider::handle:horizontal { width: 10px; margin: -7px 0; background: #3d95d8; border-radius: 3px; }
QSlider::groove:vertical { width: 4px; background: #36404a; border-radius: 2px; }
QSlider::handle:vertical { height: 10px; margin: 0 -7px; background: #3d95d8; border-radius: 3px; }
QScrollBar:vertical { background: #0d1014; width: 11px; }
QScrollBar::handle:vertical { background: #36404a; min-height: 24px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QStatusBar { background: #0d1014; color: #9aa4ad; }
QProgressBar { background: #0d1014; border: 1px solid #36404a; border-radius: 2px; height: 12px; }
QProgressBar::chunk { background: #3fbf6f; }
QToolTip { background: #232a32; color: #e8edf2; border: 1px solid #3d95d8; }
QSplitter::handle { background: #2a323b; }
QSplitter::handle:hover { background: #3d95d8; }
QSplitter::handle:horizontal { width: 6px; }
QLabel#freqDigit { font-size: 30pt; font-weight: 300; color: #f2f5f8; background: transparent; padding: 0 1px; }
QLabel#freqDigit:hover { background: #1f6fb0; }
QLabel#freqDot { font-size: 30pt; color: #6b747d; background: transparent; }
"""
OPEN_BY_DEFAULT = ("Source", "Radio")


def Section(title):
    """Cadre d'un groupe de réglages ; son titre devient l'en-tête repliable du panneau de gauche."""
    box = QtWidgets.QGroupBox()
    box.section_title = title
    return box


class SidePanel:
    """Panneau de gauche : une pile de sections que l'on replie ou déplie d'un clic sur leur en-tête."""

    def __init__(self, layout, cfg):
        self.layout, self.state = layout, cfg.setdefault("panels", {})
        self.headers = {}

    def addWidget(self, box, stretch=0):
        title = box.section_title
        header = QtWidgets.QPushButton()
        header.setObjectName("sectionHeader")
        header.setCheckable(True)
        header.toggled.connect(lambda on, t=title, h=header, b=box: self.show_section(t, h, b, on))
        self.layout.addWidget(header)
        self.layout.addWidget(box)
        self.headers[title] = header
        opened = bool(self.state.get(title, title in OPEN_BY_DEFAULT))
        header.setChecked(opened)
        self.show_section(title, header, box, opened)

    def show_section(self, title, header, box, on):
        self.state[title] = bool(on)
        header.setText(("▾  " if on else "▸  ") + title)
        box.setVisible(bool(on))

    def addStretch(self, *_):
        pass


class DigitLabel(QtWidgets.QLabel):
    """Un chiffre de la fréquence : molette, ou clic sur sa moitié haute / basse, pour le faire varier."""

    def __init__(self, power, on_delta):
        super().__init__("0")
        self.unit, self.on_delta = 10 ** power, on_delta
        self.setObjectName("freqDigit")
        self.setToolTip("Molette, ou clic en haut / en bas du chiffre")

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        if delta:
            self.on_delta(self.unit if delta > 0 else -self.unit)
        ev.accept()

    def mousePressEvent(self, ev):
        up = ev.position().y() < self.height() / 2
        self.on_delta(self.unit if up else -self.unit)
        ev.accept()


class FreqDigits(QtWidgets.QWidget):
    """Grand afficheur de fréquence, chiffre par chiffre, groupés par trois."""
    DIGITS = 10

    def __init__(self, on_delta):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(0)
        self.labels = []
        for power in range(self.DIGITS - 1, -1, -1):
            label = DigitLabel(power, on_delta)
            self.labels.append(label)
            lay.addWidget(label)
            if power in (9, 6, 3):
                dot = QtWidgets.QLabel(".")
                dot.setObjectName("freqDot")
                lay.addWidget(dot)
        self.text = ""
        self.set_value(0)

    def set_value(self, hz):
        self.text = "%0*d" % (self.DIGITS, max(0, int(round(hz))))
        leading = True
        for label, ch in zip(self.labels, self.text):
            leading = leading and ch == "0"
            label.setText(ch)
            label.setStyleSheet("color: #4a535c;" if leading else "")       # zéros de tête estompés


def s_units(dbm, vhf=False):
    """Niveau en unités S : (texte, fraction de 0 à 1 pour la barre).

    Convention des radioamateurs : S9 = -73 dBm en HF (-93 dBm au-dessus de 30 MHz), 6 dB par unité S.
    """
    s9 = -93.0 if vhf else -73.0
    over = dbm - s9
    if over > 0.5:
        text = "S9+%d dB" % int(round(over))
    else:
        text = "S%d" % int(min(max(round(9 + over / 6.0), 0), 9))
    span_low, span_high = 54.0, 60.0                         # la barre va de S0 à S9+60
    return text, float(min(max((over + span_low) / (span_low + span_high), 0.0), 1.0))


# Raccourcis clavier : (touche, description). Les actions sont branchées par la fenêtre principale.
SHORTCUTS = [("Right", "Fréquence : un pas vers le haut"), ("Left", "Fréquence : un pas vers le bas"),
             ("Ctrl+Right", "Fréquence : dix pas vers le haut"), ("Ctrl+Left", "Fréquence : dix pas vers le bas"),
             ("PgUp", "Mode suivant"), ("PgDown", "Mode précédent"), ("Space", "Couper ou rétablir le son"),
             ("M", "Mémoriser la fréquence"), ("R", "Enregistrer le son (marche / arrêt)"),
             ("L", "Verrouiller la fréquence"), ("P", "Fichier I/Q : pause / lecture"),
             ("F1", "Afficher cette liste")]
KEY_NAMES = {"Right": "→", "Left": "←", "Ctrl+Right": "Ctrl + →", "Ctrl+Left": "Ctrl + ←", "PgUp": "Page haut",
             "PgDown": "Page bas", "Space": "Espace"}
MODE_ORDER = ("NFM", "AM", "LSB", "USB", "WFM", "DSB", "CW", "RAW")       # ordre du parcours au clavier
