"""Interface graphique (PySide6 + pyqtgraph) : spectre, cascade et réglages."""
import datetime
import os
import sys
import time
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
pg.setConfigOptions(antialias=False, useOpenGL=True)

from .radio import Radio, MODES, load_config, save_config, allowed_decimations
from .sources.iqfile import IQFileSource
from .scanner import Scanner
from .bandplan import BANDS, COLORS, band_at
from .dsp.spectrum import WINDOWS, DEFAULT_WINDOW, TraceMemory
from .dsp.pictures import write_png
from .dsp.metar import MetarCollector, format_metar
import math

def maidenhead_to_latlon(locator):
    """Convertit un locator Maidenhead (4 ou 6 caractères) en (lat, lon)."""
    locator = locator.upper().strip()
    if len(locator) < 4:
        return None
    try:
        lon = (ord(locator[0]) - ord('A')) * 20 - 180
        lat = (ord(locator[1]) - ord('A')) * 10 - 90
        lon += int(locator[2]) * 2
        lat += int(locator[3]) * 1
        if len(locator) >= 6:
            lon += (ord(locator[4]) - ord('A')) * 5 / 60
            lat += (ord(locator[5]) - ord('A')) * 2.5 / 60
        # Centre du carré
        lon += 1 if len(locator) < 6 else 2.5 / 60
        lat += 0.5 if len(locator) < 6 else 1.25 / 60
        return lat, lon
    except Exception:
        return None

def haversine_km(lat1, lon1, lat2, lon2):
    """Distance en km entre deux points (formule de Haversine)."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

MY_LOCATOR = "JN33ad"
from .dsp.sigmet import SigmetCollector, format_sigmet
from .dsp.adsb import ADSBDecoder, decode_adsb_payload
from .dsp.modes import decode_mode_s
from .dsp.acars import ACARSDecoder
from .dsp.ais import AISDecoder
from .netcontrol import RigCtlServer, SerialCatServer
from . import planner

from .widgets import *        # noqa: F401,F403  (constantes, thème et petits éléments d'interface)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, radio, cfg):
        super().__init__()
        self.radio, self.cfg = radio, cfg
        self.setWindowTitle("PyPhantom SDR")
        self.setStyleSheet("""
            QMainWindow { background: #1a1a1a; }
            QWidget { background: #1a1a1a; color: #ddd; }
            QPushButton { 
                background: #2d2d2d; 
                border: 1px solid #444; 
                padding: 5px 12px; 
                border-radius: 2px;
                min-width: 60px;
            }
            QPushButton:hover { background: #3d3d3d; }
            QPushButton:pressed { background: #1d1d1d; }
            QComboBox { 
                background: #2d2d2d; 
                border: 1px solid #444; 
                padding: 4px;
                border-radius: 2px;
            }
            QComboBox:hover { border-color: #666; }
            QComboBox::drop-down { border: none; }
            QSlider::groove:horizontal { 
                background: #333; 
                height: 6px; 
                border-radius: 3px;
            }
            QSlider::handle:horizontal { 
                background: #888; 
                width: 14px; 
                margin: -4px 0;
                border-radius: 7px;
            }
            QGroupBox { 
                border: 1px solid #333; 
                margin-top: 10px; 
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 10px; 
                color: #aaa;
            }
            QLabel { color: #ccc; }
            QCheckBox { color: #ccc; }
            QCheckBox::indicator { 
                width: 14px; 
                height: 14px;
                border: 1px solid #555;
                background: #2d2d2d;
            }
            QCheckBox::indicator:checked { background: #4a9; }
            QStatusBar { background: #222; color: #888; }
            QMenuBar { background: #222; color: #ddd; }
            QMenuBar::item:selected { background: #3a3a3a; }
            QDockWidget { 
                titlebar-close-icon: none;
                font-weight: bold;
            }
            QDockWidget::title { 
                background: #252525; 
                padding: 6px;
                border-bottom: 1px solid #333;
            }
        """)
        self.resize(1280, 800)
        self.wf = np.full((WF_ROWS, WF_COLS), -140.0, dtype=np.float32)
        self.wf_range = None                                   # bande de fréquences couverte par la cascade
        self._spec_next = 0.0
        self.resolution_hz = 0.0
        self._updating = False
        self._ticks = 0
        self._shown_center = None
        self._followed_vfo = None
        self.scanner = Scanner(radio)
        self.scheduler = planner.Scheduler(cfg.setdefault("jobs", []))
        self.net = None
        self.cat = None
        self.net_port = 4532
        self.notch_values = []
        self._was_holding = False
        self._pocsag_seen = 0
        self.metar = MetarCollector()
        self.sigmet = SigmetCollector()
        self.adsb = None
        self.acars = None
        self.ais = None
        self.trace = TraceMemory()                             # moyenne et crêtes de la courbe du spectre
        self._band_set = False                                 # vrai pendant que le programme déplace lui-même la zone du filtre
        self._player_set = False
        self.pic_rows, self.pic_id, self.pic_done, self._pic_decoder = [], None, 0, None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        pg.setConfigOptions(imageAxisOrder="row-major", antialias=False, background="#0a0d11", foreground="#aeb7c0")
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 0)
        top = QtWidgets.QHBoxLayout()
        outer.addLayout(top)
        # Splitter horizontal pour permettre le redimensionnement à la souris des panneaux gauche et droit
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        outer.addWidget(self.splitter, 1)

        # panneau de gauche : sections repliables, avec défilement si l'écran est trop petit
        self.side = QtWidgets.QScrollArea()
        self.side.setWidgetResizable(True)
        self.side.setMinimumWidth(180)
        self.side.setMaximumWidth(700)
        self.side.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_page = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side_page)
        side_layout.setContentsMargins(0, 0, 4, 0)
        side_layout.setSpacing(0)
        self.side.setWidget(side_page)
        self.splitter.addWidget(self.side)
        self.panels = SidePanel(side_layout, cfg)

        # Contenu central (spectre, cascade, etc.)
        center = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(center)
        left.setContentsMargins(0, 0, 0, 0)
        self.splitter.addWidget(center)

        # ---- barre du haut : panneau, marche/arrêt, fichier, volume, fréquence, pas
        self.btn_side = QtWidgets.QPushButton("☰")
        self.btn_side.setObjectName("startButton")
        self.btn_side.setToolTip("Afficher ou masquer le panneau des réglages")
        self.btn_side.setCheckable(True)
        self.btn_side.setChecked(bool(cfg.get("side_visible", True)))
        self.btn_side.toggled.connect(self.on_side)
        self.side.setVisible(self.btn_side.isChecked())
        self.btn_start = QtWidgets.QPushButton("▶ Airspy")
        self.btn_start.setToolTip("Démarrer ou arrêter l'Airspy")
        self.btn_start.clicked.connect(self.toggle_airspy)
        self.mute = QtWidgets.QPushButton("🔊")
        self.mute.setToolTip("Couper ou rétablir le son")
        self.mute.setCheckable(True)
        self.mute.toggled.connect(self.on_mute)
        self.vol_box = QtWidgets.QHBoxLayout()
        self.freq_digits = FreqDigits(self.on_digit)
        self.btn_file = QtWidgets.QPushButton("Ouvrir un fichier I/Q…")
        self.btn_file.clicked.connect(self.open_file)
        self.freq = FreqSpin()
        self.freq.setRange(0, 1_800_000_000)
        self.freq.setGroupSeparatorShown(True)
        self.freq.setSuffix(" Hz")
        self.freq.setKeyboardTracking(False)
        f = self.freq.font(); f.setPointSize(11); self.freq.setFont(f)
        self.freq.setMinimumWidth(170)
        self.freq.setToolTip("Saisie directe de la fréquence, en hertz")
        self.freq.valueChanged.connect(self.on_freq)
        self.freq.valueChanged.connect(self.freq_digits.set_value)
        self.step = QtWidgets.QComboBox()
        for hz in STEP_VALUES:
            self.step.addItem(step_label(hz), hz)
        self.step.setMaxVisibleItems(len(STEP_VALUES))
        self.step.currentIndexChanged.connect(self.on_step)
        wanted = cfg.get("step_hz", 1000)
        self.step.setCurrentIndex(min(range(len(STEP_VALUES)), key=lambda i: abs(STEP_VALUES[i] - wanted)))
        self.on_step(self.step.currentIndex())
        for w in (self.btn_side, self.btn_start, self.btn_file, self.mute):
            top.addWidget(w)
        top.addLayout(self.vol_box)
        for w in (self.freq_digits, self.freq, QtWidgets.QLabel("Pas"), self.step):
            top.addWidget(w)
        # modes : une grille de boutons, placée en tête de la section « Radio »
        self.mode_buttons = {}
        group = QtWidgets.QButtonGroup(self)
        self.mode_grid = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(self.mode_grid)
        grid.setContentsMargins(0, 0, 0, 4)
        grid.setSpacing(4)
        for i, mode in enumerate(("NFM", "AM", "LSB", "USB", "WFM", "DSB", "CW", "RAW", "TETRA")):
            b = QtWidgets.QPushButton(mode)
            b.setObjectName("modeButton")
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, m=mode: self.on_mode(m))
            group.addButton(b)
            self.mode_buttons[mode] = b
            grid.addWidget(b, i // 4, i % 4)
        self.band_label = QtWidgets.QLabel("")
        top.addWidget(self.band_label)
        top.addStretch(1)
        self.s_bar = QtWidgets.QProgressBar()
        self.s_bar.setRange(0, 1000); self.s_bar.setTextVisible(False); self.s_bar.setFixedWidth(170)
        self.s_bar.setToolTip("S-mètre : de S0 à S9+60 dB (à étalonner dans la section Affichage)")
        self.s_text = QtWidgets.QLabel("S0")
        self.s_text.setMinimumWidth(86)
        top.addWidget(self.s_bar); top.addWidget(self.s_text)

        # ---- bandeau de décodage (modes numériques HF) : tracé du niveau et texte, sur toute la largeur
        self.digi_panel = QtWidgets.QWidget()
        dlay = QtWidgets.QVBoxLayout(self.digi_panel)
        dlay.setContentsMargins(0, 0, 0, 0)
        head = QtWidgets.QHBoxLayout()
        self.digi_title = QtWidgets.QLabel("Décodeur")
        f = self.digi_title.font(); f.setBold(True); self.digi_title.setFont(f)
        self.digi_status = QtWidgets.QLabel("")
        self.digi_clear = QtWidgets.QPushButton("Effacer le texte")
        self.digi_clear.clicked.connect(lambda *_: self.on_digi_clear())
        head.addWidget(self.digi_title); head.addWidget(self.digi_status, 1); head.addWidget(self.digi_clear)
        self.level_plot = pg.PlotWidget()
        self.level_plot.setFixedHeight(110)
        self.level_plot.setMouseEnabled(x=False, y=False)
        self.level_plot.hideAxis("bottom")
        self.level_plot.setLabel("left", "Niveau", units="dB")
        self.level_plot.showGrid(x=False, y=True, alpha=0.3)
        self.level_curve = self.level_plot.plot(pen=None, symbol="o", symbolSize=2, symbolPen=None, symbolBrush="w")
        self.level_thr = pg.InfiniteLine(pos=-60, angle=0, movable=False, pen=pg.mkPen("g"))
        self.level_plot.addItem(self.level_thr)
        self.digi_out = QtWidgets.QPlainTextEdit(); self.digi_out.setReadOnly(True)
        # retour à la ligne automatique à la largeur de la fenêtre, même au milieu d'un long mot
        self.digi_out.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.digi_out.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.digi_out.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.digi_out.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.digi_out.setFixedHeight(180)
        self.const_plot = pg.PlotWidget()
        self.const_plot.setFixedSize(110, 110)
        self.const_plot.setToolTip("Constellation : deux taches nettes à gauche et à droite = signal PSK propre")
        self.const_plot.setMouseEnabled(x=False, y=False)
        self.const_plot.hideAxis("bottom"); self.const_plot.hideAxis("left")
        self.const_plot.setXRange(-1.8, 1.8, padding=0); self.const_plot.setYRange(-1.8, 1.8, padding=0)
        self.const_curve = self.const_plot.plot(pen=None, symbol="o", symbolSize=3, symbolPen=None, symbolBrush="#7fd4ff")
        self.const_plot.setVisible(False)
        graphs = QtWidgets.QHBoxLayout()
        graphs.addWidget(self.level_plot, 1); graphs.addWidget(self.const_plot)
        dlay.addLayout(head); dlay.addLayout(graphs); dlay.addWidget(self.digi_out)
        # Fenêtre FT8/FT4 style cartes
        self.ft8_frame = QtWidgets.QFrame()
        self.ft8_frame.setVisible(False)
        self.ft8_frame.setStyleSheet("QFrame { background: #1a1d23; border-radius: 6px; }")
        ft8_lay = QtWidgets.QVBoxLayout(self.ft8_frame)
        ft8_lay.setContentsMargins(8, 6, 8, 6)
        ft8_lay.setSpacing(4)
        # En-tête
        head_ft8 = QtWidgets.QHBoxLayout()
        self.ft8_title = QtWidgets.QLabel("FT8 Messages")
        self.ft8_title.setStyleSheet("color: #e2e8f0; font-weight: bold; font-size: 13px;")
        self.ft8_far = QtWidgets.QLabel("")
        self.ft8_far.setStyleSheet("color: #a0aec0; font-size: 11px;")
        head_ft8.addWidget(self.ft8_title)
        head_ft8.addStretch()
        head_ft8.addWidget(self.ft8_far)
        ft8_lay.addLayout(head_ft8)
        # Liste
        self.ft8_list = QtWidgets.QListWidget()
        self.ft8_list.setStyleSheet("""
            QListWidget { background: transparent; border: none; color: #e2e8f0; }
            QListWidget::item { background: #252a33; border-radius: 4px; margin: 2px 0; padding: 6px 8px; }
            QListWidget::item:selected { background: #2d3748; }
        """)
        self.ft8_list.setMinimumHeight(120)
        self.ft8_list.setMaximumHeight(220)
        self.ft8_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.ft8_list.itemClicked.connect(self.on_ft8_locator_click)
        ft8_lay.addWidget(self.ft8_list)
        dlay.addWidget(self.ft8_frame)

        # Affichage des METAR extraits
        self.metar_list = QtWidgets.QListWidget()
        self.metar_list.setMaximumHeight(120)
        self.metar_list.setWordWrap(True)
        self.metar_clear = QtWidgets.QPushButton("Effacer les METAR")
        self.metar_clear.clicked.connect(self.on_metar_clear)
        self.metar_label = QtWidgets.QLabel("Messages METAR :")
        dlay.addWidget(self.metar_label)
        dlay.addWidget(self.metar_list)
        dlay.addWidget(self.metar_clear)

        # SIGMET
        self.sigmet_list = QtWidgets.QListWidget()
        self.sigmet_list.setMaximumHeight(120)
        self.sigmet_list.setWordWrap(True)
        self.sigmet_clear = QtWidgets.QPushButton("Effacer les SIGMET")
        self.sigmet_clear.clicked.connect(self.on_sigmet_clear)
        self.sigmet_label = QtWidgets.QLabel("Messages SIGMET :")
        dlay.addWidget(self.sigmet_label)
        dlay.addWidget(self.sigmet_list)
        dlay.addWidget(self.sigmet_clear)
        left.addWidget(self.digi_panel)
        self.digi_panel.setVisible(False)

        # ---- Panneau TETRA amélioré (dockable)
        self.tetra_dock = QtWidgets.QDockWidget("TETRA", self)
        self.tetra_dock.setStyleSheet("""
            QDockWidget { font-weight: bold; }
            QDockWidget::title { background: #2a2a2a; padding: 4px; }
            QGroupBox { font-weight: bold; border: 1px solid #444; margin-top: 8px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
            QLabel { color: #ddd; }
            QPushButton { background: #3a3a3a; border: 1px solid #555; padding: 4px 8px; border-radius: 3px; }
            QPushButton:hover { background: #4a4a4a; }
            QCheckBox { color: #ddd; }
        """)
        self.tetra_dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea | QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self.tetra_panel = QtWidgets.QWidget()
        self.tetra_dock.setWidget(self.tetra_panel)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.tetra_dock)
        self.tetra_dock.hide()
        tlay = QtWidgets.QVBoxLayout(self.tetra_panel)
        tlay.setContentsMargins(4, 4, 4, 4)

        # En-tête
        head = QtWidgets.QHBoxLayout()
        self.tetra_demod = QtWidgets.QCheckBox("Demodulator")
        self.tetra_demod.setChecked(True)
        head.addWidget(self.tetra_demod)
        head.addStretch()
        head.addWidget(QtWidgets.QLabel("Received"))
        tlay.addLayout(head)

        # Infos cellule
        grid = QtWidgets.QGridLayout()
        self.tetra_mcc = QtWidgets.QLabel("MCC: —")
        self.tetra_mnc = QtWidgets.QLabel("MNC: —")
        self.tetra_la = QtWidgets.QLabel("LA: —")
        self.tetra_color = QtWidgets.QLabel("Color: —")
        self.tetra_main = QtWidgets.QLabel("Main: —")
        self.tetra_current = QtWidgets.QLabel("Current: —")
        grid.addWidget(self.tetra_mcc, 0, 0)
        grid.addWidget(self.tetra_main, 0, 1)
        grid.addWidget(self.tetra_mnc, 1, 0)
        grid.addWidget(self.tetra_current, 1, 1)
        grid.addWidget(self.tetra_la, 2, 0)
        grid.addWidget(self.tetra_color, 3, 0)
        tlay.addLayout(grid)

        # Boutons
        btns = QtWidgets.QHBoxLayout()
        self.tetra_config = QtWidgets.QPushButton("Config")
        self.tetra_config.clicked.connect(self.on_tetra_config)
        self.tetra_netinfo = QtWidgets.QPushButton("Net Info")
        self.tetra_netinfo.clicked.connect(self.on_tetra_netinfo)
        self.tetra_auto_play = QtWidgets.QCheckBox("Lecture auto")
        self.tetra_auto_play.setToolTip("Joue automatiquement quand un timeslot est actif")
        self.tetra_play = QtWidgets.QPushButton("▶ Voix")
        self.tetra_play.clicked.connect(self.on_tetra_play)
        btns.addWidget(self.tetra_config)
        btns.addWidget(self.tetra_auto_play)
        btns.addWidget(self.tetra_play)
        btns.addStretch()
        btns.addWidget(self.tetra_netinfo)
        tlay.addLayout(btns)

        # SDS / Packet Data
        sds_box = QtWidgets.QGroupBox("Messages")
        sds_lay = QtWidgets.QVBoxLayout(sds_box)
        self.tetra_sds = QtWidgets.QTextEdit()
        self.tetra_sds.setReadOnly(True)
        self.tetra_sds.setMaximumHeight(80)
        self.tetra_sds.setPlaceholderText("SDS et Packet Data...")
        sds_lay.addWidget(self.tetra_sds)
        tlay.addWidget(sds_box)

        # Voice / Timeslots
        voice = QtWidgets.QGroupBox("Voice")
        vlay = QtWidgets.QGridLayout(voice)
        self.tetra_auto = QtWidgets.QCheckBox("Auto")
        self.tetra_auto.setChecked(True)
        self.tetra_auto.toggled.connect(self.on_tetra_auto)
        self.tetra_audio = QtWidgets.QCheckBox("Audio")
        self.tetra_audio.setChecked(False)
        self.tetra_audio.setToolTip("Active la lecture audio continue du timeslot sélectionné")
        vlay.addWidget(self.tetra_auto, 0, 0)
        vlay.addWidget(self.tetra_audio, 0, 3)
        vlay.addWidget(QtWidgets.QLabel("From"), 0, 1)
        vlay.addWidget(QtWidgets.QLabel("To"), 0, 2)
        self.tetra_ts_radios = []
        for i in range(4):
            rb = QtWidgets.QRadioButton(f"Ts {i+1}")
            rb.toggled.connect(lambda on, ts=i+1: self.on_tetra_ts_radio(on, ts))
            self.tetra_ts_radios.append(rb)
            vlay.addWidget(rb, i+1, 0)
            vlay.addWidget(QtWidgets.QLabel("0"), i+1, 1)
            vlay.addWidget(QtWidgets.QLabel("0"), i+1, 2)
        tlay.addWidget(voice)

        # Filtre RRC
        rrc_box = QtWidgets.QGroupBox("Filtre RRC")
        rrc_lay = QtWidgets.QFormLayout(rrc_box)
        self.rrc_alpha = QtWidgets.QDoubleSpinBox()
        self.rrc_alpha.setRange(0.1, 1.0)
        self.rrc_alpha.setSingleStep(0.05)
        self.rrc_alpha.setValue(0.35)
        self.rrc_alpha.valueChanged.connect(self.on_rrc_changed)
        self.rrc_span = QtWidgets.QSpinBox()
        self.rrc_span.setRange(4, 20)
        self.rrc_span.setValue(12)
        self.rrc_span.valueChanged.connect(self.on_rrc_changed)
        rrc_lay.addRow("Alpha", self.rrc_alpha)
        rrc_lay.addRow("Span", self.rrc_span)
        tlay.addWidget(rrc_box)

        # Courbe de réponse RRC
        self.rrc_plot = pg.PlotWidget()
        self.rrc_plot.setFixedHeight(80)
        self.rrc_plot.setLabel("bottom", "Fréquence", units="kHz")
        self.rrc_curve = self.rrc_plot.plot(pen=pg.mkPen("y"))
        tlay.addWidget(self.rrc_plot)

        # Indicateurs
        ind = QtWidgets.QHBoxLayout()
        self.tetra_fer = QtWidgets.QLabel("FER 0Hz")
        self.tetra_ber = QtWidgets.QLabel("BER 0.0%")
        self.tetra_mer = QtWidgets.QLabel("MER 0.0%")
        ind.addWidget(self.tetra_fer)
        ind.addWidget(self.tetra_ber)
        ind.addWidget(self.tetra_mer)
        tlay.addLayout(ind)

        # Diagramme (placeholder)
        self.tetra_diagram = QtWidgets.QLabel("Diagramme constellation")
        self.tetra_diagram.setMinimumHeight(80)
        self.tetra_diagram.setStyleSheet("background: #0a0d11; border: 1px solid #2a323b;")
        self.tetra_diagram.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        tlay.addWidget(self.tetra_diagram)

        # Le panneau TETRA sera ajouté dans la barre latérale
        # Le dock contrôle la visibilité

        # ---- spectre et cascade
        self.plot = TunePlot()
        self.plot.on_steps = self.on_wheel_steps
        self.plot.setLabel("bottom", "Fréquence", units="Hz")
        self.plot.setLabel("left", "Niveau", units="dB")
        self.plot.setYRange(-120, 0)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(pen=pg.mkPen("#e6ecf2"), fillLevel=-250, brush=(31, 111, 176, 110))
        self.peak_curve = self.plot.plot(pen=pg.mkPen("#ffb347"))
        self.peak_curve.setVisible(False)
        # zone du filtre : tirer un bord change la largeur, tirer le milieu déplace la fréquence
        self.band = pg.LinearRegionItem(values=(0, 1), movable=True, brush=(255, 255, 255, 45),
                                        hoverBrush=(255, 255, 255, 70), pen=pg.mkPen((255, 255, 255, 110)))
        self.band.sigRegionChangeFinished.connect(lambda *_: self.on_band_signal())
        self.plot.addItem(self.band)
        self.mem_labels = []
        for _ in range(MEM_LABELS):
            item = pg.TextItem("", color="#ffd866", anchor=(0, 0))
            item.setVisible(False)
            self.plot.addItem(item)
            self.mem_labels.append(item)
        self.vfo_line = pg.InfiniteLine(pos=0, angle=90, movable=True, pen=pg.mkPen("r", width=2))
        self.vfo_line.sigPositionChangeFinished.connect(lambda: self.tune(self.vfo_line.value()))
        self.plot.addItem(self.vfo_line)
        self.wf_plot = TunePlot()
        self.wf_plot.on_steps = self.on_wheel_steps
        self.wf_plot.setXLink(self.plot)
        self.wf_plot.setMouseEnabled(x=True, y=False)
        # même marge à gauche pour les deux graphiques : les fréquences sont alignées verticalement
        self.plot.getAxis("left").setWidth(AXIS_WIDTH)
        self.wf_plot.getAxis("left").setWidth(AXIS_WIDTH)
        self.wf_plot.getAxis("left").setStyle(showValues=False)
        self.wf_plot.hideAxis("bottom")
        self.wf_plot.getViewBox().invertY(True)
        self.img = pg.ImageItem()
        self.img.setLookupTable(waterfall_lut())
        self.wf_plot.addItem(self.img)
        self.wf_line = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=pg.mkPen("r"))
        self.wf_plot.addItem(self.wf_line)
        self.plot.scene().sigMouseClicked.connect(lambda ev: self.on_click(ev, self.plot))
        self.wf_plot.scene().sigMouseClicked.connect(lambda ev: self.on_click(ev, self.wf_plot))
        left.addWidget(self.plot, 2)
        left.addWidget(self.wf_plot, 3)

        # ---- lecteur de fichier I/Q : pause et position (visible seulement pendant la lecture d'un fichier)
        self.player = QtWidgets.QWidget()
        play = QtWidgets.QHBoxLayout(self.player)
        play.setContentsMargins(0, 0, 0, 0)
        self.play_pause = QtWidgets.QPushButton("⏸")
        self.play_pause.setCheckable(True)
        self.play_pause.setToolTip("Pause / lecture du fichier (touche P)")
        self.play_pause.toggled.connect(self.on_play_pause)
        self.play_pos = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.play_pos.setRange(0, 1000)
        self.play_pos.valueChanged.connect(self.on_play_seek)
        self.play_time = QtWidgets.QLabel("")
        play.addWidget(self.play_pause); play.addWidget(self.play_pos, 1); play.addWidget(self.play_time)
        left.addWidget(self.player)
        self.player.setVisible(False)

        # ---- images reçues (fax météo, SSTV)
        self.pic_panel = QtWidgets.QWidget()
        play = QtWidgets.QVBoxLayout(self.pic_panel)
        play.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        self.pic_title = QtWidgets.QLabel("Image reçue")
        self.pic_shift = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.pic_shift.setRange(0, 100); self.pic_shift.setFixedWidth(160)
        self.pic_shift.setToolTip("Fax pris en cours de route : décale l'image pour remettre son bord à gauche")
        self.pic_shift.valueChanged.connect(lambda *_: self.show_picture())
        self.pic_save = QtWidgets.QPushButton("Enregistrer l'image (PNG)")
        self.pic_save.clicked.connect(lambda *_: self.save_picture())
        self.pic_clear = QtWidgets.QPushButton("Effacer l'image")
        self.pic_clear.clicked.connect(lambda *_: self.clear_picture())
        row.addWidget(self.pic_title, 1); row.addWidget(QtWidgets.QLabel("Cadrage")); row.addWidget(self.pic_shift)
        row.addWidget(self.pic_save); row.addWidget(self.pic_clear)
        self.pic_plot = pg.PlotWidget()
        self.pic_plot.setFixedHeight(330)
        self.pic_plot.hideAxis("bottom"); self.pic_plot.hideAxis("left")
        self.pic_plot.getViewBox().invertY(True)
        self.pic_plot.getViewBox().setAspectLocked(True)
        self.pic_img = pg.ImageItem()
        self.pic_plot.addItem(self.pic_img)
        play.addLayout(row); play.addWidget(self.pic_plot)
        left.addWidget(self.pic_panel)
        self.pic_panel.setVisible(False)

        # ---- fenêtres d'analyse supplémentaires (cases à cocher dans « Affichage et accord »)
        def analysis_plot(title, color):
            w = pg.PlotWidget()
            w.setFixedHeight(150)
            w.setTitle(title)
            w.setLabel("bottom", "Fréquence", units="Hz")
            w.setMouseEnabled(x=False, y=False)
            w.showGrid(x=True, y=True, alpha=0.3)
            w.getAxis("left").setWidth(AXIS_WIDTH)
            curve = w.plot(pen=pg.mkPen(color), fillLevel=-200, brush=(40, 90, 150, 140))
            w.setVisible(False)
            left.addWidget(w)
            return w, curve
        self.if_plot, self.if_curve = analysis_plot("Spectre FI (autour de la fréquence écoutée)", "w")
        self.if_band = pg.LinearRegionItem(values=(0, 1), movable=False, brush=(255, 255, 255, 35))
        self.if_plot.addItem(self.if_band)
        self.if_plot.addItem(pg.InfiniteLine(pos=0, angle=90, movable=False, pen=pg.mkPen("r")))
        self.af_plot, self.af_curve = analysis_plot("Spectre audio (ce que tu entends)", "w")
        self.band_items = []                                   # plan de bandes : zones colorées sous la courbe
        for lo, hi, _, family in BANDS:
            item = pg.LinearRegionItem(values=(lo, hi), movable=False, brush=COLORS[family] + (28,),
                                       pen=pg.mkPen(COLORS[family] + (90,)))
            item.setZValue(-10)
            self.plot.addItem(item)
            self.band_items.append(item)

        # ---- curseurs verticaux à droite du spectre : zoom, contraste, plage et décalage de l'échelle
        col = QtWidgets.QVBoxLayout()
        self.right_panel = QtWidgets.QWidget()
        self.right_panel.setLayout(col)
        self.right_panel.setMinimumWidth(50)
        self.right_panel.setMaximumWidth(180)
        self.splitter.addWidget(self.right_panel)

        # Restaure les largeurs des panneaux (gauche, centre, droite) mémorisées
        default_sizes = [390, 780, 70]
        saved = cfg.get("panel_sizes")
        if isinstance(saved, list) and len(saved) == 3 and all(isinstance(x, (int, float)) and x > 0 for x in saved):
            self.splitter.setSizes([int(x) for x in saved])
        else:
            self.splitter.setSizes(default_sizes)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setHandleWidth(6)

        def vslider(title, lo, hi, value, tip):
            label = QtWidgets.QLabel(title)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
            s = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
            s.setRange(lo, hi); s.setValue(value); s.setToolTip(tip)
            s.valueChanged.connect(self.on_display)
            col.addWidget(label); col.addWidget(s, 1, QtCore.Qt.AlignmentFlag.AlignHCenter)
            return s
        self.zoom = vslider("Zoom", 0, 100, 0, "Resserre la vue autour de la fréquence écoutée")
        self.contrast = vslider("Contraste", 0, 100, int(cfg.get("contrast", 50)),
                                "Cascade en mode automatique : en haut, les signaux faibles ressortent davantage")
        self.range_db = vslider("Plage", 20, 160, int(cfg.get("range_db", 120)), "Hauteur de l'échelle du spectre, en dB")
        self.offset_db = vslider("Décalage", -100, 20, int(cfg.get("offset_db", 0)), "Niveau du haut de l'échelle, en dB")

        # ---- réglages : toutes les sections vont dans le panneau de gauche
        panel = disp_tab = noise_tab = mem_tab = info_tab = dec_tab = tools_tab = self.panels

        box = Section("Source"); form = QtWidgets.QFormLayout(box); panel.addWidget(box)
        self.rate = QtWidgets.QComboBox(); self.rate.currentIndexChanged.connect(self.on_rate)
        self.gain = self.slider(0, 21, cfg["gain"], self.on_gain)
        self.gain_mode = QtWidgets.QComboBox()
        self.gain_mode.addItem("Linéarité (signaux forts)", "linearity")
        self.gain_mode.addItem("Sensibilité (signaux faibles)", "sensitivity")
        self.gain_mode.setCurrentIndex(1 if cfg["gain_mode"] == "sensitivity" else 0)
        self.gain_mode.currentIndexChanged.connect(self.on_gain)
        self.spyverter = QtWidgets.QCheckBox("SpyVerter (HF, décalage +120 MHz)")
        self.spyverter.setChecked(bool(cfg["spyverter"]))
        self.spyverter.toggled.connect(self.on_spyverter)
        self.sv_ppm = QtWidgets.QDoubleSpinBox()
        self.sv_ppm.setRange(-200.0, 200.0); self.sv_ppm.setDecimals(2); self.sv_ppm.setSingleStep(0.05)
        self.sv_ppm.setSuffix(" ppm"); self.sv_ppm.setValue(float(cfg.get("sv_ppm", 0.0)))
        self.sv_ppm.setToolTip("Corrige l'erreur de l'oscillateur 120 MHz du SpyVerter (1 ppm = 120 Hz).\n"
                               "Régler sur une porteuse de fréquence connue jusqu'à ce qu'elle tombe juste.")
        self.sv_ppm.setEnabled(bool(cfg["spyverter"]))
        self.sv_ppm.valueChanged.connect(self.on_sv_ppm)
        self.bias_tee = QtWidgets.QCheckBox("Bias-tee (4,5 V sur l'antenne)")
        self.bias_tee.setToolTip("Alimente le SpyVerter ou un préamplificateur par le câble d'antenne.\n"
                                 "À laisser décoché si rien n'est à alimenter.")
        self.bias_tee.setChecked(bool(cfg["bias_tee"]))
        self.bias_tee.toggled.connect(self.on_bias_tee)
        self.decim = QtWidgets.QComboBox()
        self.decim.setToolTip("Divise le débit avant l'analyse : bande affichée plus étroite,\n"
                              "mais spectre et cascade d'autant plus détaillés.")
        self.decim.currentIndexChanged.connect(self.on_decim)
        self.span_label = QtWidgets.QLabel("")
        form.addRow("Débit", self.rate); form.addRow("Décimation", self.decim)
        form.addRow("Bande affichée", self.span_label); form.addRow("Gain", self.gain)
        form.addRow("Mode de gain", self.gain_mode); form.addRow(self.spyverter)
        self.calibrate = QtWidgets.QPushButton("Étalonner sur une porteuse connue…")
        self.calibrate.setToolTip("Accorder près d'un émetteur dont la fréquence exacte est connue (station de radiodiffusion,\n"
                                  "signal horaire…), puis cliquer : la correction SV PPM est calculée automatiquement.")
        self.calibrate.clicked.connect(lambda *_: self.on_calibrate())
        form.addRow("Correction SV PPM", self.sv_ppm); form.addRow(self.calibrate); form.addRow(self.bias_tee)
        self.shift_on = QtWidgets.QCheckBox("Décalage (autre convertisseur)")
        self.shift_on.setChecked(bool(cfg.get("shift_on", False)))
        self.shift_hz = QtWidgets.QSpinBox(); self.shift_hz.setRange(-2_000_000_000, 2_000_000_000)
        self.shift_hz.setGroupSeparatorShown(True); self.shift_hz.setSuffix(" Hz"); self.shift_hz.setKeyboardTracking(False)
        self.shift_hz.setValue(int(cfg.get("shift_hz", 0)))
        self.shift_hz.setToolTip("Fréquence affichée = fréquence de l'Airspy + décalage.\n"
                                 "Un convertisseur qui ajoute 125 MHz se compense avec -125 000 000.")
        self.shift_on.toggled.connect(self.on_shift); self.shift_hz.valueChanged.connect(self.on_shift)
        self.invert = QtWidgets.QCheckBox("Inverser le spectre"); self.invert.setChecked(bool(cfg.get("invert_iq", False)))
        self.dc_fix = QtWidgets.QCheckBox("Supprimer la raie centrale (continu)")
        self.dc_fix.setChecked(bool(cfg.get("remove_dc", False)))
        self.invert.toggled.connect(self.on_iq_options); self.dc_fix.toggled.connect(self.on_iq_options)
        form.addRow(self.shift_on); form.addRow(self.shift_hz); form.addRow(self.invert); form.addRow(self.dc_fix)
        self.on_iq_options()

        box = Section("Radio"); form = QtWidgets.QFormLayout(box); panel.addWidget(box)
        form.addRow(self.mode_grid)
        self.volume = self.slider(0, 100, int(cfg["volume"] * 100), self.on_audio)
        self.volume.setFixedWidth(120)
        self.volume.setToolTip("Volume")
        self.vol_box.addWidget(self.volume)
        self.squelch = self.slider(-150, 0, -150, self.on_audio)
        self.bw = QtWidgets.QSpinBox(); self.bw.setRange(100, 200000); self.bw.setSuffix(" Hz")
        self.bw.setSingleStep(100); self.bw.setKeyboardTracking(False); self.bw.valueChanged.connect(self.on_bw)
        self.agc = QtWidgets.QCheckBox("AGC (gain automatique)"); self.agc.setChecked(True)
        self.agc.toggled.connect(self.on_audio)
        self.smeter = QtWidgets.QLabel("— dB")
        form.addRow("Squelch", self.squelch)
        self.stereo = QtWidgets.QCheckBox("FM stéréo"); self.stereo.setChecked(True)
        self.stereo.toggled.connect(self.on_audio)
        self.am_sync = QtWidgets.QCheckBox("AM synchrone (contre l'évanouissement)")
        self.am_sync.setChecked(bool(cfg.get("am_sync", False))); self.am_sync.toggled.connect(self.on_audio)
        self.cw_pitch = QtWidgets.QSpinBox(); self.cw_pitch.setRange(300, 1200); self.cw_pitch.setSingleStep(50)
        self.cw_pitch.setSuffix(" Hz"); self.cw_pitch.setValue(int(cfg.get("cw_pitch", 700)))
        self.cw_pitch.setKeyboardTracking(False); self.cw_pitch.valueChanged.connect(self.on_audio)
        form.addRow("Largeur du filtre", self.bw); form.addRow(self.agc); form.addRow(self.stereo)
        self.edge_lo = QtWidgets.QSpinBox(); self.edge_hi = QtWidgets.QSpinBox()
        for w in (self.edge_lo, self.edge_hi):
            w.setRange(-100000, 100000); w.setSuffix(" Hz"); w.setSingleStep(50); w.setKeyboardTracking(False)
            w.valueChanged.connect(self.on_edges)
        self.edge_lo.setToolTip("Filtre asymétrique : bords du filtre par rapport à la fréquence affichée.")
        self.anti_fading = QtWidgets.QCheckBox("Anti-fading (meilleure bande latérale)")
        self.anti_fading.setChecked(bool(cfg.get("anti_fading", False))); self.anti_fading.toggled.connect(self.on_audio)
        self.anti_fading.setEnabled(self.am_sync.isChecked())
        self.am_sync.toggled.connect(lambda on: self.anti_fading.setEnabled(bool(on)))
        self.binaural = QtWidgets.QCheckBox("Binaural (BLU / CW, au casque)")
        self.binaural.setChecked(bool(cfg.get("binaural", False))); self.binaural.toggled.connect(self.on_audio)
        self.filter_window = QtWidgets.QComboBox()
        self.filter_window.addItem("Automatique (Kaiser)", None)
        for name in ("Blackman-Harris", "Blackman", "Hann", "Hamming", "Poisson"):
            self.filter_window.addItem(name, name)
        self.filter_order = QtWidgets.QSpinBox(); self.filter_order.setRange(0, 3001); self.filter_order.setSingleStep(100)
        self.filter_order.setSpecialValueText("Automatique"); self.filter_order.setKeyboardTracking(False)
        self.filter_order.setToolTip("Nombre de coefficients du filtre : plus il est grand, plus les flancs sont raides.")
        self.filter_window.currentIndexChanged.connect(self.on_filter_shape)
        self.filter_order.valueChanged.connect(self.on_filter_shape)
        form.addRow("Bord bas", self.edge_lo); form.addRow("Bord haut", self.edge_hi)
        form.addRow("Type de filtre", self.filter_window); form.addRow("Ordre du filtre", self.filter_order)
        form.addRow(self.am_sync); form.addRow(self.anti_fading); form.addRow(self.binaural)
        form.addRow("Note CW", self.cw_pitch)
        form.addRow("Niveau du canal", self.smeter)

        box = Section("Affichage"); form = QtWidgets.QFormLayout(box); disp_tab.addWidget(box)
        disp_tab.addStretch(1)
        self.snap_box = QtWidgets.QCheckBox("Caler la fréquence sur la grille du pas"); self.snap_box.setChecked(bool(cfg.get("snap", True)))
        self.snap_box.toggled.connect(self.on_snap)
        self.on_snap(self.snap_box.isChecked())
        form.addRow(self.snap_box)
        self.show_bands = QtWidgets.QCheckBox("Plan de bandes sur le spectre"); self.show_bands.setChecked(True)
        self.show_bands.toggled.connect(self.on_show_bands)
        self.lock = QtWidgets.QCheckBox("Verrouiller la fréquence")
        self.lock.toggled.connect(lambda on: self.freq.setEnabled(not on))
        self.fft_size = QtWidgets.QComboBox()
        self.fft_size.addItem("Automatique (suit le zoom)", None)
        for n in FFT_SIZES:
            self.fft_size.addItem("{:,} points".format(n).replace(",", " "), n)
        self.fft_window = QtWidgets.QComboBox()
        for name, (hint, _) in WINDOWS.items():
            self.fft_window.addItem(name, name)
            self.fft_window.setItemData(self.fft_window.count() - 1, hint, QtCore.Qt.ItemDataRole.ToolTipRole)
        self.fft_window.setCurrentIndex(max(self.fft_window.findData(cfg.get("fft_window", DEFAULT_WINDOW)), 0))
        self.fft_window.currentIndexChanged.connect(lambda i: self.cfg.__setitem__("fft_window", self.fft_window.currentData()))
        self.fft_info = QtWidgets.QLabel("")
        self.fft_info.setWordWrap(True)
        self.show_if = QtWidgets.QCheckBox("Fenêtre « Spectre FI »")
        self.show_af = QtWidgets.QCheckBox("Fenêtre « Spectre audio »")
        self.show_if.setChecked(bool(cfg.get("show_if", False))); self.show_af.setChecked(bool(cfg.get("show_af", False)))
        self.show_if.toggled.connect(self.on_show_analysis); self.show_af.toggled.connect(self.on_show_analysis)
        self.on_show_analysis()
        self.show_mem = QtWidgets.QCheckBox("Noms des mémoires sur le spectre")
        self.show_mem.setChecked(bool(cfg.get("show_mem", True)))
        self.show_mem.toggled.connect(lambda on: self.cfg.__setitem__("show_mem", bool(on)))
        self.peak_hold = QtWidgets.QCheckBox("Maintien des crêtes (courbe orange)")
        self.peak_hold.toggled.connect(lambda *_: self.trace.reset())
        self.avg_n = QtWidgets.QSpinBox(); self.avg_n.setRange(1, 50); self.avg_n.setValue(int(cfg.get("avg_n", 1)))
        self.avg_n.setSpecialValueText("Aucun")
        self.avg_n.setToolTip("Moyenne de plusieurs spectres successifs : le bruit se lisse et les signaux faibles ressortent.")
        self.avg_n.valueChanged.connect(lambda v: self.cfg.__setitem__("avg_n", int(v)))
        self.s_cal = QtWidgets.QSpinBox(); self.s_cal.setRange(-120, 60); self.s_cal.setSuffix(" dB")
        self.s_cal.setValue(int(cfg.get("s_cal", -20)))
        self.s_cal.setToolTip("Niveau en dBm = niveau mesuré (dB sous la pleine échelle) + cette valeur.\n"
                              "Elle dépend du gain : sans étalonnage, le S-mètre ne donne que des comparaisons.")
        self.s_cal.valueChanged.connect(lambda v: self.cfg.__setitem__("s_cal", int(v)))
        form.addRow(self.show_bands); form.addRow(self.show_mem); form.addRow(self.lock)
        form.addRow(self.show_if); form.addRow(self.show_af)
        form.addRow(self.peak_hold); form.addRow("Moyennage", self.avg_n); form.addRow("Étalonnage S-mètre", self.s_cal)
        form.addRow("Finesse (FFT)", self.fft_size); form.addRow("Fenêtre", self.fft_window); form.addRow(self.fft_info)
        self.wf_auto = QtWidgets.QCheckBox("Cascade : niveaux automatiques"); self.wf_auto.setChecked(bool(cfg.get("wf_auto", True)))
        self.wf_min = QtWidgets.QSpinBox(); self.wf_max = QtWidgets.QSpinBox()
        for w, value in ((self.wf_min, cfg.get("wf_min", -110)), (self.wf_max, cfg.get("wf_max", -50))):
            w.setRange(-180, 20); w.setSuffix(" dB"); w.setSingleStep(2); w.setValue(int(value))
            w.valueChanged.connect(self.on_display)
        self.wf_auto.toggled.connect(self.on_wf_auto)
        self.wf_min.setEnabled(not self.wf_auto.isChecked()); self.wf_max.setEnabled(not self.wf_auto.isChecked())
        form.addRow(self.wf_auto); form.addRow("Cascade min (bleu)", self.wf_min); form.addRow("Cascade max (rouge)", self.wf_max)
        self._floor_db = -100.0
        self.on_display()

        box = Section("Anti-bruit"); form = QtWidgets.QFormLayout(box); noise_tab.addWidget(box)
        noise_tab.addStretch(1)
        self.nb = QtWidgets.QCheckBox("NB : noise blanker (impulsions)"); self.nb.toggled.connect(self.on_noise)
        self.nb_thr = self.slider(20, 100, 40, self.on_noise)                 # seuil x10
        self.nr = QtWidgets.QCheckBox("NR : réduction de bruit"); self.nr.toggled.connect(self.on_noise)
        self.ekf_enable = QtWidgets.QCheckBox("Filtre de Kalman étendu (EKF)")
        self.ekf_enable.setToolTip("Kalman pour systèmes non linéaires (saturation, phase)")
        self.ekf_enable.toggled.connect(self.on_ekf)
        self.kalman_nr = QtWidgets.QCheckBox("Filtre de Kalman")
        self.kalman_nr.setToolTip("Lissage adaptatif du signal (réduction de bruit)")
        self.kalman_nr.toggled.connect(self.on_kalman)
        self.kalman_q = QtWidgets.QDoubleSpinBox()
        self.kalman_q.setRange(1e-7, 1e-2)
        self.kalman_q.setDecimals(6)
        self.kalman_q.setSingleStep(1e-5)
        self.kalman_q.setValue(1e-5)
        self.kalman_q.setToolTip("Bruit de processus Q (petit = lissage fort)")
        self.kalman_q.valueChanged.connect(self.on_kalman_params)
        self.kalman_r = QtWidgets.QDoubleSpinBox()
        self.kalman_r.setRange(1e-5, 1.0)
        self.kalman_r.setDecimals(4)
        self.kalman_r.setSingleStep(1e-3)
        self.kalman_r.setValue(1e-3)
        self.kalman_r.setToolTip("Bruit de mesure R (petit = suit le signal)")
        self.kalman_r.valueChanged.connect(self.on_kalman_params)
        self.neural_nr = QtWidgets.QCheckBox("Débruitage neuronal (RNNoise)")
        self.neural_nr.setToolTip("Supprime le bruit de fond par réseau de neurones")
        self.neural_nr.toggled.connect(self.on_neural_nr)
        self.nr_method = QtWidgets.QComboBox()
        for label, key in NR_METHODS:
            self.nr_method.addItem(label, key)
        self.nr_method.currentIndexChanged.connect(self.on_noise)
        self.nr_level = self.slider(6, 30, 18, self.on_noise)
        self.anf = QtWidgets.QCheckBox("ANF : notch automatique (porteuses)"); self.anf.toggled.connect(self.on_noise)
        form.addRow(self.nb); form.addRow("Seuil NB", self.nb_thr); form.addRow(self.nr)
        form.addRow("Méthode", self.nr_method); form.addRow("Réduction (dB)", self.nr_level)
        form.addRow(self.ekf_enable); form.addRow(self.kalman_nr)
        form.addRow("Kalman Q", self.kalman_q)
        form.addRow("Kalman R", self.kalman_r); form.addRow(self.neural_nr); form.addRow(self.anf)

        # Contour (façonnage de bande passante, style FT-710)
        self.contour_on = QtWidgets.QCheckBox("Contour (façonnage)")
        self.contour_on.setChecked(bool(cfg.get("contour_on", False)))
        self.contour_on.toggled.connect(self.on_contour)
        self.contour_freq = QtWidgets.QSpinBox(); self.contour_freq.setRange(400, 3000); self.contour_freq.setSuffix(" Hz")
        self.contour_freq.setValue(int(cfg.get("contour_freq", 1500))); self.contour_freq.setSingleStep(50)
        self.contour_freq.valueChanged.connect(self.on_contour)
        self.contour_gain = QtWidgets.QSpinBox(); self.contour_gain.setRange(-15, 15); self.contour_gain.setSuffix(" dB")
        self.contour_gain.setValue(int(cfg.get("contour_gain", 6))); self.contour_gain.valueChanged.connect(self.on_contour)
        self.contour_width = QtWidgets.QSpinBox(); self.contour_width.setRange(200, 2000); self.contour_width.setSuffix(" Hz")
        self.contour_width.setValue(int(cfg.get("contour_width", 800))); self.contour_width.valueChanged.connect(self.on_contour)
        form.addRow(self.contour_on)
        form.addRow("Fréquence Contour", self.contour_freq)
        form.addRow("Gain Contour", self.contour_gain)
        form.addRow("Largeur Contour", self.contour_width)

        # APF pour CW
        self.apf_on = QtWidgets.QCheckBox("APF CW (pic audio)")
        self.apf_on.setChecked(bool(cfg.get("apf_on", False)))
        self.apf_on.toggled.connect(self.on_apf)
        self.apf_width = QtWidgets.QSpinBox(); self.apf_width.setRange(30, 400); self.apf_width.setSuffix(" Hz")
        self.apf_width.setValue(int(cfg.get("apf_width", 100))); self.apf_width.setSingleStep(10)
        self.apf_width.valueChanged.connect(self.on_apf)
        self.apf_gain = QtWidgets.QSpinBox(); self.apf_gain.setRange(3, 24); self.apf_gain.setSuffix(" dB")
        self.apf_gain.setValue(int(cfg.get("apf_gain", 12))); self.apf_gain.valueChanged.connect(self.on_apf)
        form.addRow(self.apf_on)
        form.addRow("Largeur APF", self.apf_width)
        form.addRow("Gain APF", self.apf_gain)

        # Notch profond dans la boucle AGC
        self.deep_notch_on = QtWidgets.QCheckBox("Notch AGC (profond)")
        self.deep_notch_on.setChecked(bool(cfg.get("deep_notch_on", False)))
        self.deep_notch_on.toggled.connect(self.on_deep_notch)
        self.deep_notch_freq = QtWidgets.QSpinBox(); self.deep_notch_freq.setRange(100, 4000)
        self.deep_notch_freq.setSuffix(" Hz"); self.deep_notch_freq.setSingleStep(10)
        self.deep_notch_freq.setValue(int(cfg.get("deep_notch_freq", 1000)))
        self.deep_notch_freq.valueChanged.connect(self.on_deep_notch)
        self.deep_notch_q = QtWidgets.QSpinBox(); self.deep_notch_q.setRange(10, 100)
        self.deep_notch_q.setValue(int(cfg.get("deep_notch_q", 30)))
        self.deep_notch_q.valueChanged.connect(self.on_deep_notch)
        form.addRow(self.deep_notch_on)
        form.addRow("Fréquence Notch AGC", self.deep_notch_freq)
        form.addRow("Q Notch AGC", self.deep_notch_q)

        box = Section("Enregistrement"); lay = QtWidgets.QHBoxLayout(box); panel.addWidget(box)
        self.rec_audio = QtWidgets.QPushButton("● Audio"); self.rec_audio.setCheckable(True)
        self.rec_audio.toggled.connect(lambda on: self.on_record(on, False))
        self.rec_iq = QtWidgets.QPushButton("● I/Q"); self.rec_iq.setCheckable(True)
        self.rec_iq.toggled.connect(lambda on: self.on_record(on, True))
        lay.addWidget(self.rec_audio); lay.addWidget(self.rec_iq)
        lay.addWidget(QtWidgets.QLabel("Format I/Q :"))
        self.iq_fmt = QtWidgets.QComboBox()
        self.iq_fmt.addItems(["WAV 16 bits", "CF32 (SDR#)"])
        self.iq_fmt.setCurrentIndex(1 if self.cfg.get("iq_format") == "cf32" else 0)
        self.iq_fmt.currentIndexChanged.connect(self.on_iq_fmt)
        lay.addWidget(self.iq_fmt)
        panel.addStretch(1)

        box = Section("Mémoires"); lay = QtWidgets.QVBoxLayout(box); mem_tab.addWidget(box, 1)
        self.mem_group = QtWidgets.QComboBox()
        self.mem_group.currentIndexChanged.connect(lambda *_: self.refresh_memories(keep_groups=True))
        self._mem_rows = []
        lay.addWidget(self.mem_group)
        self.mem = QtWidgets.QListWidget(); self.mem.itemDoubleClicked.connect(self.on_recall)
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Mémoriser"); add.clicked.connect(self.on_mem_add)
        rem = QtWidgets.QPushButton("Supprimer"); rem.clicked.connect(self.on_mem_del)
        row.addWidget(add); row.addWidget(rem)
        row2 = QtWidgets.QHBoxLayout()
        imp = QtWidgets.QPushButton("Importer CSV…"); imp.clicked.connect(self.on_mem_import)
        exp = QtWidgets.QPushButton("Exporter CSV…"); exp.clicked.connect(self.on_mem_export)
        row2.addWidget(imp); row2.addWidget(exp)
        lay.addWidget(self.mem); lay.addLayout(row); lay.addLayout(row2)
        self.refresh_memories()

        box = Section("Scanner"); form = QtWidgets.QFormLayout(box); mem_tab.addWidget(box)
        self.scan_mem = QtWidgets.QPushButton("Scanner les mémoires"); self.scan_mem.setCheckable(True)
        self.scan_mem.toggled.connect(lambda on: self.on_scan(on, self.scan_mem))
        self.scan_lo, self.scan_hi = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for w, value in ((self.scan_lo, 145_000_000), (self.scan_hi, 146_000_000)):
            w.setRange(0, 1_800_000_000); w.setGroupSeparatorShown(True); w.setSuffix(" Hz")
            w.setSingleStep(100000); w.setValue(value)
        self.scan_range = QtWidgets.QPushButton("Scanner la plage"); self.scan_range.setCheckable(True)
        self.scan_range.toggled.connect(lambda on: self.on_scan(on, self.scan_range))
        hint = QtWidgets.QLabel("Le scanner s'arrête quand le niveau du canal dépasse le squelch, "
                                "et avance avec le pas et le mode choisis.")
        hint.setWordWrap(True)
        form.addRow(self.scan_mem); form.addRow("De", self.scan_lo); form.addRow("À", self.scan_hi)
        self.log_hits = QtWidgets.QCheckBox("Journal d'activité (fichier CSV du jour)")
        form.addRow(self.scan_range); form.addRow(self.log_hits); form.addRow(hint)

        box = Section("Radiodiffusion FM"); form = QtWidgets.QFormLayout(box); info_tab.addWidget(box)
        self.rds_ps = QtWidgets.QLabel("—")
        f = self.rds_ps.font(); f.setPointSize(16); f.setBold(True); self.rds_ps.setFont(f)
        self.rds_rt = QtWidgets.QLabel(""); self.rds_rt.setWordWrap(True)
        self.rds_misc = QtWidgets.QLabel("")
        self.stereo_label = QtWidgets.QLabel("—")
        form.addRow("Station (RDS)", self.rds_ps); form.addRow("Texte", self.rds_rt)
        form.addRow("PI / genre", self.rds_misc); form.addRow("Son", self.stereo_label)
        box = Section("FM étroite"); form = QtWidgets.QFormLayout(box); info_tab.addWidget(box)
        self.ctcss_label = QtWidgets.QLabel("—")
        self.dcs_label = QtWidgets.QLabel("—")
        form.addRow("Tonalité CTCSS", self.ctcss_label); form.addRow("Code DCS", self.dcs_label)
        box = Section("Spectre audio"); lay = QtWidgets.QVBoxLayout(box); info_tab.addWidget(box, 1)
        self.audio_plot = pg.PlotWidget()
        self.audio_plot.setFixedHeight(170)
        self.audio_plot.setLabel("bottom", "Fréquence", units="Hz")
        self.audio_plot.setYRange(-100, 0)
        self.audio_plot.setMouseEnabled(x=False, y=False)
        self.audio_curve = self.audio_plot.plot(pen=pg.mkPen("c"))
        self.audio_response = self.audio_plot.plot(pen=pg.mkPen("y", width=2))  # réponse des filtres
        lay.addWidget(self.audio_plot)

        # ---- ADS-B
        box = Section("ADS-B (1090 MHz)"); lay = QtWidgets.QVBoxLayout(box); dec_tab.addWidget(box)
        self.adsb_on = QtWidgets.QCheckBox("Activer le décodage ADS-B")
        self.adsb_on.toggled.connect(self.on_adsb)
        self.adsb_list = QtWidgets.QListWidget()
        self.adsb_list.setMaximumHeight(150)
        self.adsb_clear = QtWidgets.QPushButton("Effacer la liste ADS-B")
        self.adsb_clear.clicked.connect(lambda: self.adsb_list.clear())
        lay.addWidget(self.adsb_on)
        lay.addWidget(self.adsb_list)
        lay.addWidget(self.adsb_clear)
        # Carte (si WebEngine disponible)
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            self.adsb_map = QWebEngineView()
            self.adsb_map.setMinimumHeight(300)
            map_path = os.path.join(os.path.dirname(__file__), "assets", "adsb_map.html")
            if os.path.exists(map_path):
                self.adsb_map.setUrl(QtCore.QUrl.fromLocalFile(map_path))
            lay.addWidget(self.adsb_map)
        except Exception:
            self.adsb_map = None
            lay.addWidget(QtWidgets.QLabel("(Carte OpenStreetMap : installer PySide6-WebEngine)"))

        # ---- ACARS
        box = Section("ACARS (131.825 MHz)"); lay = QtWidgets.QVBoxLayout(box); dec_tab.addWidget(box)
        self.acars_on = QtWidgets.QCheckBox("Activer ACARS")
        self.acars_on.toggled.connect(self.on_acars)
        self.acars_list = QtWidgets.QListWidget()
        self.acars_list.setMaximumHeight(120)
        self.acars_clear = QtWidgets.QPushButton("Effacer ACARS")
        self.acars_clear.clicked.connect(lambda: self.acars_list.clear())
        lay.addWidget(self.acars_on)
        lay.addWidget(self.acars_list)
        lay.addWidget(self.acars_clear)

        # ---- AIS
        box = Section("AIS (bateaux, 161.975 / 162.025 MHz)"); lay = QtWidgets.QVBoxLayout(box); dec_tab.addWidget(box)
        self.ais_on = QtWidgets.QCheckBox("Activer AIS")
        self.ais_on.toggled.connect(self.on_ais)
        self.ais_list = QtWidgets.QListWidget()
        self.ais_list.setMaximumHeight(120)
        self.ais_clear = QtWidgets.QPushButton("Effacer AIS")
        self.ais_clear.clicked.connect(lambda: self.ais_list.clear())
        lay.addWidget(self.ais_on)
        lay.addWidget(self.ais_list)
        lay.addWidget(self.ais_clear)

        # ---- onglet Décodage
        box = Section("Décodage numérique (RTTY, PSK, CW, packet)"); form = QtWidgets.QFormLayout(box)
        dec_tab.addWidget(box)
        self.digi_kind = QtWidgets.QComboBox()
        for label, key in (("Aucun", None), ("RTTY 45,45 bauds", "RTTY45"), ("RTTY 50 bauds", "RTTY50"),
                           ("PSK31", "PSK31"), ("PSK63", "PSK63"), ("CW (morse)", "CW"),
                           ("Packet HF 300 bauds (AX.25)", "PACKET300"),
                           ("APRS / packet VHF 1200 bauds (en NFM)", "APRS1200"),
                           ("NAVTEX (518 et 490 kHz)", "NAVTEX"), ("Fax météo HF", "FAX"),
                           ("SSTV (images des radioamateurs)", "SSTV"),
                           ("Hellschreiber (texte en image)", "HELL"), ("WSPR (balises, autour de 1500 Hz)", "WSPR"), ("FT8 (signaux faibles)", "FT8"), ("FT4 (signaux faibles)", "FT4"), ("JS8Call (expérimental)", "JS8"), ("Olivia (HF robuste)", "OLIVIA"), ("Contestia (concours)", "CONTESTIA"), ("JT65 (DX faible)", "JT65"), ("DMR (numérique)", "DMR"), ("D-STAR (numérique)", "DSTAR"), ("C4FM (Yaesu)", "C4FM"), ("M17 (libre)", "M17"), ("Golay (24,12)", "GOLAY"),
                           ("Heure : DCF77 (77,5 kHz)", "DCF77"), ("Heure : Allouis (162 kHz)", "ALS162")):
            self.digi_kind.addItem(label, key)
        self.digi_kind.currentIndexChanged.connect(self.on_decoders)
        self.digi_kind.currentIndexChanged.connect(self._on_digi_kind_changed)
        self.digi_center = QtWidgets.QSpinBox(); self.digi_center.setRange(300, 3000); self.digi_center.setSuffix(" Hz")
        self.digi_center.setSingleStep(10); self.digi_center.setValue(1000); self.digi_center.setKeyboardTracking(False)
        self.digi_center.valueChanged.connect(self.on_decoders)
        self.digi_rev = QtWidgets.QCheckBox("Inverser mark / space (RTTY)"); self.digi_rev.toggled.connect(self.on_decoders)
        self.cw_auto_speed = QtWidgets.QCheckBox("CW : vitesse automatique"); self.cw_auto_speed.setChecked(True)
        self.cw_wpm = QtWidgets.QSpinBox(); self.cw_wpm.setRange(5, 60); self.cw_wpm.setValue(20)
        self.cw_wpm.setSuffix(" mots/min"); self.cw_wpm.setEnabled(False)
        self.cw_auto_thr = QtWidgets.QCheckBox("CW : seuil automatique"); self.cw_auto_thr.setChecked(True)
        self.cw_thr = QtWidgets.QSpinBox(); self.cw_thr.setRange(-160, 0); self.cw_thr.setValue(-60)
        self.cw_thr.setSuffix(" dB"); self.cw_thr.setEnabled(False)
        self.cw_auto_speed.toggled.connect(self.on_cw_auto)
        self.cw_auto_thr.toggled.connect(self.on_cw_auto)
        self.cw_wpm.valueChanged.connect(self.on_decoders)
        self.cw_thr.valueChanged.connect(self.on_decoders)
        hint = QtWidgets.QLabel("Le texte décodé et le tracé du niveau s'affichent au-dessus du spectre ; les images "
                                "(fax, SSTV) sous la cascade. Fax et SSTV : en USB, cliquer au milieu du signal.")
        hint.setWordWrap(True)
        form.addRow("Mode", self.digi_kind); form.addRow("Fréquence audio", self.digi_center)
        form.addRow(self.digi_rev)
        form.addRow(self.cw_auto_speed); form.addRow("Vitesse", self.cw_wpm)
        self.digi_clear2 = QtWidgets.QPushButton("Effacer le texte décodé")
        self.digi_clear2.clicked.connect(lambda *_: self.on_digi_clear())
        form.addRow(self.cw_auto_thr); form.addRow("Seuil", self.cw_thr); form.addRow(self.digi_clear2); form.addRow(hint)
        self._digi_buf = ""
        self.audio_marker = pg.InfiniteLine(pos=1000, angle=90, movable=False, pen=pg.mkPen("m"))
        self.audio_plot.addItem(self.audio_marker)
        self.audio_marker.setVisible(False)
        box = Section("POCSAG"); lay = QtWidgets.QVBoxLayout(box); dec_tab.addWidget(box, 1)
        self.pocsag = QtWidgets.QCheckBox("Décoder (mode NFM, 512 / 1200 / 2400 bit/s)")
        self.pocsag.toggled.connect(self.on_decoders)
        self.pocsag_list = QtWidgets.QListWidget(); self.pocsag_list.setWordWrap(True)
        self.pocsag_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.pocsag_list.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.pocsag_clear = QtWidgets.QPushButton("Effacer les messages POCSAG")
        self.pocsag_clear.clicked.connect(lambda *_: self.on_pocsag_clear())
        lay.addWidget(self.pocsag); lay.addWidget(self.pocsag_list, 1); lay.addWidget(self.pocsag_clear)

        # ---- onglet Outils
        box = Section("Filtres audio"); form = QtWidgets.QFormLayout(box); tools_tab.addWidget(box)
        self.bass = self.slider(-12, 12, 0, self.on_fx)
        self.treble = self.slider(-12, 12, 0, self.on_fx)
        self.notch_list = QtWidgets.QListWidget(); self.notch_list.setMaximumHeight(70)
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Ajouter un notch…"); add.clicked.connect(self.on_notch_add)
        rem = QtWidgets.QPushButton("Supprimer"); rem.clicked.connect(self.on_notch_del)
        row.addWidget(add); row.addWidget(rem)
        form.addRow("Graves (dB)", self.bass); form.addRow("Aigus (dB)", self.treble)
        form.addRow(self.notch_list); form.addRow(row)

        box = Section("Sortie audio"); form = QtWidgets.QFormLayout(box); tools_tab.addWidget(box)
        self.audio_device = QtWidgets.QComboBox()
        self.audio_device.addItem("Périphérique par défaut", "")
        try:
            import sounddevice
            for dev in sounddevice.query_devices():
                if dev["max_output_channels"] > 0 and self.audio_device.findData(dev["name"]) < 0:
                    self.audio_device.addItem(dev["name"], dev["name"])
        except Exception:
            pass
        self.audio_device.setCurrentIndex(max(self.audio_device.findData(cfg.get("audio_device", "")), 0))
        self.audio_device.currentIndexChanged.connect(self.on_audio_device)
        self.raw = QtWidgets.QCheckBox("Sortie brute du discriminateur (NFM)")
        self.raw.setToolTip("Pour un décodeur externe (DSD+, etc.) branché sur un câble audio virtuel :\n"
                            "bande plate jusqu'à 8 kHz, sans désaccentuation, squelch ni anti-bruit.")
        self.raw.toggled.connect(self.on_decoders)
        form.addRow(self.audio_device); form.addRow(self.raw)

        box = Section("Second récepteur (VFO B)"); form = QtWidgets.QFormLayout(box); tools_tab.addWidget(box)
        self.vfo2_on = QtWidgets.QCheckBox("Écouter une seconde fréquence")
        self.vfo2_freq = QtWidgets.QSpinBox(); self.vfo2_freq.setRange(0, 1_800_000_000)
        self.vfo2_freq.setGroupSeparatorShown(True); self.vfo2_freq.setSuffix(" Hz"); self.vfo2_freq.setSingleStep(1000)
        self.vfo2_freq.setKeyboardTracking(False); self.vfo2_freq.setValue(int(cfg["vfo_hz"]))
        self.vfo2_mode = QtWidgets.QComboBox()
        for m in ("AM", "LSB", "USB", "CW", "NFM", "WFM"):
            self.vfo2_mode.addItem(m, m)
        self.vfo2_mode.setCurrentIndex(2)
        self.vfo2_volume = self.slider(0, 100, 50, self.on_vfo2)
        self.vfo2_split = QtWidgets.QCheckBox("A à gauche, B à droite (sinon mélangés)")
        self.vfo2_copy = QtWidgets.QPushButton("B = fréquence et mode actuels")
        self.vfo2_copy.clicked.connect(lambda *_: self.on_vfo2_copy())
        for w in (self.vfo2_on, self.vfo2_split):
            w.toggled.connect(self.on_vfo2)
        self.vfo2_freq.valueChanged.connect(self.on_vfo2); self.vfo2_mode.currentIndexChanged.connect(self.on_vfo2)
        form.addRow(self.vfo2_on); form.addRow("Fréquence B", self.vfo2_freq); form.addRow("Mode B", self.vfo2_mode)
        form.addRow("Volume B", self.vfo2_volume); form.addRow(self.vfo2_split); form.addRow(self.vfo2_copy)
        self.vfo2_line = pg.InfiniteLine(pos=0, angle=90, movable=True, pen=pg.mkPen("c", width=2))
        self.vfo2_line.sigPositionChangeFinished.connect(lambda: self.vfo2_freq.setValue(int(round(self.vfo2_line.value()))))
        self.plot.addItem(self.vfo2_line)
        self.vfo2_line.setVisible(False)

        box = Section("TimeShift (écoute différée)"); form = QtWidgets.QFormLayout(box); tools_tab.addWidget(box)
        self.ts_on = QtWidgets.QCheckBox("Garder les dernières secondes en mémoire")
        self.ts_delay = self.slider(0, 0, 0, self.on_timeshift)
        self.ts_label = QtWidgets.QLabel("En direct")
        self.ts_live = QtWidgets.QPushButton("Revenir au direct")
        self.ts_live.clicked.connect(lambda *_: self.ts_delay.setValue(0))
        self.ts_on.toggled.connect(self.on_timeshift)
        form.addRow(self.ts_on); form.addRow("Retard", self.ts_delay); form.addRow(self.ts_label); form.addRow(self.ts_live)

        box = Section("Commande réseau"); form = QtWidgets.QFormLayout(box); tools_tab.addWidget(box)
        self.net_box = QtWidgets.QCheckBox("Serveur rigctl, 127.0.0.1:4532 (Gpredict)")
        self.net_box.toggled.connect(self.on_net)
        form.addRow(self.net_box)

        box = Section("CAT série"); form = QtWidgets.QFormLayout(box); tools_tab.addWidget(box)
        self.cat_box = QtWidgets.QCheckBox("Serveur CAT texte sur port COM (Gpredict série, OmniRig)")
        self.cat_box.toggled.connect(self.on_cat)
        self.cat_port = QtWidgets.QLineEdit(str(self.cfg.get("cat_port", "COM3")))
        self.cat_baud = QtWidgets.QComboBox()
        self.cat_baud.addItems(["4800", "9600", "19200", "38400", "57600", "115200"])
        self.cat_baud.setCurrentText(str(self.cfg.get("cat_baud", 9600)))
        self.cat_port.editingFinished.connect(self.on_cat_params)
        self.cat_baud.currentTextChanged.connect(self.on_cat_params)
        form.addRow(self.cat_box); form.addRow("Port COM", self.cat_port); form.addRow("Baud", self.cat_baud)

        box = Section("Enregistrements programmés"); lay = QtWidgets.QVBoxLayout(box); tools_tab.addWidget(box, 1)
        self.job_list = QtWidgets.QListWidget()
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Programmer…"); add.clicked.connect(self.on_job_add)
        rem = QtWidgets.QPushButton("Supprimer"); rem.clicked.connect(self.on_job_del)
        row.addWidget(add); row.addWidget(rem)
        lay.addWidget(self.job_list); lay.addLayout(row)
        self.refresh_jobs()

        box = Section("Raccourcis clavier"); lay = QtWidgets.QVBoxLayout(box); tools_tab.addWidget(box)
        keys = QtWidgets.QLabel("\n".join("%s : %s" % (KEY_NAMES.get(k, k), text) for k, text in SHORTCUTS))
        keys.setWordWrap(True)
        lay.addWidget(keys)
        self.install_shortcuts()
        side_layout.addStretch(1)
        self.status = self.statusBar()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(50)
        self.mode_buttons[cfg["mode"] if cfg["mode"] in MODES else "USB"].setChecked(True)
        self._set_freq_display(cfg["vfo_hz"])
        self.freq_digits.set_value(cfg["vfo_hz"])
        self.status.showMessage("Cliquer sur « ▶ Airspy » ou ouvrir un fichier I/Q.")

    # ------------------------------------------------------------ utilitaires
    def slider(self, lo, hi, value, slot):
        s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        s.setRange(lo, hi); s.setValue(value); s.valueChanged.connect(slot)
        return s

    def fail(self, title, exc):
        QtWidgets.QMessageBox.warning(self, title, str(exc))

    def _set_freq_display(self, hz):
        self._updating = True
        self.freq.setValue(int(round(hz)))
        self._updating = False

    def current_mode(self):
        return self.radio.rx.mode if self.ready() else self.cfg["mode"]

    def ready(self):
        return self.radio.source is not None and self.radio.rx is not None

    # ------------------------------------------------------- source / marche
    def toggle_airspy(self):
        r = self.radio
        if r.running and r.source.kind == "airspy":
            r.stop()
            self.btn_start.setText("▶ Airspy")
            return
        try:
            if r.source is None or r.source.kind != "airspy":
                from .sources.airspy import AirspySource, AirspyError
                try:
                    src = AirspySource(self.cfg.get("dll_path") or None)
                except AirspyError as e:
                    if "introuvable" not in str(e):
                        raise
                    path, _ = QtWidgets.QFileDialog.getOpenFileName(
                        self, "Indiquer airspy.dll (64 bits)", "", "Bibliothèque Airspy (*.dll *.so* *.dylib)")
                    if not path:
                        return
                    src = AirspySource(path)
                    self.cfg["dll_path"] = path
                rate = self.cfg["sample_rate"] if self.cfg["sample_rate"] in src.sample_rates else src.sample_rates[0]
                src.set_sample_rate(rate)
                src.set_sv_ppm(self.sv_ppm.value())
                src.set_shift(self.current_shift())
                src.set_spyverter(self.spyverter.isChecked())
                src.set_bias_tee(self.bias_tee.isChecked())
                src.set_gain(self.gain.value(), self.gain_mode.currentData())
                vfo = self.freq.value()
                r.decimation = self.pick_decimation(src.sample_rate)
                src.set_center(vfo - 0.125 * src.sample_rate / r.decimation)
                r.set_source(src)
                self.after_source(vfo)
            r.start()
            self.btn_start.setText("■ Arrêter")
        except Exception as e:
            self.fail("Airspy", e)

    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Fichier I/Q", self.cfg["record_dir"], "I/Q (*.wav *.cf32)")
        if path:
            self.open_iq(path)

    def open_iq(self, path, rate=None):
        try:
            if path.lower().endswith(".cf32") and not rate:            # .cf32 n'a pas d'en-tête : demander le débit
                rate = int(self.cfg.get("file_rate") or 0)
                if not rate:
                    rate, ok = QtWidgets.QInputDialog.getInt(self, "Fichier .cf32", "Débit en S/s :",
                                                             1_000_000, 1_000, 100_000_000, 1_000)
                    if not ok:
                        return
                    self.cfg["file_rate"] = rate
                    save_config(self.cfg)
            src = IQFileSource(path, loop=True, realtime=True, sample_rate=rate)
            self.radio.decimation = self.pick_decimation(src.sample_rate)
            self.radio.set_source(src)
            self.after_source(src.center_hz)
            self.radio.start()
            self.btn_start.setText("▶ Airspy")
            self.status.showMessage("Lecture en boucle : %s" % os.path.basename(path))
            self.play_pause.setChecked(False)
        except Exception as e:
            self.fail("Fichier I/Q", e)

    def after_source(self, vfo_hz):
        """Réapplique les réglages de l'interface à la nouvelle source."""
        r = self.radio
        self._updating = True
        self.rate.clear()
        for sr in r.source.sample_rates:
            self.rate.addItem("%.3g MS/s" % (sr / 1e6), sr)
        self.rate.setCurrentIndex(max(self.rate.findData(r.source.sample_rate), 0))
        self.decim.clear()
        for d in allowed_decimations(r.source.sample_rate):
            self.decim.addItem("Aucune" if d == 1 else str(d), d)
        self.decim.setCurrentIndex(max(self.decim.findData(r.decimation), 0))
        self._updating = False
        self.player.setVisible(self.is_file())
        self.trace.reset()
        fs = r.sample_rate
        self.span_label.setText("%s kHz (%s kS/s)" % (("%.6g" % (0.8 * fs / 1e3 if r.decimation > 1 else fs / 1e3)).replace(".", ","),
                                                      ("%.6g" % (fs / 1e3)).replace(".", ",")))
        self.on_mode(self.cfg["mode"] if self.cfg["mode"] in MODES else "USB")   # le mode choisi, même pour une nouvelle source
        self.on_audio(); self.on_noise(); self.on_fx(); self.on_decoders()
        self.on_filter_shape(); self.on_vfo2(); self.on_timeshift(); self.show_filter()
        self.on_contour(); self.on_apf(); self.on_deep_notch()
        r.tune(vfo_hz)
        self._set_freq_display(r.vfo_hz)
        self.frame_view(full=True)

    def frame_view(self, full=False, shift=0.0):
        """Cadre le spectre : toute la bande reçue, ou le même zoom décalé avec la fréquence centrale."""
        r = self.radio
        fs = r.sample_rate
        lo, hi = r.center_hz - fs / 2, r.center_hz + fs / 2
        vb = self.plot.getViewBox()
        vb.setLimits(xMin=None, xMax=None)
        if full:
            self.plot.setXRange(lo, hi, padding=0)
        else:
            x0, x1 = vb.viewRange()[0]
            self.plot.setXRange(x0 + shift, x1 + shift, padding=0)
        vb.setLimits(xMin=lo, xMax=hi)                         # on ne peut pas sortir de la bande reçue
        self._shown_center = r.center_hz

    def follow_vfo(self):
        """Fait glisser la vue quand la fréquence écoutée approche d'un bord ou sort de l'écran.

        Appelé seulement quand la fréquence change : on reste libre de faire glisser la vue ailleurs à la main.
        """
        r = self.radio
        self._followed_vfo = r.vfo_hz
        fs = r.sample_rate
        lo, hi = r.center_hz - fs / 2, r.center_hz + fs / 2
        x0, x1 = self.plot.getViewBox().viewRange()[0]
        width = x1 - x0
        if width >= fs * 0.999:
            return                                             # toute la bande est déjà visible
        margin = 0.1 * width
        if r.vfo_hz < x0 + margin:
            x0 = r.vfo_hz - margin
        elif r.vfo_hz > x1 - margin:
            x0 = r.vfo_hz + margin - width
        else:
            return
        x0 = min(max(x0, lo), hi - width)                      # sans dépasser la bande reçue
        self.plot.setXRange(x0, x0 + width, padding=0)

    # --------------------------------------------------------------- accord
    def tune(self, hz):
        """Accord demandé par l'utilisateur (clic, saisie, mémoire) : arrête le scanner."""
        self.stop_scan()
        if self.lock.isChecked():
            self._set_freq_display(self.radio.vfo_hz if self.ready() else self.freq.value())
            return
        if not self.ready():
            self._set_freq_display(hz)
            return
        try:
            self.radio.tune(hz)
        except Exception as e:
            self.fail("Accord", e)
        self._set_freq_display(self.radio.vfo_hz)

    def on_digit(self, delta):
        """Clic ou molette sur un chiffre du grand afficheur : la fréquence varie de l'unité de ce chiffre."""
        self.tune(float(max(0, self.freq.value() + delta)))

    def on_side(self, on):
        self.cfg["side_visible"] = bool(on)
        self.side.setVisible(bool(on))

    def on_mute(self, on):
        self.mute.setText("🔇" if on else "🔊")
        self.on_audio()

    # ------------------------------------------------- clavier et souris
    def install_shortcuts(self):
        actions = {"Right": lambda: self.key_step(1), "Left": lambda: self.key_step(-1),
                   "Ctrl+Right": lambda: self.key_step(10), "Ctrl+Left": lambda: self.key_step(-10),
                   "PgUp": lambda: self.key_mode(1), "PgDown": lambda: self.key_mode(-1),
                   "Space": lambda: self.mute.setChecked(not self.mute.isChecked()),
                   "M": lambda: self.on_mem_add(),
                   "R": lambda: self.rec_audio.setChecked(not self.rec_audio.isChecked()),
                   "L": lambda: self.lock.setChecked(not self.lock.isChecked()),
                   "P": lambda: self.play_pause.setChecked(not self.play_pause.isChecked()),
                   "F1": lambda: self.show_shortcuts()}
        self.key_actions = actions
        self._shortcuts = []
        for key, _ in SHORTCUTS:
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.activated.connect(actions[key])
            self._shortcuts.append(sc)

    def key_step(self, steps):
        self.freq.stepBy(steps)

    def key_mode(self, direction):
        mode = self.current_mode()
        i = MODE_ORDER.index(mode) if mode in MODE_ORDER else 0
        self.on_mode(MODE_ORDER[(i + direction) % len(MODE_ORDER)])

    def show_shortcuts(self):
        QtWidgets.QMessageBox.information(
            self, "Raccourcis clavier", "\n".join("%s : %s" % (KEY_NAMES.get(k, k), text) for k, text in SHORTCUTS))

    def band_dragging(self):
        """Vrai pendant que l'utilisateur tire la zone du filtre ou l'un de ses bords."""
        try:
            return self.band.moving is True or any(line.moving is True for line in self.band.lines)
        except TypeError:
            return False

    def set_band(self, lo, hi):
        """Place la zone du filtre depuis le programme (sans que cela compte pour un geste de l'utilisateur)."""
        if self.band_dragging():
            return
        self._band_set = True
        self.band.setRegion((lo, hi))
        self._band_set = False

    def on_band_signal(self):
        if not self._band_set:
            lo, hi = self.band.getRegion()
            self.on_band_dragged(float(lo), float(hi))

    def on_band_dragged(self, lo, hi):
        """La zone du filtre a été tirée à la souris : milieu = nouvelle fréquence, bord = nouvelle largeur."""
        if not self.ready():
            return
        r = self.radio
        old_lo, old_hi = r.passband()
        if abs((hi - lo) - (old_hi - old_lo)) < 1.0:              # même largeur : toute la zone a été déplacée
            hz = r.vfo_hz + (lo - old_lo)
            self.tune(snap(hz, self.step.currentData() or 1) if self.snap_box.isChecked() else round(hz))
            return
        shift = old_lo - (r.vfo_hz + r.rx.lo)                     # décalage d'affichage (note de la CW)
        new_lo, new_hi = lo - r.vfo_hz - shift, hi - r.vfo_hz - shift
        if new_hi - new_lo >= 50:
            r.rx.set_filter(new_lo, new_hi)
        self.show_filter()

    # ------------------------------------------------------ étalonnage
    def on_calibrate(self):
        """Calcule la correction SV PPM à partir d'une porteuse dont l'utilisateur donne la fréquence exacte."""
        r = self.radio
        if not self.ready() or r.source.kind != "airspy" or not self.spyverter.isChecked():
            self.fail("Étalonnage", "L'étalonnage règle la correction du SpyVerter : il faut l'Airspy en marche, "
                                    "case SpyVerter cochée.")
            return
        true_hz, ok = QtWidgets.QInputDialog.getInt(
            self, "Étalonnage", "Fréquence exacte de la porteuse, en hertz :", int(snap(r.vfo_hz, 1000)), 0, 60_000_000)
        if not ok:
            return
        found = r.measure_carrier(float(true_hz))
        if found is None or found[1] < 15.0:
            self.fail("Étalonnage", "Aucune porteuse nette à moins de 3 kHz de cette fréquence.")
            return
        ppm = self.sv_ppm.value() + (found[0] - true_hz) / 120.0      # 1 ppm de l'oscillateur 120 MHz = 120 Hz
        if abs(ppm) > 100.0:
            self.fail("Étalonnage", "Correction calculée invraisemblable (%.1f ppm) : mauvaise porteuse ?" % ppm)
            return
        self.sv_ppm.setValue(round(ppm, 2))
        self.status.showMessage("Étalonnage : porteuse mesurée à %+.1f Hz, correction SV PPM réglée à %.2f ppm"
                                % (found[0] - true_hz, self.sv_ppm.value()))

    # ---------------------------------------------------- lecteur de fichier
    def is_file(self):
        return self.radio.source is not None and self.radio.source.kind == "file"

    def on_play_pause(self, paused):
        self.play_pause.setText("▶" if paused else "⏸")
        if self.is_file():
            self.radio.source.paused = bool(paused)

    def on_play_seek(self, value):
        if not self._player_set and self.is_file():
            self.radio.source.seek(value / 1000.0)

    def update_player(self):
        src = self.radio.source
        pos = src.position()
        self._player_set = True
        self.play_pos.setValue(int(round(1000 * pos)))
        self._player_set = False
        now, total = int(pos * src.duration_s), int(src.duration_s)
        self.play_time.setText("%d:%02d / %d:%02d" % (now // 60, now % 60, total // 60, total % 60))

    # ------------------------------------------------------ images reçues
    def clear_picture(self):
        self.pic_rows = []
        self.pic_img.clear()

    def picture(self):
        """L'image en cours (tableau uint8), cadrage manuel appliqué ; None s'il n'y a rien."""
        if not self.pic_rows:
            return None
        img = np.array(self.pic_rows)
        if self.pic_shift.value():
            img = np.roll(img, -int(self.pic_shift.value() / 100.0 * img.shape[1]), axis=1)
        return img

    def show_picture(self):
        img = self.picture()
        if img is None:
            return
        img = img[-PIC_SHOWN_ROWS:]
        if img.ndim == 2:
            self.pic_img.setImage(img, autoLevels=False, levels=(0, 255))
        else:
            self.pic_img.setImage(img, autoLevels=False)
        self.pic_plot.getViewBox().autoRange(padding=0)

    def save_picture(self, kind=None):
        img = self.picture()
        if img is None or len(img) < 8:
            return None
        kind = kind or (self.radio.rx.digi_kind if self.ready() and self.radio.rx.digi_kind in ("FAX", "SSTV", "HELL") else "image")
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        path = os.path.join(self.cfg["record_dir"], "PyPhantom_%s_%s.png" % (stamp, kind))
        try:
            os.makedirs(self.cfg["record_dir"], exist_ok=True)
            write_png(path, img)
        except OSError as e:
            self.fail("Image", e)
            return None
        self.status.showMessage("Image enregistrée : %s" % path)
        return path

    def update_picture(self, d):
        """Récupère les lignes d'image produites par le décodeur (fax ou SSTV) et les affiche."""
        if d is not self._pic_decoder:                           # nouveau décodeur : on repart d'une image vide
            self._pic_decoder, self.pic_id, self.pic_done = d, d.image_id, d.finished
            self.clear_picture()
        if d.image_id != self.pic_id:                            # une nouvelle image commence
            self.pic_id = d.image_id
            self.clear_picture()
        if hasattr(d, "tape"):                                   # Hellschreiber : une bande qui défile, redessinée en entier
            self.pic_rows = list(d.tape())
            self.show_picture()
        rows = d.take_rows()
        if rows:
            if self.pic_rows and len(rows[0]) != len(self.pic_rows[0]):
                self.pic_rows = []
            self.pic_rows = (self.pic_rows + rows)[-PIC_MAX_ROWS:]
            self.show_picture()
        if d.finished != self.pic_done:                          # image terminée : enregistrée d'office
            self.pic_done = d.finished
            self.save_picture(self.radio.rx.digi_kind)
        self.pic_title.setText("Image reçue : %d lignes" % len(self.pic_rows))

    def visible_memories(self, lo, hi):
        """Mémoires dont la fréquence tombe dans la bande affichée : [(Hz, nom)], triées, en nombre limité."""
        found = sorted((float(m["hz"]), m["name"]) for m in self.cfg["memories"] if lo <= float(m["hz"]) <= hi)
        return found[:MEM_LABELS]

    def update_mem_labels(self):
        x0, x1 = self.plot.getViewBox().viewRange()[0]
        shown = self.visible_memories(x0, x1) if self.show_mem.isChecked() else []
        top, height = float(self.offset_db.value()), float(self.range_db.value())
        for i, item in enumerate(self.mem_labels):
            if i < len(shown):
                item.setText(shown[i][1])
                item.setPos(shown[i][0], top - (i % 3) * height / 16.0)     # trois hauteurs : les noms voisins se gênent moins
            item.setVisible(i < len(shown))

    def on_freq(self, value):
        if not self._updating:
            self.tune(float(value))

    def on_wheel_steps(self, steps):
        """Molette sur le spectre ou la cascade : la fréquence avance du pas choisi, calée sur sa grille."""
        self.freq.stepBy(steps)

    def on_step(self, index):
        hz = self.step.itemData(index)
        self.freq.grid = float(hz)
        self.freq.setSingleStep(max(int(round(hz)), 1))
        self.cfg["step_hz"] = hz

    def on_click(self, ev, widget):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        pos = ev.scenePos()
        vb = widget.getViewBox()
        if vb.sceneBoundingRect().contains(pos):
            hz = vb.mapSceneToView(pos).x()
            exact = self.signal_to_vfo(hz) if self.ready() else None
            if exact is not None:                                  # décodeur HF actif : on vise le signal cliqué
                self.tune(round(exact))
            else:
                self.tune(snap(hz, self.step.currentData() or 1) if self.snap_box.isChecked() else round(hz))

    def on_mode(self, mode):
        self.mode_buttons[mode].setChecked(True)
        self.cfg["mode"] = mode
        if self.ready():
            try:
                self.radio.set_mode(mode)
            except ValueError as e:
                self.fail("Mode", e)
                return
            self.show_filter()
            self.on_noise()
            self.on_decoders()

    def show_filter(self):
        """Recopie dans l'interface le filtre en vigueur (largeur et bords)."""
        rx = self.radio.rx
        self._updating = True
        self.bw.setValue(int(rx.hi - rx.lo))
        self.edge_lo.setValue(int(round(rx.lo))); self.edge_hi.setValue(int(round(rx.hi)))
        self._updating = False

    def on_bw(self, width):
        if self._updating or not self.ready():
            return
        rx = self.radio.rx
        if rx.mode == "USB":
            rx.set_filter(100.0, 100.0 + width)
        elif rx.mode == "LSB":
            rx.set_filter(-100.0 - width, -100.0)
        elif rx.mode == "CW":
            rx.set_filter(rx.cw_pitch - width / 2, rx.cw_pitch + width / 2)
        else:
            rx.set_filter(-width / 2, width / 2)
        self.show_filter()

    def on_edges(self, *_):
        """Filtre asymétrique : les deux bords sont réglés séparément."""
        if self._updating or not self.ready():
            return
        if self.edge_hi.value() - self.edge_lo.value() >= 50:
            self.radio.rx.set_filter(self.edge_lo.value(), self.edge_hi.value())
        self.show_filter()

    def on_filter_shape(self, *_):
        if self.ready():
            self.radio.rx.set_filter_shape(self.filter_window.currentData(), self.filter_order.value() or None)

    # ------------------------------------------------------------ réglages
    def on_rate(self, index):
        r = self.radio
        if self._updating or not self.ready() or r.source.kind != "airspy" or index < 0:
            return
        was_running, vfo = r.running, r.vfo_hz
        try:
            r.stop()
            r.source.set_sample_rate(self.rate.itemData(index))
            self.cfg["sample_rate"] = r.source.sample_rate
            src, r.source = r.source, None                      # même appareil, nouveau débit
            r.decimation = self.pick_decimation(src.sample_rate)
            src.set_center(vfo - 0.125 * src.sample_rate / r.decimation)
            r.set_source(src)
            self.after_source(vfo)
            if was_running:
                r.start()
        except Exception as e:
            self.fail("Débit", e)

    def pick_decimation(self, source_rate):
        """La décimation mémorisée si ce débit la permet, sinon la plus forte permise en dessous."""
        wanted = int(self.cfg.get("decimation", 1))
        return max([d for d in allowed_decimations(source_rate) if d <= wanted] or [1])

    def on_decim(self, index):
        r = self.radio
        if self._updating or not self.ready() or index < 0:
            return
        was_running, vfo = r.running, r.vfo_hz
        try:
            self.cfg["decimation"] = int(self.decim.itemData(index))
            r.stop()
            r.set_decimation(self.cfg["decimation"])
            if r.source.kind == "airspy":                       # recentrer : la bande reçue est plus étroite
                r.source.set_center(vfo - 0.125 * r.sample_rate)
                r.center_hz = float(r.source.center_hz)
            self.after_source(vfo)
            if was_running:
                r.start()
        except Exception as e:
            self.fail("Décimation", e)

    def on_gain(self, *_):
        self.cfg["gain"], self.cfg["gain_mode"] = self.gain.value(), self.gain_mode.currentData()
        if self.ready() and self.radio.source.kind == "airspy":
            try:
                self.radio.source.set_gain(self.gain.value(), self.gain_mode.currentData())
            except Exception as e:
                self.fail("Gain", e)

    def on_sv_ppm(self, ppm):
        self.cfg["sv_ppm"] = float(ppm)
        if self.ready() and self.radio.source.kind == "airspy":
            try:
                self.radio.source.set_sv_ppm(ppm)
                self.radio.set_center(self.radio.center_hz)      # même affichage, fréquence matérielle corrigée
            except Exception as e:
                self.fail("Correction SV PPM", e)

    def on_spyverter(self, on):
        self.cfg["spyverter"] = bool(on)
        self.sv_ppm.setEnabled(bool(on))
        if self.ready() and self.radio.source.kind == "airspy":
            try:
                self.radio.source.set_spyverter(on)
                self.radio.set_center(self.radio.vfo_hz - 0.125 * self.radio.sample_rate)
                self.after_source(self.radio.vfo_hz)
            except Exception as e:
                self.fail("SpyVerter", e)

    def on_bias_tee(self, on):
        self.cfg["bias_tee"] = bool(on)
        if self.ready() and self.radio.source.kind == "airspy":
            try:
                self.radio.source.set_bias_tee(on)
            except Exception as e:
                self.fail("Bias-tee", e)

    def on_audio(self, *_):
        self.cfg["volume"] = self.volume.value() / 100.0
        if self.ready():
            rx = self.radio.rx
            rx.volume = 0.0 if self.mute.isChecked() else self.volume.value() / 100.0
            rx.squelch_db = float(self.squelch.value())
            rx.set_agc(self.agc.isChecked())
            rx.stereo_enabled = self.stereo.isChecked()
            rx.set_am_sync(self.am_sync.isChecked(), self.anti_fading.isChecked())
            rx.set_binaural(self.binaural.isChecked())
            if rx.cw_pitch != self.cw_pitch.value():
                rx.set_cw_pitch(self.cw_pitch.value())
                if rx.mode == "CW":
                    self.radio.tune(self.radio.vfo_hz)
                    self.show_filter()
                self.on_apf()  # recentrer l'APF sur la nouvelle note
        self.cfg["am_sync"], self.cfg["cw_pitch"] = self.am_sync.isChecked(), self.cw_pitch.value()
        self.cfg["anti_fading"], self.cfg["binaural"] = self.anti_fading.isChecked(), self.binaural.isChecked()

    def on_iq_options(self, *_):
        self.cfg["invert_iq"], self.cfg["remove_dc"] = self.invert.isChecked(), self.dc_fix.isChecked()
        self.radio.invert_iq, self.radio.remove_dc = self.invert.isChecked(), self.dc_fix.isChecked()

    def current_shift(self):
        return float(self.shift_hz.value()) if self.shift_on.isChecked() else 0.0

    def on_shift(self, *_):
        self.cfg["shift_on"], self.cfg["shift_hz"] = self.shift_on.isChecked(), self.shift_hz.value()
        if self.ready() and self.radio.source.kind == "airspy":
            try:
                self.radio.source.set_shift(self.current_shift())
                self.radio.set_center(self.radio.vfo_hz - 0.125 * self.radio.sample_rate)
                self.after_source(self.radio.vfo_hz)
            except Exception as e:
                self.fail("Décalage", e)

    def on_snap(self, on):
        self.cfg["snap"] = bool(on)
        self.freq.snap_enabled = bool(on)

    # ---------------------------------------------------- outils et décodage
    def on_vfo2(self, *_):
        """Second récepteur : mêmes échantillons, autre fréquence et autre mode."""
        if not self.ready():
            return
        r = self.radio
        try:
            if self.vfo2_on.isChecked() != (r.rx2 is not None):
                r.vfo2_hz = float(self.vfo2_freq.value())
                r.enable_second(self.vfo2_on.isChecked())
            r.split_ears = self.vfo2_split.isChecked()
            if r.rx2 is not None:
                if r.rx2.mode != self.vfo2_mode.currentData():
                    r.rx2.set_mode(self.vfo2_mode.currentData())
                r.rx2.volume = self.vfo2_volume.value() / 100.0
                r.tune2(float(self.vfo2_freq.value()))
        except ValueError as e:
            self.fail("VFO B", e)
        self.vfo2_line.setVisible(r.rx2 is not None)

    def on_vfo2_copy(self):
        mode = self.current_mode()
        self.vfo2_mode.setCurrentIndex(max(self.vfo2_mode.findData(mode), 0))
        self.vfo2_freq.setValue(self.freq.value())

    def on_timeshift(self, *_):
        if self.ready():
            self.radio.set_timeshift(self.ts_on.isChecked(), float(self.ts_delay.value()) if self.ts_on.isChecked() else 0.0)
        if not self.ts_on.isChecked():
            self.ts_label.setText("En direct")

    def on_fx(self, *_):
        if self.ready():
            self.radio.rx.set_audio_fx(self.notch_values, self.bass.value(), self.treble.value())

    def on_notch_add(self):
        hz, ok = QtWidgets.QInputDialog.getInt(self, "Notch", "Fréquence audio à supprimer (Hz) :", 1000, 50, 6000)
        if ok:
            self.notch_values.append((hz, 100))
            self.refresh_notches()

    def on_notch_del(self):
        row = self.notch_list.currentRow()
        if 0 <= row < len(self.notch_values):
            del self.notch_values[row]
        elif self.notch_values:
            self.notch_values.pop()
        self.refresh_notches()

    def refresh_notches(self):
        self.notch_list.clear()
        for hz, width in self.notch_values:
            self.notch_list.addItem("%d Hz (largeur %d Hz)" % (hz, width))
        self.on_fx()

    def _on_digi_kind_changed(self, *_):
        kind = self.digi_kind.currentData()
        if kind in ("FT8", "FT4", "WSPR"):
            if self.digi_center.value() < 1200:
                self.digi_center.setValue(1500)        # milieu de la bande FT8/WSPR
        elif self.digi_center.value() == 1500:
            self.digi_center.setValue(1000)             # retour au centre par défaut des autres modes

    def on_decoders(self, *_):
        if self.ready():
            rx = self.radio.rx
            rx.set_pocsag(self.pocsag.isChecked())
            rx.set_raw_output(self.raw.isChecked())
            rx.set_digi(self.digi_kind.currentData(), self.digi_center.value(), self.digi_rev.isChecked())
            rx.set_cw_options(None if self.cw_auto_speed.isChecked() else self.cw_wpm.value(),
                              None if self.cw_auto_thr.isChecked() else float(self.cw_thr.value()))
        tetra = self.current_mode() == "TETRA"
        kind = self.digi_kind.currentData()
        self.digi_panel.setVisible((not tetra) and kind is not None)
        # Visibilité par mode : ne montrer que ce qui concerne le décodeur actif
        is_ft = kind in ("FT8", "FT4")
        is_wspr = kind == "WSPR"
        is_psk = kind in ("PSK31", "PSK63")
        is_image = kind in ("FAX", "SSTV", "HELL")
        is_tetra = tetra
        show_aviation = kind in ("METAR", "SIGMET") or (kind is None and not tetra)

        # FT8/FT4 : liste de messages uniquement
        if hasattr(self, "ft8_frame"):
            self.ft8_frame.setVisible(is_ft)
            if is_ft:
                self.ft8_title.setText(f"{kind} Messages")
        # Texte de decodage (miroir aussi pour FT8/FT4)
        self.digi_out.setVisible(not is_image and not is_tetra)
        # METAR / SIGMET : seulement si mode aviation ou aucun mode
        if hasattr(self, "metar_label"):
            self.metar_label.setVisible(show_aviation)
            self.metar_list.setVisible(show_aviation)
            self.metar_clear.setVisible(show_aviation)
        if hasattr(self, "sigmet_label"):
            self.sigmet_label.setVisible(show_aviation)
            self.sigmet_list.setVisible(show_aviation)
            self.sigmet_clear.setVisible(show_aviation)
        # TETRA dock — forcer l'affichage
        if hasattr(self, "tetra_dock"):
            if is_tetra:
                self.tetra_dock.setVisible(True)
                self.tetra_dock.show()
                self.tetra_dock.raise_()
                self.digi_panel.setVisible(True)  # garder le panneau digi pour les contrôles
            else:
                self.tetra_dock.hide()
                self.tetra_dock.setVisible(False)
        # Timeslot : seulement TETRA
        if hasattr(self, "tetra_ts_box"):
            self.tetra_ts_box.setVisible(is_tetra)
        # Images : FAX, SSTV, HELL
        self.pic_panel.setVisible(is_image and not is_tetra)
        # Constellation : PSK uniquement
        self.const_plot.setVisible(is_psk and not is_tetra)
        # Titre
        self.digi_title.setText("Analyse TETRA" if is_tetra else "Décodeur %s" % (kind or ""))
        if not hasattr(self, "tetra_ts"):
            self.tetra_ts_box = QtWidgets.QWidget()
            ts_lay = QtWidgets.QHBoxLayout(self.tetra_ts_box)
            ts_lay.addWidget(QtWidgets.QLabel("Timeslot :"))
            self.tetra_voice = QtWidgets.QCheckBox("Audio vocal (ACELP)")
            self.tetra_voice.setToolTip("Active le décodage vocal TETRA (nécessite sdecoder)")
            self.tetra_voice.toggled.connect(self.on_tetra_voice)
            self.tetra_ts = QtWidgets.QComboBox()
            self.tetra_ts.addItems(["Auto", "TS1", "TS2", "TS3", "TS4"])
            self.tetra_ts.setCurrentIndex(0)
            self.tetra_ts.currentIndexChanged.connect(self.on_tetra_ts)
            ts_lay.addWidget(self.tetra_voice)
            ts_lay.addWidget(self.tetra_ts)
            self.tetra_bits_label = QtWidgets.QLabel("Bits : 0")
            ts_lay.addWidget(self.tetra_bits_label)
            if hasattr(self, "digi_panel"):
                self.digi_panel.layout().addWidget(self.tetra_ts_box)
    def on_cw_auto(self, *_):
        """Passage en manuel : on part de la valeur que le décodeur utilisait en automatique."""
        d = self.radio.rx.digi if self.ready() else None
        is_cw = d is not None and self.radio.rx.digi_kind == "CW"
        if not self.cw_auto_speed.isChecked() and not self.cw_wpm.isEnabled() and is_cw:
            self.cw_wpm.setValue(int(round(d.wpm())))
        if not self.cw_auto_thr.isChecked() and not self.cw_thr.isEnabled() and is_cw and d.thr_db is not None:
            self.cw_thr.setValue(int(round(d.thr_db)))
        self.cw_wpm.setEnabled(not self.cw_auto_speed.isChecked())
        self.cw_thr.setEnabled(not self.cw_auto_thr.isChecked())
        self.on_decoders()

    def _log_ft8(self, message, dist_str="", mode="FT8"):
        """Enregistre un décodage FT8/FT4 dans un fichier journal."""
        import os
        from datetime import datetime
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        freq = getattr(self.radio, 'vfo_hz', 0) if self.ready() else 0
        line = f"{ts}  {freq/1e6:.3f} MHz  {mode}  {message}{dist_str}\n"
        today = datetime.utcnow().strftime("%Y%m%d")
        fname = f"{mode}_{today}.log"
        locations = [
            os.path.join(os.path.expanduser("~"), "Desktop", fname),
            os.path.join(os.path.expanduser("~"), fname),
            os.path.join(os.getcwd(), fname),
        ]
        for path in locations:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
                break
            except Exception:
                continue

    def on_ft8_locator_click(self, item):
        """Ouvre la carte du locator cliqué."""
        loc = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if loc:
            url = f"https://www.levinecentral.com/ham/grid_square.php?&Grid={loc}&Zoom=13&sm=y"
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))

    def on_digi_clear(self, *_):
        """Vide la fenêtre de texte décodé, ainsi que les caractères reçus mais pas encore affichés."""
        self._digi_buf = ""
        if self.ready():
            self.radio.rx.digi_text.clear()
        self.digi_out.clear()
        self.digi_out.setPlainText("")
        if hasattr(self, "ft8_list"):
            self.ft8_list.clear()

    def signal_to_vfo(self, hz):
        """Fréquence à afficher pour qu'un signal vu à `hz` tombe sur la fréquence audio du décodeur HF."""
        rx = self.radio.rx
        if rx.digi is None or rx.mode not in ("USB", "LSB"):
            return None
        # fax et SSTV : milieu du signal d'image ; WSPR : milieu de la bande des balises
        center = {"FAX": 1900.0, "SSTV": 1900.0, "WSPR": 1500.0}.get(rx.digi_kind, rx.digi_center)
        return hz - center if rx.mode == "USB" else hz + center

    def on_pocsag_clear(self, *_):
        self.pocsag_list.clear()
        if self.ready():
            self.radio.rx.pocsag_messages.clear()
        self._pocsag_seen = 0

    def on_audio_device(self, index):
        self.cfg["audio_device"] = self.audio_device.itemData(index) or ""
        try:
            self.radio.restart_audio()
        except Exception as e:
            self.fail("Sortie audio", e)

    def on_net(self, on):
        if self.net is not None:
            self.net.close()
            self.net = None
        if on:
            try:
                self.net = RigCtlServer(self.net_state, port=self.net_port)
            except OSError as e:
                self.net_box.setChecked(False)
                self.fail("Commande réseau", "Port %d indisponible : %s" % (self.net_port, e))

    def net_state(self):
        if self.ready():
            return self.radio.vfo_hz, self.radio.rx.mode, self.radio.rx.hi - self.radio.rx.lo, self.radio.rx.level_db
        return float(self.freq.value()), self.cfg["mode"], 0.0, -100.0

    def on_cat(self, on):
        if self.cat is not None:
            self.cat.close()
            self.cat = None
        if on:
            port = self.cat_port.text().strip() or "COM3"
            try:
                self.cat = SerialCatServer(self.net_state, port=port, baud=int(self.cat_baud.currentText()))
                if self.cat.error:
                    raise OSError(self.cat.error)
            except Exception as e:
                self.cat_box.setChecked(False)
                self.fail("CAT série", "%s : %s" % (port, e))

    def on_cat_params(self, *_):
        self.cfg["cat_port"] = self.cat_port.text().strip() or "COM3"
        self.cfg["cat_baud"] = int(self.cat_baud.currentText())
        save_config(self.cfg)
        if self.cat_box.isChecked():                      # rouvre le port avec les nouveaux réglages
            self.on_cat(False)
            self.on_cat(True)

    def on_iq_fmt(self, idx):
        self.cfg["iq_format"] = "cf32" if idx == 1 else "wav"
        save_config(self.cfg)

    def run_net_commands(self):
        for srv in (self.net, self.cat):
            while srv is not None and not srv.commands.empty():
                kind, value = srv.commands.get_nowait()
                if kind == "freq":
                    self.tune(value)
                elif kind == "mode":
                    self.on_mode(value)

    # --------------------------------------------- enregistrements programmés
    def refresh_jobs(self):
        self.job_list.clear()
        for job in self.cfg["jobs"]:
            self.job_list.addItem(planner.job_label(job))

    def on_job_add(self):
        mode = self.current_mode()
        start = datetime.datetime.now() + datetime.timedelta(minutes=5)
        default = "%s;30;%d;%s;audio" % (start.strftime(planner.TIME_FORMAT), self.freq.value(), mode)
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Programmer un enregistrement", "Début ; durée en minutes ; fréquence en Hz ; mode ; audio ou iq",
            text=default)
        if not ok:
            return
        try:
            self.cfg["jobs"].append(planner.parse_job(text, MODES))
            self.refresh_jobs()
            save_config(self.cfg)
        except ValueError as e:
            self.fail("Programmation", e)

    def on_job_del(self):
        row = self.job_list.currentRow()
        if 0 <= row < len(self.cfg["jobs"]):
            del self.cfg["jobs"][row]
            self.refresh_jobs()
            save_config(self.cfg)

    def run_scheduler(self, now=None):
        for action, job in self.scheduler.tick(now):
            try:
                if action == "start":
                    if not self.radio.running:
                        raise ValueError("réception arrêtée à l'heure prévue")
                    self.lock.setChecked(False)
                    self.on_mode(job["mode"])
                    self.tune(float(job["hz"]))
                    path = self.radio.start_recording(self.cfg["record_dir"], job["iq"], fmt=self.cfg.get("iq_format") if job["iq"] else None)
                    self.status.showMessage("Enregistrement programmé : %s" % path)
                elif self.ready():
                    self.radio.stop_recording(job["iq"])
                    self.status.showMessage("Enregistrement programmé terminé.")
            except Exception as e:
                self.status.showMessage("Enregistrement programmé impossible : %s" % e)
            self.refresh_jobs()
            save_config(self.cfg)

    # ------------------------------------------------- mémoires : fichiers
    def on_mem_import(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Importer des mémoires", "", "CSV (*.csv *.txt)")
        if not path:
            return
        try:
            new = planner.import_memories(path, MODES)
            self.cfg["memories"].extend(new)
            self.refresh_memories()
            save_config(self.cfg)
            self.status.showMessage("%d mémoires importées." % len(new))
        except Exception as e:
            self.fail("Import", e)

    def on_mem_export(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Exporter les mémoires", "memoires.csv", "CSV (*.csv)")
        if path:
            try:
                planner.export_memories(path, self.cfg["memories"])
            except Exception as e:
                self.fail("Export", e)

    # ------------------------------------------------- échelle et cascade
    def on_display(self, *_):
        """Curseurs de droite et niveaux de la cascade."""
        for key, widget in (("contrast", self.contrast), ("range_db", self.range_db), ("offset_db", self.offset_db),
                            ("wf_min", self.wf_min), ("wf_max", self.wf_max)):
            self.cfg[key] = widget.value()
        top = float(self.offset_db.value())
        self.plot.setYRange(top - self.range_db.value(), top, padding=0)
        if self.ready() and self.zoom.value() != getattr(self, "_zoom_applied", 0):
            self._zoom_applied = self.zoom.value()
            r = self.radio
            fs = r.sample_rate
            width = max(fs * 0.5 ** (self.zoom.value() / 8.0), min(2000.0, fs))    # 8 crans = 2 fois plus serré
            lo, hi = r.center_hz - fs / 2, r.center_hz + fs / 2
            x0 = min(max(r.vfo_hz - width / 2, lo), hi - width)
            self.plot.setXRange(x0, x0 + width, padding=0)

    def on_wf_auto(self, auto):
        """Passage en manuel : on part des niveaux que l'automatique utilisait."""
        self.cfg["wf_auto"] = bool(auto)
        if not auto and not self.wf_min.isEnabled():
            lo, hi = self.auto_levels()
            self.wf_min.setValue(int(round(lo))); self.wf_max.setValue(int(round(lo)) + int(round(hi - lo)))
        self.wf_min.setEnabled(not auto); self.wf_max.setEnabled(not auto)
        self.on_display()

    def waterfall_levels(self):
        """(niveau affiché en bleu nuit, niveau affiché en rouge/blanc), en dB."""
        if self.wf_auto.isChecked():
            return self.auto_levels()
        lo, hi = float(self.wf_min.value()), float(self.wf_max.value())
        return lo, max(hi, lo + 5.0)

    def auto_levels(self):
        """Niveaux calés sur le bruit de fond ; le contraste règle la dynamique affichée."""
        span = 100.0 - 0.8 * self.contrast.value()                     # contraste 50 -> 60 dB de dynamique
        return self._floor_db - 5.0, self._floor_db - 5.0 + span

    def on_show_analysis(self, *_):
        self.cfg["show_if"], self.cfg["show_af"] = self.show_if.isChecked(), self.show_af.isChecked()
        self.if_plot.setVisible(self.show_if.isChecked())
        self.af_plot.setVisible(self.show_af.isChecked())

    def update_filter_response(self):
        """Affiche la réponse combinée des filtres audio sur le spectre audio."""
        if not self.ready() or not hasattr(self, "audio_response"):
            return
        rx = self.radio.rx
        fs = rx.fs2
        freqs = np.linspace(0, min(8000, fs / 2), 400)
        response = np.ones_like(freqs)

        # Contour
        if rx.contour_enabled and rx.contour.bq.enabled:
            # Approximation simple de la réponse du peaking
            f0 = rx.contour.freq
            gain = 10 ** (rx.contour.gain_db / 20.0)
            q = rx.contour.q
            bw = f0 / q
            mask = np.abs(freqs - f0) < bw
            response[mask] *= gain

        # APF
        if rx.mode == "CW" and rx.apf_enabled and rx.apf.bq.enabled:
            f0 = rx.apf.center
            gain = 10 ** (rx.apf.bq.b0 / 1.0)  # rough
            # better: use configured gain
            apf_gain = 10 ** (12.0 / 20.0)  # default, refine later if needed
            bw = rx.apf.width
            mask = np.abs(freqs - f0) < bw / 2
            response[mask] *= apf_gain

        # Deep notch
        if rx.deep_notch_enabled and rx.deep_notch.bq.enabled:
            f0 = rx.deep_notch.freq
            bw = f0 / rx.deep_notch.q
            mask = np.abs(freqs - f0) < bw
            response[mask] *= 0.01  # -40 dB

        db = 20 * np.log10(np.maximum(response, 1e-4))
        self.audio_response.setData(freqs, db)

    def update_analysis(self):
        """Rafraîchit les fenêtres « Spectre FI » et « Spectre audio » quand elles sont affichées."""
        r = self.radio
        rx = r.rx
        if self.show_if.isChecked():
            spec = r.if_spectrum(1024, self.fft_window.currentData())
            if spec is not None:
                offsets, db, half = spec
                self.if_curve.setData(offsets, db)
                shift = -rx.cw_pitch if rx.mode == "CW" else 0.0
                self.if_band.setRegion((rx.lo + shift, rx.hi + shift))
                self.if_plot.setXRange(-half, half, padding=0)
                top = float(db.max())
                self.if_plot.setYRange(min(float(np.median(db)) - 15, top - 40), top + 10, padding=0)
        if self.show_af.isChecked():
            spec = r.audio_spectrum(4096)
            if spec is not None:
                limit = 16000 if rx.mode == "WFM" else (6000 if rx.mode in ("AM", "NFM") else 4000)
                keep = spec[0] <= limit
                self.af_curve.setData(spec[0][keep], spec[1][keep])
                self.af_plot.setXRange(0, limit, padding=0)
                self.af_plot.setYRange(-110, 0, padding=0)

    def on_show_bands(self, on):
        for item in self.band_items:
            item.setVisible(bool(on))

    # --------------------------------------------------------------- scanner
    def on_scan(self, on, button):
        if not on:
            if not (self.scan_mem.isChecked() or self.scan_range.isChecked()):
                self.scanner.stop()
            return
        other = self.scan_range if button is self.scan_mem else self.scan_mem
        other.setChecked(False)
        try:
            if not self.ready() or not self.radio.running:
                raise ValueError("Démarrer d'abord la réception.")
            if self.lock.isChecked():
                raise ValueError("La fréquence est verrouillée.")
            if self.squelch.value() <= -150:
                raise ValueError("Régler d'abord le squelch (onglet Réception) juste au-dessus du niveau "
                                 "du canal sans signal : c'est le seuil d'arrêt du scanner.")
            threshold = float(self.squelch.value())
            if button is self.scan_mem:
                self.scanner.start([(m["hz"], m["mode"]) for m in self.shown_memories()], threshold)
            else:
                self.scanner.start_range(self.scan_lo.value(), self.scan_hi.value(),
                                         self.step.currentData(), threshold)
        except Exception as e:
            button.setChecked(False)
            self.fail("Scanner", e)

    def stop_scan(self):
        self.scanner.stop()
        self.scan_mem.setChecked(False)
        self.scan_range.setChecked(False)

    def on_contour(self, *_):
        if not self.ready():
            return
        self.cfg["contour_on"] = self.contour_on.isChecked()
        self.cfg["contour_freq"] = self.contour_freq.value()
        self.cfg["contour_gain"] = self.contour_gain.value()
        self.cfg["contour_width"] = self.contour_width.value()
        self.radio.rx.set_contour(
            self.contour_on.isChecked(),
            self.contour_freq.value(),
            self.contour_gain.value(),
            self.contour_width.value()
        )

    def on_apf(self, *_):
        if not self.ready():
            return
        self.cfg["apf_on"] = self.apf_on.isChecked()
        self.cfg["apf_width"] = self.apf_width.value()
        self.cfg["apf_gain"] = self.apf_gain.value()
        # Centré sur la note CW actuelle
        center = self.radio.rx.cw_pitch if self.radio.rx.mode == "CW" else 700
        self.radio.rx.set_apf(
            self.apf_on.isChecked(),
            center,
            self.apf_width.value(),
            self.apf_gain.value()
        )

    def on_deep_notch(self, *_):
        if not self.ready():
            return
        self.cfg["deep_notch_on"] = self.deep_notch_on.isChecked()
        self.cfg["deep_notch_freq"] = self.deep_notch_freq.value()
        self.cfg["deep_notch_q"] = self.deep_notch_q.value()
        self.radio.rx.set_deep_notch(
            self.deep_notch_on.isChecked(),
            self.deep_notch_freq.value(),
            self.deep_notch_q.value()
        )

    def on_ekf(self, *_):
        if self.ready() and hasattr(self.radio.rx, "ekf"):
            self.radio.rx.ekf.set_enabled(self.ekf_enable.isChecked())

    def on_kalman(self, *_):
        if self.ready() and hasattr(self.radio.rx, "kalman"):
            self.radio.rx.kalman.set_enabled(self.kalman_nr.isChecked())
            self.on_kalman_params()

    def on_kalman_params(self, *_):
        if self.ready() and hasattr(self.radio.rx, "kalman"):
            self.radio.rx.kalman.kf.q = self.kalman_q.value()
            self.radio.rx.kalman.kf.r = self.kalman_r.value()

    def on_neural_nr(self, *_):
        if self.ready() and hasattr(self.radio.rx, "neural_nr"):
            self.radio.rx.neural_nr.set_enabled(self.neural_nr.isChecked())

    def on_noise(self, *_):
        if self.ready():
            rx = self.radio.rx
            rx.set_nb(self.nb.isChecked(), self.nb_thr.value() / 10.0)
            rx.set_nr(self.nr.isChecked(), self.nr_method.currentData(), float(self.nr_level.value()))
            rx.set_anf(self.anf.isChecked())

    def on_record(self, on, iq):
        if not self.ready() or not self.radio.running:
            return
        try:
            if on:
                path = self.radio.start_recording(self.cfg["record_dir"], iq, fmt=self.cfg.get("iq_format") if iq else None)
                self.status.showMessage("Enregistrement : %s" % path)
            else:
                self.radio.stop_recording(iq)
                self.status.showMessage("Enregistrement terminé.")
        except Exception as e:
            self.fail("Enregistrement", e)

    # ------------------------------------------------------------- mémoires
    def shown_memories(self):
        """Les mémoires du groupe choisi (toutes si aucun groupe n'est choisi)."""
        group = self.mem_group.currentData()
        return [m for m in self.cfg["memories"] if group is None or m.get("group", "") == group]

    def refresh_memories(self, keep_groups=False):
        if not keep_groups:                                    # la liste des groupes a pu changer
            current = self.mem_group.currentData()
            self._updating_groups = True
            self.mem_group.clear()
            self.mem_group.addItem("Tous les groupes", None)
            for g in sorted({m.get("group", "") for m in self.cfg["memories"]} - {""}):
                self.mem_group.addItem(g, g)
            self.mem_group.setCurrentIndex(max(self.mem_group.findData(current), 0))
            self._updating_groups = False
        elif getattr(self, "_updating_groups", False):
            return
        self.mem.clear()
        self._mem_rows = self.shown_memories()
        for m in self._mem_rows:
            group = (" [%s]" % m["group"]) if m.get("group") else ""
            self.mem.addItem("%s — %s Hz — %s%s" % (m["name"], format(int(m["hz"]), ",").replace(",", " "), m["mode"], group))

    def on_mem_add(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Mémoriser", "Nom de la station :")
        if ok and name:
            group, ok2 = QtWidgets.QInputDialog.getText(self, "Mémoriser", "Groupe (facultatif) :",
                                                        text=self.mem_group.currentData() or "")
            mode = self.current_mode()
            self.cfg["memories"].append({"name": name, "hz": self.freq.value(), "mode": mode,
                                         "group": group.strip() if ok2 else ""})
            self.refresh_memories()
            save_config(self.cfg)

    def on_mem_del(self):
        row = self.mem.currentRow()
        if 0 <= row < len(self._mem_rows):
            self.cfg["memories"].remove(self._mem_rows[row])
            self.refresh_memories()
            save_config(self.cfg)

    def on_recall(self, item):
        m = self._mem_rows[self.mem.row(item)]
        self.on_mode(m["mode"])
        self.tune(float(m["hz"]))

    # ------------------------------------------------------------ affichage
    def on_timer(self):
        r = self.radio
        if not self.ready():
            return
        lo, hi = r.passband()
        self.set_band(lo, hi)
        self.vfo_line.blockSignals(True); self.vfo_line.setValue(r.vfo_hz); self.vfo_line.blockSignals(False)
        self.wf_line.setValue(r.vfo_hz)
        if not r.running:
            return
        self._ticks += 1
        self.run_net_commands()
        if self._ticks % 20 == 0:
            self.run_scheduler()
        if self.scanner.active:
            self.scanner.tick()
            self.mode_buttons[r.rx.mode].setChecked(True)
            if self.scanner.holding and not self._was_holding and self.log_hits.isChecked():
                try:
                    tone = ("%.1f Hz" % r.rx.ctcss_hz) if r.rx.ctcss_hz else (r.rx.dcs_label or "")
                    planner.log_activity(self.cfg["record_dir"], r.vfo_hz, r.rx.mode, r.rx.level_db, tone)
                except OSError:
                    pass
        self._was_holding = self.scanner.active and self.scanner.holding
        if int(round(r.vfo_hz)) != self.freq.value():         # fréquence changée par le scanner ou le réseau
            self._set_freq_display(r.vfo_hz)
        if r.center_hz != self._shown_center:                 # la fréquence centrale a bougé : on garde le zoom
            self.frame_view(shift=r.center_hz - (self._shown_center if self._shown_center is not None else r.center_hz))
        if r.vfo_hz != self._followed_vfo:                    # la fréquence écoutée a changé : la vue la suit
            self.follow_vfo()
        if self._ticks % 5 == 0:
            self.update_info()
            self._tetra_auto_play_check()
            self.update_mem_labels()
            if self.is_file():
                self.update_player()
        if self._ticks % 2 == 0:
            self.update_analysis()
            self.update_filter_response()
        # ADS-B
        if self.adsb is not None and self.radio.last_block is not None:
            try:
                self.adsb.process_iq(self.radio.last_block)
            except Exception:
                pass
        # ACARS / AIS (sur audio démodulé, mode NFM)
        if (self.acars or self.ais) and self.ready():
            try:
                # On utilise le spectre audio ou un buffer simple
                # Pour une version plus complète, brancher sur la sortie audio du récepteur
                pass
            except Exception:
                pass
        now = time.perf_counter()
        if now < self._spec_next:                             # une grande FFT coûte cher : on espace les calculs
            return
        fs = r.sample_rate
        x0, x1 = self.plot.getViewBox().viewRange()[0]
        x0, x1 = max(x0, r.center_hz - fs / 2), min(x1, r.center_hz + fs / 2)
        if x1 - x0 < 10:
            return
        spec = r.spectrum_view(x0, x1, WF_COLS, self.fft_size.currentData(), self.fft_window.currentData())
        if spec is None:
            return
        freqs, db, nfft = spec
        self._spec_next = now + 2.0 * (time.perf_counter() - now)   # au plus un tiers du temps passé à calculer
        self.resolution_hz = fs / nfft
        shown, peak = self.trace.update(freqs, db, self.avg_n.value(), self.peak_hold.isChecked())
        self.curve.setData(freqs, shown)
        self.peak_curve.setVisible(peak is not None)
        if peak is not None:
            self.peak_curve.setData(freqs, peak)
        self.scroll_waterfall(x0, x1, freqs, db)
        self._floor_db += 0.2 * (float(np.median(db)) - self._floor_db)    # niveau du bruit, lissé
        self.img.setImage(self.wf, autoLevels=False, levels=self.waterfall_levels())
        self.img.setRect(QtCore.QRectF(x0, 0, x1 - x0, WF_ROWS))
        if self._ticks % 5 == 0:
            self.fft_info.setText("%s Hz par point (FFT de %s points, %.0f ms de signal)" % (
                ("%.1f" % self.resolution_hz).replace(".", ","), "{:,}".format(nfft).replace(",", " "), 1000.0 * nfft / fs))
        dbm = r.rx.level_db + self.s_cal.value()
        s_text, s_part = s_units(dbm, vhf=r.vfo_hz > 30e6)
        self.s_bar.setValue(int(round(1000 * s_part)))
        self.s_text.setText(s_text)
        self.smeter.setText("%s  (%.0f dBm, %.0f dB sous la pleine échelle)" % (s_text, dbm, -r.rx.level_db))
        msg = "%s  |  %.3f MS/s  |  charge %d %%" % (r.rx.mode, r.sample_rate / 1e6, round(100 * r.load))
        if r.load > 0.9:
            msg += " (trop élevée : augmenter la décimation ou baisser le débit)"
        if r.lost_blocks:
            msg += "  |  blocs perdus : %d (ordinateur trop lent)" % r.lost_blocks
        if getattr(r.source, "dropped", 0):
            msg += "  |  échantillons perdus (USB) : %d" % r.source.dropped
        if r.audio is not None and r.audio.underruns:
            msg += "  |  trous audio : %d" % r.audio.underruns
        if r.error:
            msg += "  |  ERREUR : %s" % r.error
        if not (self.rec_audio.isChecked() or self.rec_iq.isChecked()):
            self.status.showMessage(msg)

    def scroll_waterfall(self, x0, x1, freqs, db):
        """Ajoute une ligne à la cascade ; si le zoom a changé, les anciennes lignes sont recalées."""
        if self.wf_range is not None and self.wf_range != (x0, x1):
            o0, o1 = self.wf_range
            new_x = x0 + (np.arange(WF_COLS) + 0.5) * (x1 - x0) / WF_COLS
            idx = np.floor((new_x - o0) / (o1 - o0) * WF_COLS).astype(np.int64)
            valid = (idx >= 0) & (idx < WF_COLS)
            self.wf = np.where(valid, self.wf[:, np.clip(idx, 0, WF_COLS - 1)], np.float32(-140.0))
        self.wf_range = (x0, x1)
        self.wf[1:] = self.wf[:-1]
        cols = x0 + (np.arange(WF_COLS) + 0.5) * (x1 - x0) / WF_COLS
        self.wf[0] = np.interp(cols, freqs, db)

    def update_info(self):
        """Onglet Analyse et nom de la bande (4 fois par seconde)."""
        r = self.radio
        rx = r.rx
        self.band_label.setText(band_at(r.vfo_hz))
        if r.rx2 is not None:
            self.vfo2_line.blockSignals(True); self.vfo2_line.setValue(r.vfo2_hz); self.vfo2_line.blockSignals(False)
        if self.ts_on.isChecked():
            available = r.history_seconds()
            self.ts_delay.setRange(0, int(available))
            self.ts_label.setText(("En direct" if self.ts_delay.value() == 0 else "%d s en arrière" % self.ts_delay.value())
                                  + " — %d s en mémoire" % available)
        info = rx.rds_info() if rx.mode == "WFM" else None
        if info is not None:
            self.rds_ps.setText(info["ps"] or "—")
            self.rds_rt.setText(info["rt"])
            self.rds_misc.setText(("%04X  %s" % (info["pi"], info["pty"])) if info["pi"] is not None else "")
            self.stereo_label.setText("Stéréo" if rx.is_stereo else "Mono")
        else:
            self.rds_ps.setText("—"); self.rds_rt.setText(""); self.rds_misc.setText(""); self.stereo_label.setText("—")
        tone = rx.ctcss_hz if rx.mode == "NFM" else None
        self.ctcss_label.setText("%.1f Hz" % tone if tone else "—")
        self.dcs_label.setText((rx.dcs_label if rx.mode == "NFM" else None) or "—")
        new = ""
        while rx.digi_text:
            new += rx.digi_text.popleft()
            # Extraire les METAR du texte décodé
            for m in self.metar.feed(new):
                self.metar_list.addItem(format_metar(m))
                self.metar_list.scrollToBottom()
            for s in self.sigmet.feed(new):
                self.sigmet_list.addItem(format_sigmet(s))
                self.sigmet_list.scrollToBottom()
        if new:
            kind = self.digi_kind.currentData() or ""
            if kind in ("FT8", "FT4") and hasattr(self, "ft8_list"):
                for ln in new.splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    # Chercher un locator dans le message et calculer la distance
                    dist_str = ""
                    for word in ln.split():
                        w = word.upper().strip(".,;")
                        if len(w) >= 4 and w[0].isalpha() and w[1].isalpha() and w[2].isdigit():
                            coords = maidenhead_to_latlon(w)
                            my = maidenhead_to_latlon(MY_LOCATOR)
                            if coords and my:
                                km = haversine_km(my[0], my[1], coords[0], coords[1])
                                dist_str = f"  [{w} {km:.0f} km]"
                                break
                    display = f"{time.strftime('%H:%M:%S')}  {kind}  {ln}{dist_str}"
                    item = QtWidgets.QListWidgetItem(display)
                    item.setForeground(QtGui.QColor("#e2e8f0"))
                    # Stocker le locator pour le lien
                    for word in ln.split():
                        w = word.upper().strip(".,;")
                        if len(w) >= 4 and w[0].isalpha() and w[1].isalpha() and w[2].isdigit():
                            item.setData(QtCore.Qt.ItemDataRole.UserRole, w)
                            item.setToolTip(f"Cliquer pour voir {w} sur la carte")
                            break
                    if self.ft8_list.count() == 0 or self.ft8_list.item(self.ft8_list.count()-1).text() != item.text():
                        self.ft8_list.addItem(item)
                        # Journal FT8/FT4
                        self._log_ft8(ln, dist_str, mode=kind)
                while self.ft8_list.count() > 50:
                    self.ft8_list.takeItem(0)
                # Défilement auto seulement si l'utilisateur est en bas
                sb = self.ft8_list.verticalScrollBar()
                at_bottom = sb.value() >= sb.maximum() - 5
                if at_bottom:
                    self.ft8_list.scrollToBottom()
            self._digi_buf = (self._digi_buf + new)[-6000:]
            self.digi_out.setPlainText(self._digi_buf)
            bar = self.digi_out.verticalScrollBar()
            bar.setValue(bar.maximum())
        d = rx.digi
        self.audio_marker.setVisible(d is not None)
        if d is not None:
            self.audio_marker.setValue(rx.digi_audio_center())
            levels = np.array(d.history, dtype=np.float32)
            if len(levels) > 10:
                self.level_curve.setData(np.arange(len(levels)), levels)
                top = float(np.percentile(levels, 99)) + 6
                self.level_plot.setYRange(top - 50, top, padding=0)
                self.level_plot.setXRange(0, d.history.maxlen, padding=0)
            thr_db = getattr(d, "thr_db", None)
            self.level_thr.setVisible(thr_db is not None)
            if thr_db is not None:
                self.level_thr.setValue(thr_db)
            if hasattr(d, "take_rows"):
                self.update_picture(d)
            if hasattr(d, "points") and len(d.points) > 8:
                pts = np.array(d.points)
                self.const_curve.setData(pts.real, pts.imag)
            if hasattr(d, "on_timer"):
                try:
                    d.on_timer()
                except Exception:
                    pass
            if hasattr(d, "status") and self.current_mode() != "TETRA":
                try:
                    st = d.status()
                    if st and hasattr(self, "digi_title"):
                        kind = self.digi_kind.currentData() or ""
                        self.digi_title.setText(f"Décodeur {kind} — {st}")
                except Exception:
                    pass
            if hasattr(d, "status"):                             # TETRA, NAVTEX, fax, SSTV
                if self.current_mode() == "TETRA" and hasattr(self, "tetra_mcc"):
                    # Lecture audio continue si la case est cochée
                    if hasattr(self, "tetra_audio") and self.tetra_audio.isChecked():
                        if hasattr(d, "decode_voice"):
                            try:
                                audio = d.decode_voice()
                                if len(audio) > 100:
                                    import sounddevice as sd
                                    sd.play(audio, 8000, blocking=False)
                            except Exception:
                                pass
                    cell = getattr(d, "cell", None)
                    if cell:
                        self.tetra_mcc.setText(f"MCC:{cell.get('mcc', '—')}")
                        self.tetra_mnc.setText(f"MNC:{cell.get('mnc', '—')}")
                        self.tetra_color.setText(f"Color:{cell.get('colour', '—')}")
                    self.tetra_fer.setText(f"FER {getattr(d, 'freq_error_hz', 0):+.0f}Hz")
                    self.tetra_ber.setText(f"BER {100*(1-getattr(d, 'quality', 0)):.1f}%")
                    self.tetra_mer.setText(f"MER {100*getattr(d, 'quality', 0):.1f}%")
                self.digi_status.setText(d.status())
            elif rx.digi_kind == "CW":
                self.digi_status.setText("Vitesse estimée : %.0f mots/min" % d.wpm())
            elif rx.digi_kind and rx.digi_kind and rx.digi_kind.startswith("PSK"):
                self.digi_status.setText("Qualité %.0f %%, signal à %.0f Hz" % (100 * d.quality, d.bb.center))
            elif rx.digi_kind in ("PACKET300", "APRS1200"):
                self.digi_status.setText("Trames valides : %d" % d.frames)
            else:
                self.digi_status.setText("Qualité %.0f %%" % (100 * d.q))
        else:
            self.digi_status.setText("" if rx.digi_kind is None else
                                     ("Passer en NFM" if rx.digi_kind == "APRS1200" else "Passer en USB, LSB ou CW"))
        messages = list(rx.pocsag_messages)
        for m in messages[self._pocsag_seen:]:
            self.pocsag_list.addItem("%s  %d bit/s  adresse %d\n%s" % (m["time"], m["baud"], m["ric"], m["text"]))
        self._pocsag_seen = len(messages)
        spec = r.audio_spectrum()
        if spec is not None:
            keep = spec[0] <= (16000 if rx.mode == "WFM" else 6000)
            self.audio_curve.setData(spec[0][keep], spec[1][keep])

    def on_acars(self, on):
        if on and self.ready():
            fs = self.radio.sample_rate
            def emit(msg):
                reg = msg.get("registration") or "?"
                txt = msg.get("text", "")[:80]
                self.acars_list.addItem(f"{reg}  {txt}")
                self.acars_list.scrollToBottom()
            self.acars = ACARSDecoder(fs, emit)
        else:
            self.acars = None

    def on_ais(self, on):
        if on and self.ready():
            fs = self.radio.sample_rate
            def emit(msg):
                mmsi = msg.get("mmsi") or "?"
                lat = msg.get("lat")
                lon = msg.get("lon")
                pos = f"{lat:.4f}, {lon:.4f}" if lat and lon else ""
                self.ais_list.addItem(f"MMSI {mmsi}  {pos}")
                self.ais_list.scrollToBottom()
                # Afficher sur la carte si disponible
                if self.adsb_map and lat and lon:
                    import json
                    ac = {"icao": str(mmsi), "callsign": f"MMSI {mmsi}", "lat": lat, "lon": lon, "altitude": 0}
                    js = "updateAircraft([" + json.dumps(ac) + "]);"
                    self.adsb_map.page().runJavaScript(js)
            self.ais = AISDecoder(fs, emit)
        else:
            self.ais = None

    def on_rrc_changed(self, *_):
        if self.ready() and self.radio.rx.mode == "TETRA":
            digi = getattr(self.radio.rx, "digi", None)
            if digi and hasattr(digi, "set_rrc"):
                digi.set_rrc(self.rrc_alpha.value(), self.rrc_span.value())
                self._update_rrc_plot()

    def _update_rrc_plot(self):
        if not self.ready():
            return
        digi = getattr(self.radio.rx, "digi", None)
        if digi and hasattr(digi, "get_rrc_response"):
            freqs, mag = digi.get_rrc_response()
            # Afficher en kHz
            self.rrc_curve.setData(freqs / 1000, mag)

    def on_tetra_auto(self, on):
        if self.ready() and self.radio.rx.mode == "TETRA":
            digi = getattr(self.radio.rx, "digi", None)
            if digi:
                digi.set_timeslot(0 if on else 1)

    def _tetra_auto_play_check(self):
        """Vérifie et joue automatiquement si voix active."""
        if not getattr(self, "tetra_auto_play", None) or not self.tetra_auto_play.isChecked():
            return
        if not self.ready() or self.radio.rx.mode != "TETRA":
            return
        digi = getattr(self.radio.rx, "digi", None)
        if digi and hasattr(digi, "decode_voice"):
            audio = digi.decode_voice()
            if len(audio) > 100:  # assez de samples = voix active
                try:
                    import sounddevice as sd
                    sd.play(audio, 8000)
                except Exception:
                    pass

    def on_tetra_config(self):
        """Affiche les options de configuration TETRA."""
        QtWidgets.QMessageBox.information(self, "TETRA Config",
            "Options TETRA :\n"
            "- Timeslot : sélection manuelle ou auto\n"
            "- Audio vocal : active le décodage ACELP\n"
            "- Lecture auto : joue quand un slot est actif\n\n"
            "Le codec est dans WSL : /tetra/bin/sdecoder")

    def on_tetra_netinfo(self):
        """Affiche les informations réseau TETRA."""
        if not self.ready() or self.radio.rx.mode != "TETRA":
            QtWidgets.QMessageBox.information(self, "Net Info", "Pas en mode TETRA")
            return
        digi = getattr(self.radio.rx, "digi", None)
        if digi:
            info = f"MCC : {getattr(digi, 'mcc', '—')}\n"
            info += f"MNC : {getattr(digi, 'mnc', '—')}\n"
            info += f"LA : {getattr(digi, 'la', '—')}\n"
            info += f"Color : {getattr(digi, 'color', '—')}\n"
            info += f"Fréquence : {self.radio.vfo_hz/1e6:.3f} MHz"
            QtWidgets.QMessageBox.information(self, "TETRA Net Info", info)
        else:
            QtWidgets.QMessageBox.information(self, "Net Info", "Pas de données")

    def on_tetra_play(self):
        if not self.ready() or self.radio.rx.mode != "TETRA":
            return
        digi = getattr(self.radio.rx, "digi", None)
        if digi and hasattr(digi, "decode_voice"):
            audio = digi.decode_voice()
            if len(audio) > 0:
                # Jouer via la sortie audio existante si possible
                try:
                    import sounddevice as sd
                    sd.play(audio, 8000)
                    self.tetra_play.setText("▶ Lecture...")
                except Exception as e:
                    self.tetra_bits_label.setText(f"Audio: {len(audio)} échantillons")
            else:
                self.tetra_bits_label.setText("Pas assez de bits")

    def on_tetra_ts_radio(self, on, ts):
        if on and self.ready() and self.radio.rx.mode == "TETRA":
            digi = getattr(self.radio.rx, "digi", None)
            if digi:
                digi.set_timeslot(ts)
                self.tetra_auto.setChecked(False)

    def on_tetra_voice(self, *_):
        if self.ready() and hasattr(self.radio.rx, "tetra_voice"):
            self.radio.rx.tetra_voice.set_enabled(self.tetra_voice.isChecked())

    def on_tetra_ts(self, index):
        if self.ready() and self.radio.rx.mode == "TETRA":
            digi = getattr(self.radio.rx, "digi", None)
            if digi and hasattr(digi, "set_timeslot"):
                # index 0 = Auto, 1-4 = TS1-TS4
                digi.set_timeslot(index)  # 0 = auto
                bits = digi.get_ts_bits()
                mode = "Auto" if index == 0 else f"TS{index}"
                self.tetra_bits_label.setText(f"{mode} — Bits : {len(bits)}")

    def on_adsb(self, on):
        if on and self.ready():
            fs = self.radio.sample_rate
            def emit(msg):
                self.adsb._time += 0.1
                msg = decode_mode_s(msg)
                msg = decode_adsb_payload(msg, self.adsb.cpr, self.adsb._time)
                icao = msg.get("icao") or "?"
                dfname = msg.get("df_name", "")
                cs = msg.get("callsign") or ""
                alt = msg.get("altitude")
                squawk = msg.get("squawk") or ""
                lat = msg.get("lat")
                lon = msg.get("lon")
                pos = f"{lat:.3f},{lon:.3f}" if lat and lon else ""
                extra = " ".join(filter(None, [cs, f"{alt}ft" if alt else "", squawk, pos]))
                line = f"{icao}  [{dfname}]  {extra}".strip()
                self.adsb_list.addItem(line)
                self.adsb_list.scrollToBottom()
                # Update map if available
                if self.adsb_map and lat and lon:
                    ac = {"icao": icao, "callsign": cs, "lat": lat, "lon": lon, "altitude": alt}
                    import json
                    js = "updateAircraft([" + json.dumps(ac) + "]);"
                    self.adsb_map.page().runJavaScript(js)
            self.adsb = ADSBDecoder(fs, emit)
        else:
            self.adsb = None

    def on_metar_clear(self):
        self.metar.clear()
        self.metar_list.clear()

    def on_sigmet_clear(self):
        self.sigmet.clear()
        self.sigmet_list.clear()

    def closeEvent(self, ev):
        self.timer.stop()
        if self.net is not None:
            self.net.close()
        if self.cat is not None:
            self.cat.close()
        self.cfg["vfo_hz"] = self.freq.value()
        try:
            self.cfg["panel_sizes"] = self.splitter.sizes()
        except Exception:
            pass
        save_config(self.cfg)
        self.radio.close()
        ev.accept()


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="pyphantom", description="PyPhantom SDR (port de RadioLab 0.15)")
    p.add_argument("--iq", default=None, help="fichier I/Q (WAV ou .cf32) à lire en boucle au lancement")
    p.add_argument("--rate", type=float, default=None, help="débit (S/s) d'un fichier .cf32 (pas d'en-tête)")
    a = p.parse_args(argv)
    cfg = load_config()

    def audio_factory():
        from .audio import AudioOutput
        return AudioOutput(device=cfg.get("audio_device") or None)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    win = MainWindow(Radio(audio_factory=audio_factory), cfg)
    win.show()
    if a.iq:
        win.open_iq(a.iq, a.rate)
    return app.exec()
