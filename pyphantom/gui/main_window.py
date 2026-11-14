"""Fenetre principale de PyPhantom (inspire de l'UI de PhantomSDR-Plus)."""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import bookmarks as bm
from .. import config as cfgmod
from .. import device as devmod
from ..audio import AudioSink
from ..dsp import MODES, MODE_BW, s_unit_str
from ..recorder import AudioRecorder, IqRecorder
from .widgets import SpectrumWaterfall
from .worker import SdrWorker

SAMPLE_RATES = [250_000, 500_000, 1_000_000, 1_024_000, 1_800_000, 2_000_000, 2_048_000, 2_400_000, 2_500_000, 3_200_000, 10_000_000]


class MainWindow(QMainWindow):
    def __init__(self, iq_file: str | None = None, iq_rate: float = 1e6, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PyPhantom SDR (inspiré de PhantomSDR-Plus)")
        self.resize(1280, 800)
        self.cfg = cfgmod.load()
        self.bookmarks = bm.load()
        self.source = None
        self.iq_file = iq_file
        self.iq_rate = iq_rate
        self.iq_rec: IqRecorder | None = None
        self.wav_rec: AudioRecorder | None = None
        self.worker = SdrWorker()
        self.worker.audio_rate = int(self.cfg["audio_rate"])
        self.worker.calibration_db = float(self.cfg["calibration_db"])
        self.audio = AudioSink(int(self.cfg["audio_rate"]), float(self.cfg["volume"]), bool(self.cfg["muted"]))
        self.worker.audio_sink = self.audio
        self._build_ui()
        self._connect()
        self._restore_state()
        self._refresh_devices()
        self.statusBar().showMessage("Prêt. Choisissez une source puis START.")
        if self.iq_file:
            self.statusBar().showMessage(f"Source fichier IQ : {self.iq_file} ({self.iq_rate/1e6:g} MS/s)")
            self._start()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        self.view = SpectrumWaterfall()
        root.addWidget(self.view, stretch=1)

        grid = QGridLayout()
        row = 0

        grid.addWidget(QLabel("Source :"), row, 0)
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(220)
        grid.addWidget(self.device_combo, row, 1)
        self.btn_refresh = QPushButton("Lister")
        grid.addWidget(self.btn_refresh, row, 2)
        self.btn_open_iq = QPushButton("Fichier IQ…")
        grid.addWidget(self.btn_open_iq, row, 3)
        self.btn_start = QPushButton("START")
        self.btn_start.setCheckable(True)
        grid.addWidget(self.btn_start, row, 4)
        grid.addWidget(QLabel("Ech. (S/s) :"), row, 5)
        self.rate_combo = QComboBox()
        self.rate_combo.setEditable(True)
        for r in SAMPLE_RATES:
            self.rate_combo.addItem(f"{r:,}".replace(",", " "), r)
        grid.addWidget(self.rate_combo, row, 6)
        self.rate_label = QLabel("")
        grid.addWidget(self.rate_label, row, 7)
        row += 1

        grid.addWidget(QLabel("Fréquence :"), row, 0)
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setDecimals(6)
        self.freq_spin.setSuffix(" MHz")
        self.freq_spin.setRange(0.0, 10_000.0)
        self.freq_spin.setSingleStep(0.001)
        self.freq_spin.setAccelerated(True)
        grid.addWidget(self.freq_spin, row, 1)
        grid.addWidget(QLabel("Mode :"), row, 2)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(MODES)
        grid.addWidget(self.mode_combo, row, 3)
        grid.addWidget(QLabel("Bande passante (Hz) :"), row, 4)
        self.bw_spin = QSpinBox()
        self.bw_spin.setRange(100, 500_000)
        self.bw_spin.setSingleStep(100)
        grid.addWidget(self.bw_spin, row, 5)
        grid.addWidget(QLabel("Zoom :"), row, 6)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(1, 64)
        grid.addWidget(self.zoom_slider, row, 7)
        row += 1

        grid.addWidget(QLabel("Gain RF :"), row, 0)
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(0, 100)
        grid.addWidget(self.gain_slider, row, 1)
        self.agc_check = QCheckBox("AGC")
        grid.addWidget(self.agc_check, row, 2)
        self.gain_label = QLabel("")
        grid.addWidget(self.gain_label, row, 3)
        grid.addWidget(QLabel("Mode gain :"), row, 4)
        self.gain_mode_combo = QComboBox()
        self.gain_mode_combo.addItems(["Auto", "Linéarité", "Sensibilité"])
        grid.addWidget(self.gain_mode_combo, row, 5)
        self.spyverter_check = QCheckBox("Spyverter (+120 MHz)")
        grid.addWidget(self.spyverter_check, row, 6)
        self.bias_check = QCheckBox("Bias-tee (Spyverter)")
        grid.addWidget(self.bias_check, row, 7)
        row += 1

        grid.addWidget(QLabel("Volume :"), row, 0)
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        grid.addWidget(self.vol_slider, row, 1)
        self.mute_check = QCheckBox("Muet")
        grid.addWidget(self.mute_check, row, 2)
        grid.addWidget(QLabel("Squelch (dB) :"), row, 3)
        self.squelch_slider = QSlider(Qt.Horizontal)
        self.squelch_slider.setRange(-120, 0)
        grid.addWidget(self.squelch_slider, row, 4, 1, 2)
        grid.addWidget(QLabel("S-mètre :"), row, 6)
        self.smeter_bar = QProgressBar()
        self.smeter_bar.setRange(-140, -20)
        self.smeter_bar.setTextVisible(False)
        grid.addWidget(self.smeter_bar, row, 7)
        self.smeter_label = QLabel("--")
        grid.addWidget(self.smeter_label, row + 1, 6)
        row += 1

        self.btn_rec_iq = QPushButton("● Rec IQ")
        self.btn_rec_iq.setCheckable(True)
        grid.addWidget(self.btn_rec_iq, row, 6)
        self.btn_rec_wav = QPushButton("● Rec audio")
        self.btn_rec_wav.setCheckable(True)
        grid.addWidget(self.btn_rec_wav, row, 7)
        row += 1

        grid.addWidget(QLabel("Calibration (dB) :"), row, 0)
        self.calib_spin = QDoubleSpinBox()
        self.calib_spin.setRange(-100.0, 100.0)
        self.calib_spin.setSingleStep(1.0)
        grid.addWidget(self.calib_spin, row, 1)

        root.addLayout(grid)
        self.setCentralWidget(central)

        dock = QDockWidget("Favoris (fréquences)", self)
        dock_widget = QWidget()
        dv = QVBoxLayout(dock_widget)
        self.bm_list = QListWidget()
        dv.addWidget(self.bm_list)
        dh = QHBoxLayout()
        self.btn_bm_add = QPushButton("Ajouter la fréquence")
        self.btn_bm_del = QPushButton("Supprimer")
        dh.addWidget(self.btn_bm_add)
        dh.addWidget(self.btn_bm_del)
        dv.addLayout(dh)
        dock.setWidget(dock_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self._reload_bookmarks()

    def _connect(self) -> None:
        self.btn_refresh.clicked.connect(self._refresh_devices)
        self.btn_open_iq.clicked.connect(self._open_iq_file)
        self.btn_start.clicked.connect(self._toggle_start)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.freq_spin.valueChanged.connect(self._on_freq_changed)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.bw_spin.valueChanged.connect(self._on_bw_changed)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        self.gain_slider.valueChanged.connect(self._on_gain_changed)
        self.agc_check.toggled.connect(self._on_agc_toggled)
        self.gain_mode_combo.currentTextChanged.connect(self._on_gain_mode_changed)
        self.spyverter_check.toggled.connect(self._on_spyverter_toggled)
        self.bias_check.toggled.connect(self._on_bias_toggled)
        self.vol_slider.valueChanged.connect(self._on_vol_changed)
        self.mute_check.toggled.connect(self._on_mute_toggled)
        self.squelch_slider.valueChanged.connect(self._on_squelch_changed)
        self.calib_spin.valueChanged.connect(self._on_calib_changed)
        self.btn_rec_iq.toggled.connect(self._toggle_rec_iq)
        self.btn_rec_wav.toggled.connect(self._toggle_rec_wav)
        self.btn_bm_add.clicked.connect(self._bm_add_current)
        self.btn_bm_del.clicked.connect(self._bm_del_selected)
        self.bm_list.itemDoubleClicked.connect(self._bm_activate)
        self.view.tuneRequested.connect(self._on_tune_requested)
        self.worker.spectrumReady.connect(self._on_spectrum)
        self.worker.iqReady.connect(self._on_iq)
        self.worker.audioReady.connect(self._on_audio)
        self.worker.smeterReady.connect(self._on_smeter)
        self.worker.infoReady.connect(self.statusBar().showMessage)

    def _restore_state(self) -> None:
        c = self.cfg
        self.freq_spin.setValue(float(c["frequency_hz"]) / 1e6)
        self.mode_combo.setCurrentText(str(c["mode"]))
        self.bw_spin.setValue(int(c["bandwidth_hz"]))
        self.rate_combo.setCurrentIndex(0)
        for i in range(self.rate_combo.count()):
            if abs(self.rate_combo.itemData(i) - float(c["sample_rate"])) < 1:
                self.rate_combo.setCurrentIndex(i)
                break
        self.zoom_slider.setValue(int(c["zoom"]))
        self.gain_slider.setValue(int(c["gain"]))
        self.agc_check.setChecked(bool(c["agc"]))
        self.vol_slider.setValue(int(float(c["volume"]) * 100))
        self.mute_check.setChecked(bool(c["muted"]))
        self.squelch_slider.setValue(int(c["squelch_db"]))
        self.calib_spin.setValue(float(c["calibration_db"]))
        self.spyverter_check.setChecked(bool(c.get("spyverter", False)))
        self.bias_check.setChecked(bool(c.get("bias_tee", False)))
        gm = str(c.get("gain_mode", "Auto"))
        idx = self.gain_mode_combo.findText(gm)
        self.gain_mode_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.view.set_center(float(c["frequency_hz"]))
        self.view.set_zoom(float(c["zoom"]))

    def _refresh_devices(self) -> None:
        self.device_combo.clear()
        try:
            devs = devmod.list_devices()
        except Exception as exc:
            self.device_combo.addItem("(SoapySDR indisponible — voir README)", None)
            self.statusBar().showMessage(str(exc))
            return
        for d in devs:
            self.device_combo.addItem(d["label"], d["args"])
        if not devs:
            self.device_combo.addItem("(aucun récepteur détecté)", None)

    def _on_device_changed(self, _idx: int) -> None:
        args = self.device_combo.currentData()
        if not args:
            return
        try:
            probe = devmod.SdrDevice(args)
            rates = probe.supported_sample_rates(SAMPLE_RATES)
            probe.close()
        except Exception:
            return
        current = float(self.rate_combo.currentData() or 0)
        self.rate_combo.clear()
        for r in rates:
            self.rate_combo.addItem(f"{r:,}".replace(",", " "), r)
        for i in range(self.rate_combo.count()):
            if abs(self.rate_combo.itemData(i) - current) < 1:
                self.rate_combo.setCurrentIndex(i)
                break

    def _on_spyverter_toggled(self, on: bool) -> None:
        if self.source is not None and hasattr(self.source, "lo_offset"):
            self.source.lo_offset = 120_000_000.0 if on else 0.0
            self.source.set_center_freq(self.freq_spin.value() * 1e6)

    def _on_bias_toggled(self, on: bool) -> None:
        if self.source is not None and hasattr(self.source, "set_bias_tee"):
            self.source.set_bias_tee(on)

    def _on_gain_mode_changed(self, mode: str) -> None:
        if self.source is not None and hasattr(self.source, "set_gain_mode"):
            if mode == "Linéarité":
                self.source.set_gain_mode("linearity")
            elif mode == "Sensibilité":
                self.source.set_gain_mode("sensitivity")

    def _open_iq_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir un fichier IQ .cf32", "", "IQ CF32 (*.cf32);;Tous (*)")
        if path:
            self.iq_file = path
            self.device_combo.setCurrentIndex(-1)
            self.statusBar().showMessage(f"Fichier IQ sélectionné : {os.path.basename(path)}")

    def _toggle_start(self) -> None:
        if self.btn_start.isChecked():
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        rate = float(self.rate_combo.currentData() or self.iq_rate)
        freq = self.freq_spin.value() * 1e6
        try:
            if self.iq_file:
                self.source = devmod.IqFileSource(self.iq_file, self.iq_rate if self.iq_file else rate)
                self.source.sample_rate = self.iq_rate
                rate = self.iq_rate
            else:
                args = self.device_combo.currentData()
                if args is None:
                    raise RuntimeError("Aucun récepteur sélectionné. Branchez un dongle ou choisissez un fichier IQ.")
                self.source = devmod.SdrDevice(args)
                self.source.lo_offset = 120_000_000.0 if self.spyverter_check.isChecked() else 0.0
                gm = self.gain_mode_combo.currentText()
                if gm == "Linéarité":
                    self.source.set_gain_mode("linearity")
                elif gm == "Sensibilité":
                    self.source.set_gain_mode("sensitivity")
                self.source.set_bias_tee(self.bias_check.isChecked())
                self.source.setup(rate, freq, self.gain_slider.value() / 100.0)
                self.source.set_agc(self.agc_check.isChecked())
                rate = self.source.sample_rate
                freq = self.source.center_freq
                self.freq_spin.blockSignals(True)
                self.freq_spin.setValue(freq / 1e6)
                self.freq_spin.blockSignals(False)
            self.source.set_center_freq(freq)
        except Exception as exc:
            QMessageBox.warning(self, "Erreur", str(exc))
            self.btn_start.setChecked(False)
            return
        self.worker.set_source(self.source, rate)
        out_rate = int(round(self.worker.out_rate))
        if self.audio.rate != out_rate:
            vol, muted = self.audio.volume, self.audio.muted
            self.audio.close()
            self.audio = AudioSink(out_rate, vol, muted)
            self.worker.audio_sink = self.audio
        self.worker.demod.set_mode(self.mode_combo.currentText(), float(self.bw_spin.value()))
        self.worker.demod.squelch_db = float(self.squelch_slider.value())
        self.worker.demod.agc_enabled = True
        self.worker.start()
        self.view.fs = rate
        self.view.set_zoom(float(self.zoom_slider.value()))
        self.rate_label.setText(f"{rate/1e6:.3f} MS/s")
        self.btn_start.setText("STOP")
        self.statusBar().showMessage("Réception en cours…")

    def _stop(self) -> None:
        self.worker.stop()
        self.worker.wait(2000)
        if self.source is not None:
            self.source.close()
            self.source = None
        self._stop_rec_iq()
        self._stop_rec_wav()
        self.btn_start.setChecked(False)
        self.btn_start.setText("START")
        self.statusBar().showMessage("Arrêté.")

    def _on_freq_changed(self, mhz: float) -> None:
        freq = mhz * 1e6
        self.view.set_center(freq)
        if self.source is not None:
            self.source.set_center_freq(freq)

    def _on_tune_requested(self, freq: float) -> None:
        self.freq_spin.blockSignals(True)
        self.freq_spin.setValue(freq / 1e6)
        self.freq_spin.blockSignals(False)
        if self.source is not None:
            self.source.set_center_freq(freq)

    def _on_mode_changed(self, mode: str) -> None:
        self.bw_spin.blockSignals(True)
        self.bw_spin.setValue(int(MODE_BW.get(mode, 12_000)))
        self.bw_spin.blockSignals(False)
        self.worker.demod.set_mode(mode, MODE_BW.get(mode, 12_000))

    def _on_bw_changed(self, bw: int) -> None:
        self.worker.demod.set_bw(float(bw))

    def _on_zoom_changed(self, z: int) -> None:
        self.view.set_zoom(float(z))

    def _on_gain_changed(self, v: int) -> None:
        if self.source is not None and not self.agc_check.isChecked():
            if hasattr(self.source, "set_gain_fraction"):
                self.source.set_gain_fraction(v / 100.0)
            else:
                self.source.set_gain(v)
            self.gain_label.setText(f"{v} %")

    def _on_agc_toggled(self, on: bool) -> None:
        if self.source is not None:
            self.source.set_agc(on)

    def _on_vol_changed(self, v: int) -> None:
        self.audio.volume = v / 100.0

    def _on_mute_toggled(self, on: bool) -> None:
        self.audio.muted = on

    def _on_squelch_changed(self, v: int) -> None:
        self.worker.demod.squelch_db = float(v)

    def _on_calib_changed(self, v: float) -> None:
        self.worker.calibration_db = float(v)

    def _on_spectrum(self, db, center: float, fs: float) -> None:
        self.view.update_spectrum(np.asarray(db), center, fs)

    def _on_iq(self, iq) -> None:
        if self.iq_rec is not None:
            self.iq_rec.write(iq)

    def _on_audio(self, audio) -> None:
        self.audio.push(audio)
        if self.wav_rec is not None:
            self.wav_rec.write(audio)

    def _on_smeter(self, dbm: float) -> None:
        v = int(np.clip(dbm, -140, -20))
        self.smeter_bar.setValue(v)
        self.smeter_label.setText(f"{s_unit_str(dbm, self.freq_spin.value())}  ({dbm:.0f} dBm)")

    def _toggle_rec_iq(self, on: bool) -> None:
        if on:
            path, _ = QFileDialog.getSaveFileName(self, "Enregistrer l'IQ", "recording.cf32", "IQ CF32 (*.cf32)")
            if not path:
                self.btn_rec_iq.setChecked(False)
                return
            freq = self.freq_spin.value() * 1e6
            rate = self.source.sample_rate if self.source else float(self.rate_combo.currentData())
            self.iq_rec = IqRecorder(path, freq, rate)
            self.worker.emit_iq = True
            self.statusBar().showMessage(f"Enregistrement IQ : {path}")
        else:
            self._stop_rec_iq()

    def _stop_rec_iq(self) -> None:
        if self.iq_rec is not None:
            self.iq_rec.close()
            self.iq_rec = None
        self.worker.emit_iq = False
        self.btn_rec_iq.setChecked(False)

    def _toggle_rec_wav(self, on: bool) -> None:
        if on:
            path, _ = QFileDialog.getSaveFileName(self, "Enregistrer l'audio", "recording.wav", "WAV (*.wav)")
            if not path:
                self.btn_rec_wav.setChecked(False)
                return
            self.wav_rec = AudioRecorder(path, int(round(self.worker.out_rate)))
            self.worker.emit_audio = True
            self.statusBar().showMessage(f"Enregistrement audio : {path}")
        else:
            self._stop_rec_wav()

    def _stop_rec_wav(self) -> None:
        if self.wav_rec is not None:
            self.wav_rec.close()
            self.wav_rec = None
        self.worker.emit_audio = False
        self.btn_rec_wav.setChecked(False)

    def _reload_bookmarks(self) -> None:
        self.bm_list.clear()
        for b in self.bookmarks:
            item = QListWidgetItem(f"{b['frequency']/1e6:.6f} MHz — {b['name']} [{b.get('mode','NFM')}]")
            item.setData(Qt.UserRole, b)
            self.bm_list.addItem(item)

    def _bm_add_current(self) -> None:
        name, ok = QInputDialog.getText(self, "Favori", "Nom :", text=f"{self.freq_spin.value():.6f} MHz")
        if not ok or not name:
            return
        self.bookmarks.append(
            {"frequency": self.freq_spin.value() * 1e6, "name": name, "mode": self.mode_combo.currentText()}
        )
        bm.save(self.bookmarks)
        self._reload_bookmarks()

    def _bm_del_selected(self) -> None:
        item = self.bm_list.currentItem()
        if item is None:
            return
        b = item.data(Qt.UserRole)
        self.bookmarks = [x for x in self.bookmarks if x is not b]
        bm.save(self.bookmarks)
        self._reload_bookmarks()

    def _bm_activate(self, item: QListWidgetItem) -> None:
        b = item.data(Qt.UserRole)
        self.freq_spin.setValue(float(b["frequency"]) / 1e6)
        mode = b.get("mode")
        if mode in MODES:
            self.mode_combo.setCurrentText(mode)

    def closeEvent(self, ev) -> None:
        self.cfg.update(
            {
                "frequency_hz": self.freq_spin.value() * 1e6,
                "mode": self.mode_combo.currentText(),
                "bandwidth_hz": self.bw_spin.value(),
                "sample_rate": float(self.rate_combo.currentData() or 1e6),
                "gain": self.gain_slider.value(),
                "agc": self.agc_check.isChecked(),
                "volume": self.vol_slider.value() / 100.0,
                "muted": self.mute_check.isChecked(),
                "squelch_db": self.squelch_slider.value(),
                "zoom": self.zoom_slider.value(),
                "calibration_db": self.calib_spin.value(),
                "spyverter": self.spyverter_check.isChecked(),
                "bias_tee": self.bias_check.isChecked(),
                "gain_mode": self.gain_mode_combo.currentText(),
            }
        )
        cfgmod.save(self.cfg)
        bm.save(self.bookmarks)
        self._stop()
        self.audio.close()
        super().closeEvent(ev)
