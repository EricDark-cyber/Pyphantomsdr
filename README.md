# PyPhantom SDR

Récepteur SDR logiciel en **Python pur pour Windows**, inspiré de
[PhantomSDR-Plus](https://github.com/sv1btl/PhantomSDR-Plus) (spectre + waterfall
cliquable, zoom, modes AM/NFM/WFM/USB/LSB/CW, S-mètre en S-units/dBm, favoris de
fréquences, enregistrement IQ `.cf32` et audio `.wav`).

Architecture : **PySide6** (GUI Qt native Windows) + **SoapySDR** (accès au matériel :
RTL-SDR, HackRF, Airspy, etc.) + **numpy/scipy** (DSP) + **pyqtgraph** (spectre/waterfall)
+ **sounddevice** (audio).

```
pyphantom/
├── pyphantom/
│   ├── __main__.py        # point d'entrée : python -m pyphantom
│   ├── device.py          # SoapySDR (matériel) + IqFileSource (fichier .cf32)
│   ├── dsp.py             # démodulateurs AM/FM/SSB/CW, spectre, S-mètre, AGC, squelch
│   ├── audio.py           # sortie audio temps réel
│   ├── recorder.py        # enregistrements IQ (.cf32) et audio (.wav)
│   ├── bookmarks.py       # favoris (JSON : fréquence, nom, mode)
│   ├── config.py          # réglages persistants (%APPDATA%\PyPhantom)
│   └── gui/               # fenêtre principale, widget spectre/waterfall, thread d'acquisition
├── tests/test_dsp.py      # tests hors-ligne du DSP (signaux synthétiques)
└── tools/make_demo_iq.py  # génère un fichier IQ de démonstration
```

## Installation sous Windows

1. **Python 3.11+** : https://www.python.org/downloads/ — cochez « Add Python to PATH ».
2. **PothosSDR** (fournit `SoapySDR.dll` + les modules récepteurs) :
   téléchargez l'installeur sur https://downloads.pothosware.com/ et installez-le,
   puis ajoutez `C:\Program Files\PothosSDR\bin` à la variable d'environnement **PATH**
   (Paramètres système → Variables d'environnement).
3. **Pilote du dongle** : avec [Zadig](https://zadig.akeo.ie/), remplacez le pilote USB
   du RTL-SDR (etc.) par **WinUSB**.
4. Dans un terminal (`Win + R`, `cmd`), créez l'environnement et installez les dépendances :
   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   **Ne faites PAS `pip install SoapySDR`** : le paquet PyPI n'a pas de roue Windows.
   Les bindings Python de SoapySDR viennent de PothosSDR. Deux possibilités :
   - Si l'installeur PothosSDR a enregistré ses bindings (c'est le cas des versions
     récentes), testez directement :
     ```bat
     python -c "import SoapySDR; print(SoapySDR.Device.enumerate())"
     ```
   - Sinon, pointez `PYTHONPATH` vers les bindings de PothosSDR (vérifiez le dossier
     exact avec `dir "C:\Program Files\PothosSDR\lib\site-packages"` ; le nom du fichier
     `_SoapySDR.cp3XX-*.pyd` indique la version de Python requise, ex. `cp312` →
     installez Python 3.12) :
     ```bat
     set PATH=C:\Program Files\PothosSDR\bin;%PATH%
     set PYTHONPATH=C:\Program Files\PothosSDR\lib\site-packages;%PYTHONPATH%
     python -c "import SoapySDR; print(SoapySDR.Device.enumerate())"
     ```
     L'Airspy doit apparaître avec `driver=airspy` (sinon refaire le pilote WinUSB avec Zadig).
5. Lancez :
   ```bat
   python -m pyphantom
   ```

## Airspy R2 + Spyverter (HF 0–60 MHz)

Branchez le **Spyverter** sur l'entrée antenne de l'Airspy R2 (IN du Spyverter côté
Airspy, antenne HF sur l'entrée du Spyverter). Dans PyPhantom :

1. Cochez **Spyverter (+120 MHz)** : la fréquence affichée reste la vraie fréquence HF
   (ex. 7,1 MHz) tandis que le LO de l'Airspy est réglé à fréquence + 120 MHz.
2. Cochez **Bias-tee (Spyverter)** pour alimenter le Spyverter en 4,5 V via le bias-tee
   de l'Airspy R2 (sinon alimentation externe).
3. **Ech. (S/s)** : l'Airspy R2 ne supporte que **2 500 000** et **10 000 000** S/s —
   la liste est automatiquement réduite quand le récepteur est détecté.
4. **Mode gain** : *Linéarité* (LNA/MIX/VGA, recommandé) ou *Sensibilité*. Le curseur
   « Gain RF » répartit le gain sur les étages exposés par le pilote SoapySDR.

Le pilote `airspy` est fourni par PothosSDR (étape 2). Le récepteur apparaît dans la
liste avec `driver=airspy`.

## Test sans récepteur (fichier IQ)

```bat
python tools/make_demo_iq.py demo.cf32 10 1000000
python -m pyphantom --iq demo.cf32 --rate 1000000
```

Le fichier `demo.cf32` contient des stations synthétiques : AM à −150 kHz, FM à
+200 kHz, USB à +60 kHz par rapport à la fréquence centrale (100 MHz par défaut).

## Utilisation

- **START** : ouvre la source sélectionnée (récepteur listé, ou fichier IQ via « Fichier IQ… »).
- **Clic sur le spectre/waterfall** : accorde la fréquence. **Molette** : zoom (curseur ×1…×64).
- **Mode / Bande passante** : AM, NFM, WFM, USB, LSB, CW (BFO 700 Hz).
- **Gain RF + AGC**, **volume**, **muet**, **squelch**, **calibration dBm** (pour le S-mètre).
- **● Rec IQ** : enregistre le flux brut en `.cf32` (+ métadonnées `.json`).
- **● Rec audio** : enregistre le signal démodulé en `.wav`.
- **Favoris** (panneau de droite) : double-clic pour accorder ; bouton « Ajouter la fréquence ».

## Vérification (développement)

```bat
python tests/test_dsp.py
```

Teste hors-ligne les démodulateurs AM/NFM/USB/LSB/CW, le rééchantillonnage, le spectre
et le S-mètre sur des signaux synthétiques.
