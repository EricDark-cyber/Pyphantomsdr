@echo off
rem Fabrique PyPhantom.exe : un dossier autonome qui fonctionne sans installer Python.
rem A lancer une fois, sur un PC ou PyPhantom fonctionne deja (Python et les bibliotheques installes).
cd /d "%~dp0"
python -m pip install --upgrade pyinstaller
if errorlevel 1 goto erreur
python -m PyInstaller --noconfirm --clean --windowed --name PyPhantom --collect-all pyqtgraph --collect-all sounddevice --collect-all PySide6 run_pyphantom.py
if errorlevel 1 goto erreur
rem Les DLL de l'Airspy R2 doivent se trouver a cote de PyPhantom.exe.
for %%F in (airspy.dll libusb-1.0.dll pthreadVC2.dll pthreadVCE2.dll libwinpthread-1.dll) do if exist "%%F" copy /y "%%F" "dist\PyPhantom\" >nul
echo.
echo Termine : le dossier dist\PyPhantom contient PyPhantom.exe.
echo Copier ce dossier entier sur l'autre PC (le pilote WinUSB de l'Airspy doit y etre installe, voir Zadig).
pause
exit /b 0
:erreur
echo.
echo La fabrication a echoue : lire les messages ci-dessus.
pause
exit /b 1
