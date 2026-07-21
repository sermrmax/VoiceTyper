@echo off

echo Building Voice Typer...

python -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onefile ^
    --windowed ^
    --name VoiceTyper ^
    --collect-all faster_whisper ^
    --collect-all ctranslate2 ^
    --collect-all av ^
    main.py

echo.
echo Build completed: dist\VoiceTyper.exe
pause