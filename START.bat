@echo off
chcp 65001 >nul
title NovaByte Sketch Vectorizer - Demo
cd /d "%~dp0"

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║   NovaByte Flat Sketch Vectorizer v3     ║
echo  ║   Technical Test Demo                    ║
echo  ╚══════════════════════════════════════════╝
echo.

if not exist "output\jewelry_03\metrics.json" (
    echo  [1/2] Running pipeline on 10 images...
    echo.
    python main.py "D:\profile\10hinh_novabyte" -o output
    echo.
    echo  Pipeline complete!
) else (
    echo  [1/2] Output already exists, skipping pipeline.
)

echo.
echo  [2/2] Starting viewer at http://localhost:5125
echo.
echo  Press Ctrl+C to stop the server.
echo.

start http://localhost:5125
python serve_viewer.py
