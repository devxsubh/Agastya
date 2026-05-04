#!/bin/bash
set -e

echo "====================================="
echo "Agastya Pipeline: One-Click Setup"
echo "====================================="

# 1. Check Python version
if ! command -v python3 &> /dev/null
then
    echo "python3 could not be found. Please install Python 3.9+"
    exit 1
fi

echo "[1/4] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "[2/4] Upgrading pip..."
pip install --upgrade pip

echo "[3/4] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "[4/4] Setting up Spacy..."
python -m spacy download en_core_web_sm

echo "====================================="
echo "Setup Complete!"
echo "To activate the environment, run:"
echo "source .venv/bin/activate"
echo "====================================="
