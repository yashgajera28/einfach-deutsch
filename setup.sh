#!/usr/bin/env bash
set -e

echo "Setting up Einfach Deutsch environment..."

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python -m spacy download de_core_news_lg || true

mkdir -p data/raw data/processed data/evaluation checkpoints outputs logs

echo "Setup complete. Activate with: source .venv/bin/activate"
