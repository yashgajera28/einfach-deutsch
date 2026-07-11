# Einfach Deutsch — German Text Simplification

A local, open-source system for simplifying complex German text into plain language (B1/A2).
It includes a full data pipeline, two training recipes (mT5 baseline + LoRA 7B), automatic evaluation metrics, a FastAPI backend, and a Streamlit web interface.

---

## Overview

- **Input:** complex German sentences from Wikipedia, legal texts, contracts, or arbitrary documents (PDF/DOCX/TXT).
- **Output:** simplified German at the requested CEFR level (A2, B1, B2).
- **Metrics:** SARI, BLEU, LIX, and Wiener Sachtextformel to verify that simplification actually happened.
- **Models:**
  - `google/mt5-small` baseline seq2seq model.
  - `LeoLM/leo-hessianai-7b` instruction-tuned with LoRA (fallback to `mistralai/Mistral-7B-Instruct-v0.2`).

All components run locally; no paid API keys are required.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Einfach Deutsch                           │
├──────────────────┬───────────────────────┬───────────────────────┤
│   Data Pipeline  │    Training Pipeline  │  Inference & Demo     │
├──────────────────┼───────────────────────┼───────────────────────┤
│ Wikipedia dump   │  Baseline: mT5-small  │  FastAPI backend      │
│ Simple Wikipedia │  (Seq2SeqTrainer)     │  - /simplify          │
│ Klexikon         ├───────────────────────┤  - /simplify/pdf      │
│ Muster-Vorlagen  │  LoRA 7B              │  - /evaluate          │
│ Synthetic pairs  │  (SFTTrainer + PEFT)  │  - /health            │
│                  │                       │  - /model/info        │
│ Preprocessing:   │ Evaluation: SARI,     ├───────────────────────┤
│ - spacy de       │ BLEU, LIX, WSTF       │  Streamlit frontend   │
│ - alignment      │                       │  - single text        │
│ - filters        │ Outputs: checkpoints, │  - batch CSV          │
│ - train/val/test │ metrics, logs         │  - evaluation         │
│                  │                       │  - about / model info │
└──────────────────┴───────────────────────┴───────────────────────┘
```

---

## Directory Structure

```
einfach-deutsch/
├── configs/
│   └── config.yaml                 # Hyperparameters, paths, model names
├── data/
│   ├── raw/                        # Downloaded/scraped sources
│   ├── processed/                  # Cleaned parallel datasets
│   └── evaluation/                 # Test sets and human-eval templates
├── notebooks/
│   └── eda.ipynb                   # EDA / error-analysis skeleton
├── scripts/                        # Utility scripts (tests, demo, cache warming)
├── src/
│   ├── api/                        # FastAPI app + text extraction
│   ├── data_collection/            # Scrapers and dataset builders
│   ├── evaluation/                 # SARI, BLEU, readability metrics
│   ├── frontend/                   # Streamlit app
│   ├── models/                     # Simplifier, baseline + LoRA trainers
│   ├── preprocessing/              # Cleaning, alignment, quality filters
│   └── utils/                      # Config, logging, path helpers
├── docker-compose.yml
├── Dockerfile
├── evaluate.py                     # CLI evaluation runner
├── prepare_data.py                 # CLI data-pipeline runner
├── run_api.py                      # CLI API runner
├── run_frontend.py                 # CLI frontend runner
├── setup.sh                        # Environment setup
├── train_baseline.py               # CLI baseline training
└── train_lora.py                   # CLI LoRA training
```

---

## Setup

### Local Setup

```bash
# Option A: run the setup script
bash setup.sh

# Option B: manual setup
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download de_core_news_lg
mkdir -p data/raw data/processed data/evaluation checkpoints outputs logs
```

### Docker

```bash
docker-compose up --build
```

The API is exposed on port `8000` and the frontend on port `8501` by default.

---

## Quick Start

All commands below assume the virtual environment is active and you are in the project root.

```bash
# 1. Build a small demo dataset (no Wikipedia dump download)
python prepare_data.py --demo

# 2. Run a smoke-test evaluation on a dummy test file
python evaluate.py --test-file data/evaluation/dummy_test.csv

# 3. Start the API server in one terminal
python run_api.py

# 4. Start the frontend in another terminal
python run_frontend.py
```

Then open `http://localhost:8501` in your browser.

---

## Training

### Baseline (mT5-small)

Demo / smoke test:

```bash
python train_baseline.py --demo
```

Full training:

```bash
python train_baseline.py --output-dir checkpoints/baseline
```

### LoRA 7B

Demo / smoke test:

```bash
python train_lora.py --demo
```

Full training:

```bash
python train_lora.py --output-dir checkpoints/lora
```

> **Compute note:** Full LoRA training of a 7B model with the config defaults (`batch_size=4`, gradient accumulation 4, fp16, 512 tokens) needs a GPU with at least 16 GB VRAM. The demo modes run a tiny subset on CPU for pipeline verification; expect slow generation on CPU.

---

## API Documentation

The API runs on `http://localhost:8000` by default.

### `GET /health`

```bash
curl http://localhost:8000/health
```

Returns `{"status": "ok"}` or `{"status": "degraded"}` when the simplifier model is not loaded.

### `GET /model/info`

```bash
curl http://localhost:8000/model/info
```

Returns model name/path and backend (`baseline` or `lora`).

### `POST /simplify`

```bash
curl -X POST http://localhost:8000/simplify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Die Kommission hat beschlossen, die Richtlinie zu überarbeiten.",
    "target_level": "B1",
    "preserve_entities": true,
    "explain": false
  }'
```

Response:

```json
{
  "simplified": "...",
  "readability_before": {"lix": 45.0, "wstf": 8.5},
  "readability_after": {"lix": 32.0, "wstf": 6.1},
  "confidence": 0.85,
  "level": "B1",
  "backend": "lora",
  "entities_preserved": [...],
  "explanation": null
}
```

### `POST /simplify/pdf`

```bash
curl -X POST http://localhost:8000/simplify/pdf \
  -F "file=@document.pdf"
```

Returns a page-by-page breakdown plus `combined_text` and `combined_simplified`.

### `POST /evaluate`

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "source": "Der Antragsteller legte umfangreiche Unterlagen vor.",
    "prediction": "Der Antragsteller hat viele Unterlagen gezeigt.",
    "reference": ["Der Antragsteller zeigte viele Unterlagen."]
  }'
```

Returns SARI, BLEU, and readability scores for source and prediction.

---

## Frontend Usage

The Streamlit app has four pages:

1. **Simplify** — paste text or upload a PDF/DOCX/TXT file. Shows original vs. simplified side-by-side with highlighted changes, LIX/WSTF scores, preserved entities, and optional explanations.
2. **Batch Process** — upload a CSV with a `text` column, pick the level, and download a results CSV containing `simplified`, `lix_before`, `lix_after`, etc.
3. **Evaluation** — upload a CSV with `source`, `reference`, and optionally `prediction` columns. Computes SARI, BLEU, and readability deltas row-by-row and shows histograms.
4. **About** — project description, configured model names, supported levels, and example inputs.

Use the sidebar to switch the UI between German and English.

---

## Configuration

`configs/config.yaml` is the single source of truth for:

- Paths (`data_raw`, `data_processed`, `outputs`, etc.)
- Model names and max sequence lengths
- LoRA parameters (`r`, `lora_alpha`, `target_modules`, dropout)
- Training hyperparameters for both models
- Simplification levels and default level
- Readability thresholds for color coding
- API host/port and frontend port
- spaCy and sentence-transformer model names

Environment variables can override a few runtime values:

- `PYTHON` — Python interpreter used by `setup.sh`
- `VENV_DIR` — virtual-environment directory (default: `.venv`)
- `API_URL` — optional backend URL for a future HTTP-only frontend mode

---

## Success Criteria

Targets used to judge whether training is successful:

| Metric | Target |
| --- | --- |
| SARI (test set) | > 35 |
| LIX reduction (mean) | ≥ 10 points |
| Frontend response time | < 20 s per request on CPU |
| Bureaucratic test set | better scores than generic Wikipedia test set |

These are reference targets; exact scores depend on the dataset size and training budget.

---

## Notes

- The full Wikipedia and Klexikon data pipeline downloads large dumps and takes significant disk space and time. Use `python prepare_data.py --demo` for a quick smoke test.
- The LoRA 7B path is designed for a single GPU with ~16 GB VRAM. CPU training is possible but very slow.
- All scripts are production-ready in structure but the demo/smoke modes are intended for CI and local verification without downloading multi-gigabyte model weights.
- OCR fallback for scanned PDFs requires a local Tesseract installation with the German language pack.

---

## License

MIT — see `LICENSE` for details.
