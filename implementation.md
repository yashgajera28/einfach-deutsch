\*\*EINFACH DEUTSCH — IMPLEMENTATION SPECIFICATION FOR CODE GENERATION AI\*\*



\---



\## 1. ARCHITECTURE OVERVIEW



Build a complete German text simplification system with three integrated components:

\- \*\*Data Pipeline\*\*: Collect, align, and preprocess parallel complex/simple German text

\- \*\*Training Pipeline\*\*: Fine-tune a German sequence-to-sequence model for text simplification

\- \*\*Inference \& Demo\*\*: REST API + web interface for real-time simplification with readability scoring



\---



\## 2. TECH STACK REQUIREMENTS



\- \*\*Language\*\*: Python 3.10+

\- \*\*ML Framework\*\*: PyTorch + Hugging Face Transformers + PEFT (LoRA)

\- \*\*NLP Preprocessing\*\*: spaCy with `de\_core\_news\_lg`

\- \*\*Data\*\*: pandas, datasets (HuggingFace), BeautifulSoup (scraping)

\- \*\*API\*\*: FastAPI

\- \*\*Frontend\*\*: Streamlit

\- \*\*PDF Processing\*\*: PyPDF2 + pytesseract (OCR fallback)

\- \*\*Metrics\*\*: custom SARI implementation + readability formulas (Wiener Sachtextformel, LIX)

\- \*\*Storage\*\*: SQLite for caching results, JSONL for datasets

\- \*\*Container\*\*: Docker (optional but recommended)



\---



\## 3. PROJECT STRUCTURE



```

einfach-deutsch/

├── data/

│   ├── raw/                    # Raw scraped/downloaded data

│   ├── processed/              # Cleaned parallel datasets

│   └── evaluation/             # Test sets + human eval templates

├── src/

│   ├── data\_collection/        # Scrapers + dataset builders

│   ├── preprocessing/          # Text cleaning, sentence alignment, tokenization

│   ├── models/                 # Model training scripts (baseline + LoRA)

│   ├── evaluation/             # Metrics computation (SARI, LIX, readability)

│   ├── api/                    # FastAPI backend

│   └── frontend/               # Streamlit app

├── notebooks/                  # EDA + error analysis

├── configs/                    # YAML config files for training

├── requirements.txt

├── Dockerfile

└── README.md

```



\---



\## 4. DATA PIPELINE IMPLEMENTATION



\### 4.1 Dataset Sources to Implement

\- \*\*Primary\*\*: Download German Wikipedia dump + Simple German Wikipedia dump. Align articles by inter-language links. Extract sentence pairs using sentence alignment (length ratio + bilingual embedding similarity).

\- \*\*Secondary\*\*: Scrape `Muster-Vorlage.net` for German contract templates (Mietvertrag, Arbeitsvertrag). Manually create 50-100 simplified versions.

\- \*\*Tertiary\*\*: Download Klexikon (German children's encyclopedia) articles aligned with standard Wikipedia.

\- \*\*Quaternary\*\*: Generate synthetic bureaucratic pairs using a teacher LLM (prompt: "Simplify this German bureaucratic text to B1 level while preserving all factual information").



\### 4.2 Preprocessing Steps

\- Clean HTML/wiki markup using `mwparserfromhell`.

\- Remove references, tables, and infoboxes.

\- Sentence segmentation using spaCy German model.

\- Filter pairs: length ratio between 0.5-2.0, no empty sequences, max length 256 tokens.

\- Split: 80% train, 10% validation, 10% test. Ensure no article overlap between splits.

\- Save as HuggingFace `datasets` object in Arrow format.



\### 4.3 Data Quality Filters

\- Remove pairs where complex and simple text are identical.

\- Remove pairs with >30% non-German characters.

\- Ensure bureaucratic dataset has at least 500 pairs for domain fine-tuning.



\---



\## 5. MODEL TRAINING PIPELINE



\### 5.1 Baseline Model

\- Base: `google/mt5-small` (multilingual T5 small, 300M parameters).

\- Task prefix: `"vereinfachen: "` prepended to every input.

\- Tokenizer: AutoTokenizer with `max\_length=256`.

\- Training: Seq2SeqTrainer with:

&#x20; - batch\_size=16

&#x20; - learning\_rate=5e-5

&#x20; - num\_epochs=5

&#x20; - warmup\_steps=500

&#x20; - eval\_strategy="steps", eval\_steps=500

&#x20; - save\_strategy="steps", save\_total\_limit=2

&#x20; - predict\_with\_generate=True

&#x20; - generation\_max\_length=256



\### 5.2 Advanced Model (Primary)

\- Base: `LeoLM/leo-hessianai-7b` (German LLM) OR `mistralai/Mistral-7B-Instruct-v0.2`.

\- Method: LoRA fine-tuning via PEFT.

\- LoRA config: r=16, lora\_alpha=32, target\_modules=\["q\_proj", "v\_proj", "k\_proj", "o\_proj"], lora\_dropout=0.05, bias="none", task\_type="CAUSAL\_LM".

\- Format data as instruction-following:

&#x20; ```

&#x20; SYSTEM: Du bist ein deutscher Textvereinfacher. Wandle komplexe Texte in einfache Sprache (Niveau B1) um. Behalte alle Fakten bei. Verwende kurze Sätze und einfache Wörter.

&#x20; USER: Vereinfache: {complex\_text}

&#x20; ASSISTANT: {simple\_text}

&#x20; ```

\- Training: SFTTrainer (TRL library) with:

&#x20; - max\_seq\_length=512

&#x20; - batch\_size=4 (gradient accumulation=4 for effective 16)

&#x20; - learning\_rate=2e-4

&#x20; - num\_epochs=3

&#x20; - fp16=True

&#x20; - logging\_steps=10



\### 5.3 Training Infrastructure

\- Use `accelerate` for mixed precision training.

\- Save checkpoints every 500 steps.

\- Implement early stopping based on validation SARI score (patience=3).

\- Log experiments with Weights \& Biases or TensorBoard.



\---



\## 6. EVALUATION PIPELINE



\### 6.1 Automatic Metrics

\- \*\*SARI\*\*: Implement exact SARI formula comparing source, prediction, and reference. Use n-gram overlap for addition, deletion, and keeping operations.

\- \*\*BLEU\*\*: SacreBLEU for reference-based evaluation.

\- \*\*Readability\*\*:

&#x20; - \*\*LIX\*\* (Läsbarhetsindex): Implement `LIX = (words/sentences) + (long\_words\*100/words)` where long\_words > 6 characters.

&#x20; - \*\*Wiener Sachtextformel\*\*: Implement WSTF = 0.1935\*MS + 0.1672\*SL + 0.1297\*IW - 0.0327\*ES - 0.875 (where MS=mean sentence length, SL=syllables per word, IW=percentage of words with >3 syllables, ES=percentage of words with 1 syllable).

\- Compute delta scores: `readability(original) - readability(simplified)` must be positive (simplification actually occurred).



\### 6.2 Human Evaluation Framework

\- Generate a CSV template with columns: `original`, `simplified`, `simplicity\_score (1-5)`, `meaning\_preservation (1-5)`, `fluency (1-5)`.

\- Create a script that samples 50 random test pairs and formats them for human annotation.

\- Compute inter-annotator agreement if multiple annotators available.



\### 6.3 Error Analysis

\- Implement a script that categorizes errors: hallucination, fact loss, grammar error, insufficient simplification, over-simplification.

\- Generate confusion matrix of error types by source domain (Wikipedia vs. bureaucratic).



\---



\## 7. BACKEND API IMPLEMENTATION



\### 7.1 FastAPI Endpoints

\- `POST /simplify`: Accepts JSON `{"text": "...", "target\_level": "B1"}` and returns `{"simplified": "...", "readability\_before": {"lix": X, "wstf": Y}, "readability\_after": {"lix": X, "wstf": Y}, "confidence": 0.95}`.

\- `POST /simplify/pdf`: Accepts multipart file upload (PDF). Extract text via PyPDF2, run simplification, return structured response with page-by-page breakdown.

\- `POST /evaluate`: Accepts `{"source": "...", "prediction": "...", "reference": "..."}` and returns SARI + BLEU + readability scores.

\- `GET /health`: Health check endpoint.

\- `GET /model/info`: Returns model name, training date, dataset size.



\### 7.2 Model Loading

\- Load model at startup using lifespan context manager.

\- Cache model in GPU memory if available, else CPU.

\- Implement request queue for concurrent users (max 5 simultaneous requests).



\### 7.3 Text Extraction

\- PDF: PyPDF2 primary. If text extraction fails (<50 chars), fallback to OCR via pytesseract with German language pack.

\- DOCX: python-docx for Word documents.



\---



\## 8. FRONTEND IMPLEMENTATION



\### 8.1 Streamlit App Pages

\- \*\*Page 1 — Simplify\*\*: Text area input + file upload (PDF/DOCX/TXT). Side-by-side display: original (left) vs. simplified (right). Highlight changed words using difflib. Display readability gauges (before/after LIX).

\- \*\*Page 2 — Batch Process\*\*: Upload CSV with column `text`. Download results CSV with `simplified`, `lix\_before`, `lix\_after`.

\- \*\*Page 3 — Evaluation\*\*: Upload test set (CSV with `source`, `reference`, `prediction` or just `source`, `reference` to run model and evaluate). Display SARI, BLEU, readability histograms.

\- \*\*Page 4 — About\*\*: Project description, model info, example inputs.



\### 8.2 UI Requirements

\- German/English language toggle for UI labels.

\- Color-coded readability: Red (>50 LIX), Yellow (35-50), Green (<35).

\- Export simplified text as PDF or TXT.

\- Show loading spinner during generation (can take 5-15 seconds on CPU).



\---



\## 9. ADVANCED FEATURES



\### 9.1 Controllable Simplification

\- Allow user to select target level: A2, B1, B2.

\- Implement via prompt engineering for LLM approach: prepend `"Schreibe dies auf Niveau A2: "` vs `"Schreibe dies auf Niveau B1: "`.

\- For seq2seq, train separate control tokens: `<A2>`, `<B1>`, `<B2>`.



\### 9.2 Named Entity \& Number Preservation

\- Use spaCy German NER to extract entities (dates, amounts, legal references like § 439 BGB).

\- Post-process model output: if entity is missing or changed, inject original entity back into output.

\- Implement regex-based legal reference detector (`§\\s\*\\d+\[a-z]?\\s\*(Abs\\.\\s\*\\d+)?\\s\*(Satz\\s\*\\d+)?`).



\### 9.3 Explanation Mode

\- Generate explanation of what was simplified: "Passive voice converted to active", "Compound word split", "Legal term explained".

\- Use diff algorithm + rule-based mapping to generate explanations.



\---



\## 10. DEPLOYMENT \& DELIVERABLES



\### 10.1 Docker

\- Create Dockerfile with Python 3.10, install all requirements, download spaCy model and HuggingFace model at build time.

\- Expose port 8000 for API.

\- docker-compose.yml with API service + optional frontend service.



\### 10.2 Scripts to Provide

\- `setup.sh`: Install dependencies, download models, prepare data directories.

\- `train\_baseline.py`: End-to-end baseline training script.

\- `train\_lora.py`: End-to-end LoRA training script.

\- `run\_api.py`: Start FastAPI server.

\- `run\_frontend.py`: Start Streamlit app.

\- `evaluate.py`: Run full evaluation on test set.

\- `prepare\_data.py`: Run entire data pipeline from raw to processed.



\### 10.3 Configuration

\- `config.yaml` with all hyperparameters, paths, model names. All scripts read from this config.



\---



\## 11. STEP-BY-STEP BUILD ORDER FOR CODE GEN



Implement in this exact sequence:



1\. \*\*Project skeleton\*\*: Create directory structure, requirements.txt, config.yaml.

2\. \*\*Data module\*\*: Implement Wikipedia scraper/alignment, preprocessing, dataset builder. Verify by printing 5 sample pairs.

3\. \*\*Preprocessing module\*\*: spaCy pipeline, sentence alignment, quality filters. Test on sample data.

4\. \*\*Baseline training\*\*: mT5-small training script. Train for 1 epoch on small subset to verify pipeline works.

5\. \*\*Evaluation module\*\*: Implement SARI, LIX, WSTF. Test on dummy data.

6\. \*\*LoRA training\*\*: Implement instruction-formatting + SFTTrainer. Train on full dataset.

7\. \*\*Model inference\*\*: Create a `Simplifier` class that wraps both models with unified interface.

8\. \*\*FastAPI backend\*\*: All endpoints with model loading and caching.

9\. \*\*Streamlit frontend\*\*: All 4 pages with side-by-side diff and readability gauges.

10\. \*\*PDF/Docx processing\*\*: Integrate text extraction into API and frontend.

11\. \*\*Advanced features\*\*: Controllable levels, entity preservation, explanation mode.

12\. \*\*Docker + deployment\*\*: Containerize everything.

13\. \*\*Documentation\*\*: README with setup instructions, training commands, API docs.



\---



\## 12. CONSTRAINTS \& REQUIREMENTS



\- All code must be fully typed (Python type hints).

\- All functions must have docstrings in English.

\- All user-facing text in frontend must support German and English.

\- No hardcoded paths; everything via config.yaml or environment variables.

\- GPU optional: training scripts must gracefully fall back to CPU with warning.

\- Model size must be runnable on a single GPU with 16GB VRAM (hence LoRA on 7B, not full fine-tune).

\- No paid API dependencies (no OpenAI, no Google Cloud). Fully local/open-source.



\---



\## 13. SUCCESS CRITERIA



\- SARI score > 35 on test set (reasonable for German TS).

\- LIX reduction of at least 10 points on average.

\- Frontend demo functional with <20s response time per request on CPU.

\- API handles concurrent requests without crashing.

\- Bureaucratic domain test set shows better performance than generic Wikipedia test set (proving domain adaptation worked).

