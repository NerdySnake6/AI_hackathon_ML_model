# QWEN.md — Mediascope AI Business SPB Hackathon 2026

## Project Overview

This is a **machine learning hackathon project** for the AI Business SPB 2026 competition, organized by **Mediascope** (Russia's leading media research company). The goal is to build an ML model that classifies search queries by three attributes:

| # | Attribute | Description |
|---|-----------|-------------|
| 1 | **TypeQuery** (binary) | Whether the query relates to professional video content (1/0) |
| 2 | **ContentType** | Category: `фильм`, `сериал`, `мультфильм`, `мультсериал`, `прочее`, or empty |
| 3 | **Title** | Normalized franchise/title name, or empty |

The model must handle noisy user queries with typos, transliteration, and vague formulations. It runs as a `PredictionModel` class inside `solution.py`, deployed as a zip bundle to a sandbox environment (6 CPU, 48 GB RAM, 1× RTX 4090, 10 min timeout).

### Key Metric

```
combined_score = 0.35 * typequery_f2 + 0.30 * contenttype_macro_f1 + 0.35 * title_token_f1
```

- `typequery_f2` — F-beta (beta=2) over all rows
- `contenttype_macro_f1` — macro F1 across 6 classes (only where GT TypeQuery=1)
- `title_token_f1` — token-level F1 on bag-of-words (only where GT TypeQuery=1)

### Technical Constraints

- **On-premise, open-source only** — no generative LLM API calls during inference
- No network access except allowed LLM APIs (Yandex Cloud, OpenAI, Anthropic)
- Must process 15K queries within 4 hours
- Must support confidence scoring for operator routing (low-confidence → manual review)

## Directory Structure

```
h_t2/
├── README.md              # Hackathon task description & instructions
├── QWEN.md                # This file — project context for AI assistant
├── pyproject.toml         # uv project config with dependencies
├── .python-version        # Python 3.11
├── .env.example           # Template for API credentials
├── solution.py            # Main entry point — PredictionModel class
├── train.csv              # Training data (downloaded)
├── tz.md                  # Detailed technical specification (in Russian)
├── notebooks/
│   └── explore.ipynb      # Data exploration & baseline notebook
├── scripts/
│   ├── download_data.py   # Downloads training data from hackathon API
│   └── submit.py          # Builds bundle.zip and submits to sandbox
└── data/                  # Downloaded datasets (gitignored)
```

## Dependencies

Managed via **uv** (fast Python package manager). Key libraries:

- `pandas >= 2.0` — data manipulation
- `numpy >= 1.24` — numerical operations
- `scikit-learn >= 1.3` — ML models (TF-IDF, classifiers, etc.)
- `requests >= 2.31` — HTTP API calls
- `python-dotenv >= 1.0` — environment variable loading

## Building and Running

### Setup

```bash
uv sync
cp .env.example .env
# Edit .env and add your API_KEY from app.ai-business-spb.ru
```

### Download Training Data

```bash
uv run scripts/download_data.py
```

### Run Baseline Locally

```bash
uv run python -c "
import pandas as pd
from solution import PredictionModel
df = pd.read_csv('data/train.csv').head(20)
model = PredictionModel()
print(model.predict(df[['QueryText']]))
"
```

### Explore Data

Open `notebooks/explore.ipynb` in Jupyter to inspect label distributions and test the baseline.

### Submit Solution

```bash
uv run scripts/submit.py
```

This automatically builds `bundle.zip` (excluding `data/`, `.venv/`, `notebooks/`, etc.), submits it to the sandbox, and polls for results.

### Manual Submit

Upload `bundle.zip` via the web interface at https://app.ai-business-spb.ru

## Solution Interface

The sandbox expects `solution.py` with a `PredictionModel` class:

```python
class PredictionModel:
    batch_size: int = 10  # optional, default 10

    def __init__(self) -> None:
        # Called once at startup — load models, dictionaries, etc.
        ...

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # Input: DataFrame with 'QueryText' column (batch of rows)
        # Output: DataFrame with columns: QueryText, TypeQuery (int 0/1), Title (str), ContentType (str)
        ...
```

The sandbox calls `predict()` in batches of `batch_size` rows. If parallel processing is needed (e.g., concurrent LLM API calls), implement it inside `predict()`.

## Current State

The current `solution.py` is a **trivial baseline** that predicts:
- `TypeQuery = 0` for all queries
- `Title = ""` for all queries
- `ContentType = "other"` for all queries

This serves as a starting point. A production solution needs to implement actual classification logic.

## Technical Specification (tz.md)

The file `tz.md` contains a detailed architectural plan in Russian describing:

1. **Cascade-parallel pipeline** — preprocessing → binary classifier → parallel 3-branch detection (fuzzy dictionary + contrastive embeddings + knowledge graph) → meta-aggregator
2. **Preprocessing** — lowercasing, symbol removal, typo correction via fuzzy dictionary, transliteration normalization, lemmatization, hand-crafted feature extraction
3. **TypeQuery classifier** — gradient boosting on TF-IDF n-grams + hand-crafted features, optimized for F2 score
4. **Three detection branches:**
   - **Fuzzy dictionary** — exact match → fuzzy match (edit distance ≤2) → regex patterns for season/part/year
   - **Contrastive embeddings** — transformer model fine-tuned with triplet loss, prototype vectors per franchise, nearest-prototype search
   - **Knowledge graph** — nodes for franchises/actors/directors/genres/years, edges for semantic/production links, bounded graph traversal
5. **Meta-aggregator** — weighted voting across branches, conflict resolution via arbiter classifier, canonical form selection
6. **Confidence scoring** — per-branch confidence, inter-branch agreement, historical accuracy, isotonic regression calibration
7. **Incremental updates** — low-confidence queries routed for manual review feed back into incremental fine-tuning buffer

## Development Notes

- All training data is **confidential** — do not share or commit `train.csv`
- The sandbox has **no internet** except whitelisted LLM APIs
- Bundle excludes `data/`, `.venv/`, `.git/`, `notebooks/`, `scripts/` automatically
- Solutions must use **open-source, on-premise** technology stack
- Monthly data updates require support for periodic/incremental retraining
