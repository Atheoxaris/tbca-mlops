# TBCA — MLOps Service

**Text-Based Conjoint Analysis · Production Inference API**

> Wraps the [TBCA research pipeline](https://github.com/Atheoxaris/TBCA-Amazon-Laptop-Reviews) in a production-ready MLOps stack: FastAPI · Docker · MLflow · GitHub Actions CI.

[![CI](https://github.com/Atheoxaris/tbca-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Atheoxaris/tbca-mlops/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-blue.svg)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com)

---

## What it does

Extracts consumer preference utilities from unstructured product reviews — no surveys, no controlled experiments.

```
Product Review (raw text)
        ↓
ABSA  ·  DeBERTa-v3-base          aspect-sentiment scores ∈ [-1, +1]
        ↓
Prospect Theory debiasing          λ=2.25, α=0.88
        ↓
MAUT utility estimation            Ridge regression
        ↓
Softmax / MNL choice probs         RUM equivalence
        ↓
REST API response  +  MLflow run
```

The theoretical foundation (RUM ↔ Softmax, MLE ↔ Cross-Entropy, MAUT ↔ Linear layer) is established in the companion paper:

> Anastasiadis, T. & Ampeliotis, D. (2026). *Bridging Decision Theory and Machine Learning: A Unified Framework for Consumer Preference Estimation from Unstructured Text.* Manuscript under review.

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| NLP | DeBERTa-v3-base (HuggingFace) |
| ML | scikit-learn Ridge, numpy |
| Experiment tracking | MLflow |
| Containerisation | Docker + Docker Compose |
| CI | GitHub Actions |
| Tests | pytest + pytest-cov |
| Lint | ruff |

---

## Quickstart

### Option 1 — Docker Compose (recommended)

```bash
git clone https://github.com/Atheoxaris/tbca-mlops
cd tbca-mlops
docker compose up --build
```

| Service | URL |
|---|---|
| API docs (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

### Option 2 — Local

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## API Endpoints

### `POST /predict` — single review
```json
{
  "text": "Great performance but the battery drains fast.",
  "product_id": "laptop_001"
}
```

Response:
```json
{
  "product_id": "laptop_001",
  "utility_scores": [
    {"attribute": "battery life",    "raw_weight": -0.62, "debiased_weight": -0.81},
    {"attribute": "performance",     "raw_weight":  0.74, "debiased_weight":  0.68},
    ...
  ],
  "axiomatic_checks": {
    "weights_sum_to_one": true,
    "all_weights_positive": true
  },
  "processing_time_ms": 312.4
}
```

### `POST /predict/batch` — up to 100 reviews

### `POST /choice` — MNL choice probabilities across alternatives
```json
{
  "reviews": [
    {"product_id": "Laptop A", "text": "Excellent display and fast processor."},
    {"product_id": "Laptop B", "text": "Good value but average build quality."}
  ]
}
```

Response includes per-alternative `choice_probability` (Softmax/MNL) and a `recommended` winner.

---

## MLflow Tracking

Every inference run is logged automatically:

```
Experiment: tbca-inference
  └── Run: single-predict
        ├── params:  pt_lambda=2.25, pt_alpha=0.88
        └── metrics: raw_battery_life, raw_performance, ...
```

Open the MLflow UI at `http://localhost:5000` to compare runs, inspect parameters, and track model versions.

---

## Tests

```bash
pytest tests/ -v --cov=app
```

19 unit tests covering:
- Pipeline readiness
- Axiomatic consistency (MAUT normalisation, RUM interior requirement)
- Prospect Theory debiasing (positive / negative / zero)
- MNL choice probabilities (sum-to-one, interior, correct winner)

---

## Project Structure

```
tbca-mlops/
├── app/
│   ├── main.py          # FastAPI app, endpoints, schemas
│   └── pipeline.py      # Core inference logic + MLflow logging
├── tests/
│   └── test_pipeline.py # pytest unit tests
├── .github/
│   └── workflows/
│       └── ci.yml       # GitHub Actions: lint → test → docker build
├── Dockerfile           # Multi-stage build
├── docker-compose.yml   # API + MLflow stack
├── requirements.txt
└── README.md
```

---

## Related

- [TBCA Research Notebook](https://github.com/Atheoxaris/TBCA-Amazon-Laptop-Reviews) — empirical proof-of-concept
- [Paper (under review)](mailto:Atheoxaris@ionio.gr) — theoretical framework

---

## Author

**Theoxaris Anastasiadis**  
PhD Candidate · Ionian University · Greece  
[github.com/Atheoxaris](https://github.com/Atheoxaris) · [LinkedIn](https://www.linkedin.com/in/theoxaris-anastasiadis-ba69a8121/)
