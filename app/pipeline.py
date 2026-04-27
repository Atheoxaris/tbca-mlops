"""
TBCA Pipeline — core inference logic with MLflow experiment tracking.

Architecture mirrors the research notebook but restructured for:
  - Production inference (single + batch)
  - MLflow run logging
  - Stateless per-request execution
"""

from __future__ import annotations

import os
import time
import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)

# ── Prospect Theory constants (Tversky & Kahneman, 1992) ──────────────────────
PT_LAMBDA = 2.25   # loss aversion coefficient
PT_ALPHA  = 0.88   # curvature parameter

# ── Laptop aspects (SemEval-2016 Laptop14 taxonomy) ──────────────────────────
ASPECTS = [
    "battery life",
    "display quality",
    "performance",
    "build quality",
    "value for money",
]

# MLflow experiment name
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "tbca-inference")
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


class TBCAPipeline:
    """
    End-to-end TBCA inference pipeline.

    Lazy-loads the ABSA model on first call to keep startup fast.
    Logs every training/calibration run to MLflow.
    """

    def __init__(self):
        self._absa_model = None
        self._absa_tokenizer = None
        self._weights: Optional[np.ndarray] = None
        self._ready = False
        self._setup_mlflow()
        self._load_weights()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_mlflow(self):
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(EXPERIMENT_NAME)
            logger.info(f"MLflow tracking: {MLFLOW_URI} | experiment: {EXPERIMENT_NAME}")
        except Exception as e:
            logger.warning(f"MLflow setup failed (offline mode): {e}")

    def _load_absa_model(self):
        """Lazy-load DeBERTa-v3 ABSA model."""
        if self._absa_model is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            model_name = "yangheng/deberta-v3-base-absa-v1.1"
            logger.info(f"Loading ABSA model: {model_name}")
            self._absa_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._absa_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._absa_model.eval()
            logger.info("ABSA model loaded.")
        except Exception as e:
            logger.error(f"ABSA model load failed: {e}")
            raise

    def _load_weights(self):
        """
        Load pre-fitted MAUT weights (from research notebook results).
        In production these would be loaded from MLflow Model Registry.
        """
        # Debiased weights from paper (Table 1, n=200, B=1000 bootstrap)
        self._weights = np.array([0.0199, 0.1747, 0.2836, 0.2454, 0.2763])
        self._weights /= self._weights.sum()   # ensure normalisation
        self._ready = True
        logger.info("MAUT weights loaded (pre-fitted from paper).")

    def is_ready(self) -> bool:
        return self._ready

    # ── ABSA ──────────────────────────────────────────────────────────────────

    def _absa_scores(self, text: str) -> np.ndarray:
        """
        Return aspect-sentiment scores s_ij ∈ [-1, +1] for each aspect.
        Falls back to keyword heuristic if model unavailable (CI-safe).
        """
        try:
            self._load_absa_model()
            import torch

            scores = []
            for aspect in ASPECTS:
                inputs = self._absa_tokenizer(
                    aspect, text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                with torch.no_grad():
                    logits = self._absa_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze().numpy()
                # labels: [negative, neutral, positive]
                score = float(probs[2] - probs[0])   # ∈ [-1, +1]
                scores.append(score)
            return np.array(scores)

        except Exception:
            logger.warning("ABSA model unavailable — using keyword heuristic fallback.")
            return self._keyword_fallback(text)

    def _keyword_fallback(self, text: str) -> np.ndarray:
        """Lightweight keyword heuristic for CI / unit-test environments."""
        positive = {"great", "excellent", "good", "love", "fast", "sharp", "solid", "worth"}
        negative = {"bad", "poor", "slow", "dim", "flimsy", "expensive", "worst", "terrible"}
        tokens = set(text.lower().split())
        pos = len(tokens & positive)
        neg = len(tokens & negative)
        base = np.zeros(len(ASPECTS))
        # Distribute crude sentiment evenly
        if pos + neg > 0:
            base += (pos - neg) / (pos + neg) * 0.5
        return np.clip(base + np.random.uniform(-0.1, 0.1, len(ASPECTS)), -1, 1)

    # ── Prospect Theory debiasing ─────────────────────────────────────────────

    def _debias(self, scores: np.ndarray) -> np.ndarray:
        """Apply Prospect Theory value function (Tversky & Kahneman, 1992)."""
        debiased = np.where(
            scores >= 0,
            scores ** PT_ALPHA,
            -PT_LAMBDA * ((-scores) ** PT_ALPHA),
        )
        return debiased

    # ── MAUT utility ─────────────────────────────────────────────────────────

    def _maut_utility(self, debiased_scores: np.ndarray) -> float:
        """Additive MAUT: V = Σ wᵢ · sᵢ"""
        return float(np.dot(self._weights, debiased_scores))

    # ── Axiomatic checks ──────────────────────────────────────────────────────

    def _axiomatic_checks(self, weights: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
        return {
            "weights_sum_to_one":   bool(abs(weights.sum() - 1.0) < 1e-6),
            "all_weights_positive": bool((weights > 0).all()),
            "probs_sum_to_one":     bool(abs(probs.sum() - 1.0) < 1e-6) if probs is not None else None,
            "all_probs_in_01":      bool(((probs > 0) & (probs < 1)).all()) if probs is not None else None,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Full pipeline for one review → utility scores."""
        raw_scores      = self._absa_scores(text)
        debiased_scores = self._debias(raw_scores)

        utility_scores = [
            {
                "attribute":      ASPECTS[i],
                "raw_weight":     round(float(raw_scores[i]), 4),
                "debiased_weight": round(float(debiased_scores[i]), 4),
            }
            for i in range(len(ASPECTS))
        ]

        checks = self._axiomatic_checks(self._weights, None)

        # Log to MLflow
        try:
            with mlflow.start_run(run_name="single-predict", nested=True):
                mlflow.log_params({"pt_lambda": PT_LAMBDA, "pt_alpha": PT_ALPHA})
                mlflow.log_metrics({
                    f"raw_{ASPECTS[i].replace(' ', '_')}": float(raw_scores[i])
                    for i in range(len(ASPECTS))
                })
        except Exception:
            pass   # non-blocking

        return {"utility_scores": utility_scores, "axiomatic_checks": checks}

    def predict_choice(self, alternatives: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        MNL choice probabilities across multiple product alternatives.

        alternatives: list of (product_id, review_text)
        """
        utilities = []
        for pid, text in alternatives:
            raw      = self._absa_scores(text)
            debiased = self._debias(raw)
            v        = self._maut_utility(debiased)
            utilities.append((pid, v))

        # Softmax = MNL choice probabilities
        vs    = np.array([u for _, u in utilities])
        exp_v = np.exp(vs - vs.max())   # numerically stable
        probs = exp_v / exp_v.sum()

        results = [
            {
                "alternative":      pid,
                "utility":          round(float(v), 4),
                "choice_probability": round(float(p), 4),
            }
            for (pid, v), p in zip(utilities, probs)
        ]

        recommended = utilities[int(np.argmax(vs))][0]

        # Log to MLflow
        try:
            with mlflow.start_run(run_name="choice-predict", nested=True):
                mlflow.log_params({"n_alternatives": len(alternatives)})
                for r in results:
                    mlflow.log_metric(
                        f"prob_{r['alternative'].replace(' ', '_')}",
                        r["choice_probability"],
                    )
        except Exception:
            pass

        return {
            "alternatives": results,
            "recommended":  recommended,
            "axiomatic_checks": self._axiomatic_checks(self._weights, probs),
        }
