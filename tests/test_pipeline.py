"""
Unit tests for TBCA pipeline.
Runs without GPU or network — uses keyword fallback ABSA.
"""

import numpy as np
import pytest
from app.pipeline import TBCAPipeline, ASPECTS, PT_ALPHA


@pytest.fixture(scope="module")
def pipeline():
    return TBCAPipeline()


# ── Core pipeline tests ───────────────────────────────────────────────────────

def test_pipeline_is_ready(pipeline):
    assert pipeline.is_ready()


def test_predict_single_returns_expected_keys(pipeline):
    result = pipeline.predict_single("Great performance but average battery life.")
    assert "utility_scores" in result
    assert "axiomatic_checks" in result


def test_utility_scores_cover_all_aspects(pipeline):
    result = pipeline.predict_single("The display is sharp and the build feels solid.")
    attributes = [u["attribute"] for u in result["utility_scores"]]
    assert set(attributes) == set(ASPECTS)


def test_utility_scores_have_required_fields(pipeline):
    result = pipeline.predict_single("Good value for money.")
    for u in result["utility_scores"]:
        assert "attribute" in u
        assert "raw_weight" in u
        assert "debiased_weight" in u


def test_weights_normalised(pipeline):
    """MAUT normalisation axiom: Σ wᵢ = 1."""
    assert abs(pipeline._weights.sum() - 1.0) < 1e-6


def test_all_weights_positive(pipeline):
    """MAUT monotonicity axiom: all wᵢ > 0."""
    assert (pipeline._weights > 0).all()


# ── Prospect Theory debiasing ─────────────────────────────────────────────────

def test_debias_positive_scores(pipeline):
    scores = np.array([0.5, 0.8, 0.3])
    debiased = pipeline._debias(scores)
    # Positive scores should remain positive
    assert (debiased >= 0).all()


def test_debias_negative_scores(pipeline):
    scores = np.array([-0.5, -0.8, -0.3])
    debiased = pipeline._debias(scores)
    # Loss aversion: debiased negative > raw negative in magnitude
    assert (np.abs(debiased) >= np.abs(scores) * PT_ALPHA * 0.9).all()


def test_debias_zero(pipeline):
    scores = np.zeros(5)
    debiased = pipeline._debias(scores)
    assert (debiased == 0).all()


# ── Choice / MNL ──────────────────────────────────────────────────────────────

def test_predict_choice_probs_sum_to_one(pipeline):
    alternatives = [
        ("laptop_a", "Excellent performance and great display quality."),
        ("laptop_b", "Good battery life and solid build quality."),
        ("laptop_c", "Decent value for money but slow performance."),
    ]
    result = pipeline.predict_choice(alternatives)
    probs = [a["choice_probability"] for a in result["alternatives"]]
    assert abs(sum(probs) - 1.0) < 1e-5


def test_predict_choice_recommended_is_valid(pipeline):
    alternatives = [
        ("a", "Amazing performance, excellent display."),
        ("b", "Poor battery, bad build quality."),
    ]
    result = pipeline.predict_choice(alternatives)
    assert result["recommended"] in ["a", "b"]


def test_predict_choice_probs_in_01(pipeline):
    alternatives = [
        ("x", "Great laptop overall."),
        ("y", "Terrible experience."),
    ]
    result = pipeline.predict_choice(alternatives)
    for a in result["alternatives"]:
        assert 0 < a["choice_probability"] < 1


# ── Axiomatic checks ──────────────────────────────────────────────────────────

def test_axiomatic_checks_pass(pipeline):
    result = pipeline.predict_single("Solid build and great value.")
    checks = result["axiomatic_checks"]
    assert checks["weights_sum_to_one"] is True
    assert checks["all_weights_positive"] is True
