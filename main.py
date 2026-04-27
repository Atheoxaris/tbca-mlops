"""
TBCA — Text-Based Conjoint Analysis
FastAPI inference service
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import logging
import time

from app.pipeline import TBCAPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TBCA Inference API",
    description=(
        "Text-Based Conjoint Analysis — extracts consumer preference utilities "
        "from unstructured product reviews using ABSA + MAUT + RUM/MNL equivalence."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pipeline once at startup
pipeline = TBCAPipeline()


# ---------- Schemas ----------

class ReviewInput(BaseModel):
    text: str = Field(..., min_length=10, example="Great battery life but the display is average.")
    product_id: Optional[str] = Field(None, example="laptop_001")


class BatchReviewInput(BaseModel):
    reviews: List[ReviewInput] = Field(..., min_items=1, max_items=100)


class UtilityScore(BaseModel):
    attribute: str
    raw_weight: float
    debiased_weight: float


class AlternativeChoice(BaseModel):
    alternative: str
    utility: float
    choice_probability: float


class PredictResponse(BaseModel):
    product_id: Optional[str]
    utility_scores: List[UtilityScore]
    axiomatic_checks: dict
    processing_time_ms: float


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total_reviews: int
    processing_time_ms: float


class ChoiceResponse(BaseModel):
    alternatives: List[AlternativeChoice]
    recommended: str
    processing_time_ms: float


# ---------- Endpoints ----------

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "TBCA Inference API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model_loaded": pipeline.is_ready()}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(review: ReviewInput):
    """
    Extract part-worth utility weights from a single product review.

    - Runs ABSA (DeBERTa-v3) to extract aspect-sentiment scores
    - Applies Prospect Theory debiasing (λ=2.25, α=0.88)
    - Estimates MAUT utility weights via Ridge regression
    - Returns axiomatic consistency checks (RUM/MAUT)
    """
    t0 = time.perf_counter()
    try:
        result = pipeline.predict_single(review.text)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    return PredictResponse(
        product_id=review.product_id,
        utility_scores=result["utility_scores"],
        axiomatic_checks=result["axiomatic_checks"],
        processing_time_ms=elapsed,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(payload: BatchReviewInput):
    """
    Process up to 100 reviews in a single request.
    Returns per-review utility scores aggregated across the batch.
    """
    t0 = time.perf_counter()
    results = []
    for review in payload.reviews:
        try:
            result = pipeline.predict_single(review.text)
            results.append(
                PredictResponse(
                    product_id=review.product_id,
                    utility_scores=result["utility_scores"],
                    axiomatic_checks=result["axiomatic_checks"],
                    processing_time_ms=0,
                )
            )
        except Exception as e:
            logger.warning(f"Skipped review {review.product_id}: {e}")

    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    return BatchPredictResponse(
        results=results,
        total_reviews=len(results),
        processing_time_ms=elapsed,
    )


@app.post("/choice", response_model=ChoiceResponse, tags=["Inference"])
def choice(reviews: BatchReviewInput):
    """
    Given reviews for multiple product alternatives, compute MNL choice probabilities
    and recommend the highest-utility alternative.

    Each review.product_id is treated as a distinct alternative label.
    """
    t0 = time.perf_counter()
    try:
        choice_result = pipeline.predict_choice(
            [(r.product_id or f"alt_{i}", r.text) for i, r in enumerate(reviews.reviews)]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    return ChoiceResponse(
        alternatives=choice_result["alternatives"],
        recommended=choice_result["recommended"],
        processing_time_ms=elapsed,
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
