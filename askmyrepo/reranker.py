"""Reorder a short list with local Qwen3-Reranker-0.6B (it reads question and passage together)."""

from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder

from askmyrepo.vector_index import Hit

MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
MAX_TOKENS = 512  # only the start of a long passage is read: same recall (27/30), 3.0 s not 4.4
# Measured on an M1 Pro (2026-11-18), 20 passages: float32 took 6 s on the CPU against 91 s in
# the model's default bfloat16; batches of 4 waste less time padding short passages than 8 or 20.
BATCH_SIZE = 4


# Loads the local Qwen3 reranker once and reuses it.
@lru_cache(maxsize=1)
def load_model() -> CrossEncoder:
    """Load the model once (the first call downloads it, about 1.2 GB)."""
    model = CrossEncoder(MODEL_NAME, model_kwargs={"dtype": torch.float32})
    model.max_length = MAX_TOKENS
    return model


# Re-orders candidates by reading question and passage together.
def rerank_top(question: str, hits: list[Hit], keep: int = 5, model=None) -> list[Hit]:
    """Score every (question, passage) pair, keep the best `keep`, with the new scores."""
    if not hits:
        return []
    model = model or load_model()
    scores = model.predict([(question, hit.text) for hit in hits], batch_size=BATCH_SIZE)
    ranked = sorted(zip(scores, hits, strict=True), key=lambda pair: pair[0], reverse=True)
    reranked = []
    for score, hit in ranked[:keep]:
        hit.score = float(score)
        reranked.append(hit)
    return reranked
