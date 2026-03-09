"""
Sentence-BERT embedding utilities.

Loads all-MiniLM-L6-v2 once (module-level singleton) and exposes:
  - encode(texts)             -> np.ndarray  (N, D)
  - cosine_dist(a, b)         -> float in [0, 1]
  - intent_centroid(utterances) -> np.ndarray (D,)
"""

from __future__ import annotations
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Singleton model — loaded once at import time (lazy)
# ---------------------------------------------------------------------------

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Encode a list of strings into L2-normalised embeddings.

    Returns array of shape (len(texts), embedding_dim).
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.array(embeddings, dtype=np.float32)


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine distance in [0, 1] between two L2-normalised vectors.

    dist = (1 - cosine_similarity) / 2  so that identical → 0, opposite → 1.
    Using the half-angle formula keeps the range strictly [0, 1].
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # Ensure unit vectors (re-normalise for numerical safety)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 1.0
    sim = float(np.dot(a / a_norm, b / b_norm))
    sim = max(-1.0, min(1.0, sim))  # numerical clamp
    return (1.0 - sim) / 2.0


def intent_centroid(utterances: List[str]) -> np.ndarray:
    """
    Compute the mean embedding (centroid) for a list of utterances.

    Returns a 1-D array of shape (embedding_dim,).
    If utterances is empty, returns a zero vector.
    """
    if not utterances:
        return np.zeros(384, dtype=np.float32)
    embs = encode(utterances)
    centroid = embs.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid.astype(np.float32)


def pairwise_cosine_dist(
    embs_a: np.ndarray, embs_b: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise cosine distances between two sets of embeddings.

    Returns matrix of shape (len(embs_a), len(embs_b)).
    """
    # Both are assumed L2-normalised
    sims = embs_a @ embs_b.T  # (N, M)
    sims = np.clip(sims, -1.0, 1.0)
    return (1.0 - sims) / 2.0
