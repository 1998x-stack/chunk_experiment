from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_arr = np.asarray(left, dtype=np.float32)
    right_arr = np.asarray(right, dtype=np.float32)
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"embedding shape mismatch: {left_arr.shape} != {right_arr.shape}")
    left_norm = float(np.linalg.norm(left_arr))
    right_norm = float(np.linalg.norm(right_arr))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left_arr, right_arr) / (left_norm * right_norm))


_TERM_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9_]+", re.UNICODE)


class HashEmbeddingProvider:
    """Deterministic lexical-hash embeddings for tests and offline baselines.

    This is deliberately *not* presented as a semantic model. It exists so CI,
    examples and experiment plumbing are reproducible without a network model.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 8:
            raise ValueError("dimension must be >= 8")
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for term in _TERM_RE.findall(text.lower()):
                digest = hashlib.blake2b(term.encode("utf-8"), digest_size=16).digest()
                bucket = int.from_bytes(digest[:8], "little") % self.dimension
                sign = 1.0 if digest[8] & 1 else -1.0
                result[row, bucket] += sign
        return l2_normalize(result)


class HttpEmbeddingProvider:
    """Adapter for the repository's ``textList -> resultList`` HTTP contract."""

    def __init__(
        self,
        url: str,
        *,
        model: str = "m3e",
        version: str = "m3e",
        timeout_seconds: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        if not url:
            raise ValueError("url is required")
        self.url = url
        self.model = model
        self.version = version
        self.timeout_seconds = timeout_seconds
        self.headers = {"Content-Type": "application/json", **(headers or {})}

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install chunk-experiment[http] to use HTTP embeddings") from exc

        response = requests.post(
            self.url,
            headers=self.headers,
            json={
                "model": self.model,
                "version": self.version,
                "uniqueId": "chunk-experiment",
                "textList": list(texts),
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"embedding service error: {payload['error']}")
        values = payload.get("data", {}).get("resultList")
        if not isinstance(values, list) or len(values) != len(texts):
            raise RuntimeError(
                f"embedding service returned {0 if values is None else len(values)} vectors "
                f"for {len(texts)} texts"
            )
        vectors = np.asarray(values, dtype=np.float32)
        if vectors.ndim != 2:
            raise RuntimeError("embedding service must return a 2D vector matrix")
        return l2_normalize(vectors)
