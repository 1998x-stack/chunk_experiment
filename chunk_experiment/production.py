from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Generic, TypeVar

import numpy as np

from .embeddings import EmbeddingProvider
from .rerank import RerankScoreProvider


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Serializable model identity without storing credentials or raw endpoint URLs."""

    kind: str
    provider: str
    model: str
    revision: str = ""
    endpoint_label: str = ""

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("kind is required")
        if not self.provider:
            raise ValueError("provider is required")
        if not self.model:
            raise ValueError("model is required")

    def fingerprint(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, str]:
        return {
            **asdict(self),
            "fingerprint": self.fingerprint(),
        }


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    requests: int
    backend_calls: int
    requested_items: int
    backend_items: int
    cache_hits: int
    cache_misses: int
    backend_characters: int

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            **asdict(self),
            "cache_hit_rate": self.cache_hit_rate,
        }


class _UsageCounter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.requests = 0
        self.backend_calls = 0
        self.requested_items = 0
        self.backend_items = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.backend_characters = 0

    def snapshot(self) -> ProviderUsage:
        return ProviderUsage(
            requests=self.requests,
            backend_calls=self.backend_calls,
            requested_items=self.requested_items,
            backend_items=self.backend_items,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            backend_characters=self.backend_characters,
        )


T = TypeVar("T")


class MemoryCache(Generic[T]):
    """Minimal explicit in-memory cache used by reproducible benchmark wrappers."""

    def __init__(self) -> None:
        self._values: dict[str, T] = {}

    def get(self, key: str) -> T | None:
        return self._values.get(key)

    def set(self, key: str, value: T) -> None:
        self._values[key] = value

    def clear(self) -> None:
        self._values.clear()

    def __len__(self) -> int:
        return len(self._values)


def _text_key(namespace: str, text: str) -> str:
    payload = f"{namespace}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pair_key(namespace: str, query: str, text: str) -> str:
    payload = f"{namespace}\0{query}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class CachedEmbeddingProvider:
    """Cache embeddings by model fingerprint + exact input text."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        identity: ModelIdentity,
    ) -> None:
        if identity.kind != "embedding":
            raise ValueError("embedding cache requires identity.kind='embedding'")
        self.provider = provider
        self.identity = identity
        self.cache: MemoryCache[np.ndarray] = MemoryCache()
        self._usage = _UsageCounter()

    def clear_cache(self, *, reset_usage: bool = False) -> None:
        self.cache.clear()
        if reset_usage:
            self._usage.reset()

    def reset_usage(self) -> None:
        self._usage.reset()

    def usage(self) -> ProviderUsage:
        return self._usage.snapshot()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self._usage.requests += 1
        self._usage.requested_items += len(texts)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        namespace = self.identity.fingerprint()
        values: list[np.ndarray | None] = [None] * len(texts)
        missing: dict[str, tuple[str, list[int]]] = {}

        for index, text in enumerate(texts):
            key = _text_key(namespace, text)
            cached = self.cache.get(key)
            if cached is not None:
                self._usage.cache_hits += 1
                values[index] = cached
                continue
            self._usage.cache_misses += 1
            if key not in missing:
                missing[key] = (text, [])
            missing[key][1].append(index)

        if missing:
            missing_items = list(missing.items())
            missing_texts = [entry[1][0] for entry in missing_items]
            backend = np.asarray(
                self.provider.embed(missing_texts),
                dtype=np.float32,
            )
            if backend.ndim != 2 or backend.shape[0] != len(missing_texts):
                raise ValueError(
                    "embedding provider returned invalid cached-batch shape "
                    f"{backend.shape}"
                )
            self._usage.backend_calls += 1
            self._usage.backend_items += len(missing_texts)
            self._usage.backend_characters += sum(
                len(text) for text in missing_texts
            )
            for row, (key, (text, indexes)) in enumerate(missing_items):
                del text
                vector = backend[row].copy()
                self.cache.set(key, vector)
                for index in indexes:
                    values[index] = vector

        if any(value is None for value in values):
            raise RuntimeError("embedding cache failed to populate all requested values")
        return np.stack(
            [value for value in values if value is not None],
            axis=0,
        ).astype(np.float32, copy=False)


class CachedRerankScoreProvider:
    """Cache rerank scores by model fingerprint + exact query/document pair."""

    def __init__(
        self,
        provider: RerankScoreProvider,
        *,
        identity: ModelIdentity,
    ) -> None:
        if identity.kind != "rerank":
            raise ValueError("rerank cache requires identity.kind='rerank'")
        self.provider = provider
        self.identity = identity
        self.cache: MemoryCache[float] = MemoryCache()
        self._usage = _UsageCounter()

    def clear_cache(self, *, reset_usage: bool = False) -> None:
        self.cache.clear()
        if reset_usage:
            self._usage.reset()

    def reset_usage(self) -> None:
        self._usage.reset()

    def usage(self) -> ProviderUsage:
        return self._usage.snapshot()

    def score(
        self,
        query: str,
        texts: Sequence[str],
    ) -> Sequence[float]:
        self._usage.requests += 1
        self._usage.requested_items += len(texts)
        if not texts:
            return []

        namespace = self.identity.fingerprint()
        scores: list[float | None] = [None] * len(texts)
        missing: dict[str, tuple[str, list[int]]] = {}

        for index, text in enumerate(texts):
            key = _pair_key(namespace, query, text)
            cached = self.cache.get(key)
            if cached is not None:
                self._usage.cache_hits += 1
                scores[index] = cached
                continue
            self._usage.cache_misses += 1
            if key not in missing:
                missing[key] = (text, [])
            missing[key][1].append(index)

        if missing:
            missing_items = list(missing.items())
            missing_texts = [entry[1][0] for entry in missing_items]
            backend_scores = list(
                self.provider.score(
                    query,
                    missing_texts,
                )
            )
            if len(backend_scores) != len(missing_texts):
                raise ValueError(
                    "rerank provider returned "
                    f"{len(backend_scores)} scores for {len(missing_texts)} texts"
                )
            self._usage.backend_calls += 1
            self._usage.backend_items += len(missing_texts)
            self._usage.backend_characters += sum(
                len(query) + len(text)
                for text in missing_texts
            )
            for raw_score, (key, (text, indexes)) in zip(
                backend_scores,
                missing_items,
                strict=True,
            ):
                del text
                score = float(raw_score)
                self.cache.set(key, score)
                for index in indexes:
                    scores[index] = score

        if any(score is None for score in scores):
            raise RuntimeError("rerank cache failed to populate all requested scores")
        return [float(score) for score in scores if score is not None]


@dataclass(frozen=True, slots=True)
class UnitPricing:
    """Optional user-supplied item pricing for comparable benchmark estimates."""

    embedding_usd_per_1k_items: float = 0.0
    rerank_usd_per_1k_pairs: float = 0.0

    def __post_init__(self) -> None:
        if self.embedding_usd_per_1k_items < 0:
            raise ValueError("embedding pricing must be >= 0")
        if self.rerank_usd_per_1k_pairs < 0:
            raise ValueError("rerank pricing must be >= 0")

    def estimate_usd(
        self,
        *,
        embedding_usage: ProviderUsage | None = None,
        rerank_usage: ProviderUsage | None = None,
    ) -> float:
        embedding_items = (
            embedding_usage.backend_items
            if embedding_usage is not None
            else 0
        )
        rerank_pairs = (
            rerank_usage.backend_items
            if rerank_usage is not None
            else 0
        )
        return (
            embedding_items / 1000.0 * self.embedding_usd_per_1k_items
            + rerank_pairs / 1000.0 * self.rerank_usd_per_1k_pairs
        )
