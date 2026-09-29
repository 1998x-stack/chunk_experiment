from __future__ import annotations

from collections.abc import Sequence


class HttpRerankScoreProvider:
    """Generic HTTP rerank adapter.

    Request:
      {"model": "...", "query": "...", "documents": ["...", ...]}

    Accepted responses:
      {"scores": [0.1, 0.9]}
    or
      {"results": [{"index": 1, "relevance_score": 0.9}, ...]}
    """

    def __init__(
        self,
        url: str,
        *,
        model: str,
        timeout_seconds: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        if not url:
            raise ValueError("url is required")
        if not model:
            raise ValueError("model is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        self.url = url
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.headers = {
            "Content-Type": "application/json",
            **(headers or {}),
        }

    def score(
        self,
        query: str,
        texts: Sequence[str],
    ) -> Sequence[float]:
        if not texts:
            return []
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Install chunk-experiment[http] to use HTTP reranking"
            ) from exc

        response = requests.post(
            self.url,
            headers=self.headers,
            json={
                "model": self.model,
                "query": query,
                "documents": list(texts),
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("rerank service must return a JSON object")
        if "error" in payload:
            raise RuntimeError(f"rerank service error: {payload['error']}")

        scores = payload.get("scores")
        if scores is not None:
            return self._parse_scores(scores, expected=len(texts))

        results = payload.get("results")
        if results is not None:
            return self._parse_ranked_results(
                results,
                expected=len(texts),
            )
        raise RuntimeError(
            "rerank service response must contain 'scores' or 'results'"
        )

    @staticmethod
    def _parse_scores(
        values: object,
        *,
        expected: int,
    ) -> list[float]:
        if not isinstance(values, list) or len(values) != expected:
            raise RuntimeError(
                "rerank service returned invalid score cardinality"
            )
        try:
            return [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "rerank service scores must be numeric"
            ) from exc

    @staticmethod
    def _parse_ranked_results(
        values: object,
        *,
        expected: int,
    ) -> list[float]:
        if not isinstance(values, list) or len(values) != expected:
            raise RuntimeError(
                "rerank service returned invalid result cardinality"
            )
        resolved: list[float | None] = [None] * expected
        for row in values:
            if not isinstance(row, dict):
                raise RuntimeError("rerank result rows must be objects")
            index = row.get("index")
            if not isinstance(index, int) or not 0 <= index < expected:
                raise RuntimeError("rerank result index is invalid")
            if resolved[index] is not None:
                raise RuntimeError("rerank result index is duplicated")
            raw_score = (
                row.get("relevance_score")
                if "relevance_score" in row
                else row.get("score")
            )
            try:
                resolved[index] = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "rerank result score must be numeric"
                ) from exc
        if any(score is None for score in resolved):
            raise RuntimeError("rerank results did not cover every input index")
        return [float(score) for score in resolved if score is not None]
