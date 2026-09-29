import os
from typing import Dict, List, Optional

import requests


class EmbeddingClient:
    """Client for the repository's embedding HTTP service contract.

    The endpoint can be passed explicitly or read from ``EMBEDDING_URL``.
    Authentication/session headers must be injected by the caller; no credentials
    or cookies are stored in source code.
    """

    def __init__(
        self,
        embedding_url: Optional[str] = None,
        *,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 30.0,
    ):
        self.url = embedding_url or os.getenv("EMBEDDING_URL")
        if not self.url:
            raise ValueError(
                "Embedding URL must be provided either as an argument or via "
                "environment variable 'EMBEDDING_URL'."
            )
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")

        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.timeout_seconds = timeout_seconds

    def get_embeddings(
        self,
        text_list: List[str],
        model: str = "m3e",
        version: str = "m3e",
        unique_id: str = "chunk-experiment",
    ) -> dict:
        if not isinstance(text_list, list) or not all(isinstance(x, str) for x in text_list):
            raise TypeError("text_list must be List[str]")

        payload = {
            "model": model,
            "textList": text_list,
            "version": version,
            "uniqueId": unique_id,
        }

        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            response_data = response.json()
        except requests.RequestException as exc:
            raise ValueError(f"Embedding request failed: {exc}") from exc
        except ValueError as exc:
            raise ValueError("Embedding service returned invalid JSON") from exc

        if "error" in response_data:
            raise ValueError(f"Embedding API error: {response_data['error']}")

        result_list = response_data.get("data", {}).get("resultList")
        if not isinstance(result_list, list):
            raise ValueError("Embedding API response is missing data.resultList")
        if len(result_list) != len(text_list):
            raise ValueError(
                f"Embedding API returned {len(result_list)} vectors for "
                f"{len(text_list)} input texts"
            )
        return response_data

    def print_embeddings(self, text_list: List[str]) -> None:
        embeddings = self.get_embeddings(text_list)
        result_list = embeddings["data"]["resultList"]
        for text, embedding in zip(text_list, result_list):
            print(f"Text: {text}")
            print(f"Embedding: {embedding}")


if __name__ == "__main__":
    url = os.getenv("EMBEDDING_URL")
    if not url:
        raise SystemExit("Set EMBEDDING_URL before running this module directly.")
    EmbeddingClient(url).print_embeddings(["embedding smoke test"])
