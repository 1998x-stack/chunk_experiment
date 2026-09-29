from __future__ import annotations

import re
from typing import Protocol


class LengthMetric(Protocol):
    def __call__(self, text: str) -> int: ...


def character_length(text: str) -> int:
    """Count Unicode code points. This is exact and dependency-free."""

    return len(text)


# Conservative, language-neutral approximation used only when an actual model
# tokenizer is unavailable. CJK characters, Latin words/numbers and visible
# punctuation each count as one unit.
_APPROX_TOKEN_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]"
    r"|[A-Za-z]+(?:'[A-Za-z]+)?"
    r"|\d+(?:\.\d+)?"
    r"|[^\s]",
    re.UNICODE,
)


def approximate_token_length(text: str) -> int:
    """Estimate token count without pretending whitespace == tokens.

    This is deterministic and works for unsegmented Chinese text. It is still
    an approximation; production experiments should inject the target model's
    tokenizer via the ``LengthMetric`` protocol.
    """

    return len(_APPROX_TOKEN_RE.findall(text))
