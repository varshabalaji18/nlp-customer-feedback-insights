"""Text normalization and input loading utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving useful punctuation and casing."""
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_feedback(path: str | Path, text_column: str = "text") -> pd.DataFrame:
    """Load CSV/TSV/JSON/TXT feedback into a canonical DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".tsv":
        frame = pd.read_csv(path, sep="\t")
    elif suffix == ".json":
        frame = pd.read_json(path)
    elif suffix in {".txt", ".text"}:
        frame = pd.DataFrame({text_column: path.read_text(encoding="utf-8").splitlines()})
    else:
        raise ValueError(f"Unsupported input type: {suffix}")

    if text_column not in frame.columns:
        candidates = [c for c in frame.columns if c.lower() in {"text", "review", "content", "body", "ticket"}]
        if not candidates:
            raise ValueError(f"Could not find a text column. Expected '{text_column}' or a common alias.")
        text_column = candidates[0]
    result = frame.copy()
    result["text"] = result[text_column].map(normalize_text)
    result = result[result["text"].str.len() > 0].reset_index(drop=True)
    result["feedback_id"] = result.index + 1
    return result


def chunk_texts(texts: Iterable[str], max_chars: int = 512) -> list[str]:
    """Conservatively chunk long texts for transformer token limits."""
    chunks: list[str] = []
    for text in texts:
        words = normalize_text(text).split()
        current: list[str] = []
        size = 0
        for word in words:
            if current and size + len(word) + 1 > max_chars:
                chunks.append(" ".join(current))
                current, size = [], 0
            current.append(word)
            size += len(word) + 1
        if current:
            chunks.append(" ".join(current))
    return chunks

