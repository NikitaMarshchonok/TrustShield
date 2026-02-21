from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    cleaned = text.lower().strip()
    cleaned = re.sub(r"https?://\S+", " url_token ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
