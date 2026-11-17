"""Minimal Gemini REST client (no SDK): JSON generation and text embeddings."""
from __future__ import annotations

import json
import os

import requests

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiError(Exception):
    pass


def api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or None


def _post(path: str, body: dict, timeout: float) -> dict:
    key = api_key()
    if not key:
        raise GeminiError("GEMINI_API_KEY is not set")
    try:
        response = requests.post(
            f"{BASE_URL}/{path}",
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=body,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise GeminiError(f"network error: {exc.__class__.__name__}") from exc
    if response.status_code != 200:
        raise GeminiError(f"HTTP {response.status_code}: {response.text[:200]}")
    return response.json()


def generate_json(system: str, prompt: str, timeout: float = 30) -> dict:
    """Ask the model for a JSON object and return it parsed."""
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    data = _post(
        f"{model}:generateContent",
        {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
        },
        timeout,
    )
    try:
        candidate = data["candidates"][0]
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        return json.loads(text)
    except (KeyError, IndexError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        reason = data.get("candidates", [{}])[0].get("finishReason") if isinstance(data, dict) else None
        raise GeminiError(f"unexpected model response (finishReason={reason})") from exc


def embed(texts: list[str], task_type: str, timeout: float = 30) -> list[list[float]]:
    """Embed texts with Gemini; task_type is RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY."""
    model = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    data = _post(
        f"{model}:batchEmbedContents",
        {
            "requests": [
                {
                    "model": f"models/{model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                    "outputDimensionality": 768,
                }
                for text in texts
            ]
        },
        timeout,
    )
    try:
        return [item["values"] for item in data["embeddings"]]
    except (KeyError, TypeError) as exc:
        raise GeminiError("unexpected embedding response") from exc
