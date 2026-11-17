"""Bilingual (English/Arabic) keyword retrieval over the Markdown knowledge base.

Each "## " section of a document is one chunk. Chunks are ranked with BM25,
using light normalisation and stemming for Arabic and English.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent / "kb"

ARABIC_LETTER = re.compile(r"[ء-ي]")
LATIN_LETTER = re.compile(r"[A-Za-z]")
TOKEN = re.compile(r"[A-Za-z0-9ء-ي]+")
# Harakat, superscript alef, Quranic marks and tatweel.
ARABIC_MARKS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")

EN_STOP = set(
    """a an the and or but if of to in on at by for from with about into over after before
    is are was were be been being do does did can could should would will shall may might must
    i me my we our you your he she it its they them their this that these those there here
    what which who whom whose when where why how much many any some all no not yes please
    have has had get got tell know want need""".split()
)

AR_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
AR_SUFFIXES = ("ات", "ون", "ين", "ها", "هم", "كم", "نا")


def detect_language(text: str) -> str:
    """Return "ar" when the text has more Arabic than Latin letters, else "en"."""
    return "ar" if len(ARABIC_LETTER.findall(text)) > len(LATIN_LETTER.findall(text)) else "en"


def normalize_arabic(text: str) -> str:
    text = ARABIC_MARKS.sub("", text)
    text = re.sub("[إأآا]", "ا", text)
    return text.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")


AR_STOP = {
    normalize_arabic(w)
    for w in """في من على الى إلى عن ما ماذا هل كم كيف متى اين أين هو هي هذا هذه ذلك التي الذي
    او أو و ثم مع لا نعم انا أنا انت أنت نحن لي لك عند كل اي أي يمكن يمكنني ممكن قبل بعد""".split()
}


def _stem_ar(token: str) -> str:
    for prefix in AR_PREFIXES:
        if token.startswith(prefix) and len(token) - len(prefix) >= 2:
            token = token[len(prefix):]
            break
    for suffix in AR_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    return token


def _stem_en(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in TOKEN.findall(normalize_arabic(text)):
        if ARABIC_LETTER.search(raw):
            if raw in AR_STOP:
                continue
            tokens.append(_stem_ar(raw))
        else:
            word = raw.lower()
            if word in EN_STOP:
                continue
            tokens.append(_stem_en(word))
    return tokens


@dataclass(frozen=True)
class Chunk:
    id: str  # e.g. "en/refunds-and-cancellations#late-deliveries"
    lang: str
    doc: str
    doc_title: str
    section: str
    text: str

    @property
    def anchor(self) -> str:
        return self.id.split("#", 1)[1]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    matched_terms: list[str] = field(default_factory=list)


def _slug(text: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9ء-ي]+", "-", normalize_arabic(text.lower())).strip("-")
    return slug or f"section-{index}"


def parse_markdown(path: Path, lang: str) -> list[Chunk]:
    doc = path.stem
    title, chunks, section, lines = doc, [], None, []

    def flush():
        if section is not None and lines:
            index = len(chunks) + 1
            chunks.append(
                Chunk(
                    id=f"{lang}/{doc}#{_slug(section, index)}",
                    lang=lang,
                    doc=doc,
                    doc_title=title,
                    section=section,
                    text=" ".join(line.strip() for line in lines if line.strip()),
                )
            )

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            section, lines = line[3:].strip(), []
        elif section is not None:
            lines.append(line)
    flush()
    return chunks


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        # The section heading is counted twice: it is the best summary of the chunk.
        self.docs = [Counter(tokenize(f"{c.section} {c.section} {c.text}")) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df = Counter(term for d in self.docs for term in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, top_k: int = 4) -> list[Hit]:
        terms = list(dict.fromkeys(tokenize(query)))
        hits = []
        for chunk, tf, length in zip(self.chunks, self.docs, self.lengths):
            score, matched = 0.0, []
            for term in terms:
                f = tf.get(term)
                if not f:
                    continue
                matched.append(term)
                denom = f + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                score += self.idf[term] * f * (self.k1 + 1) / denom
            if score > 0:
                hits.append(Hit(chunk, round(score, 3), matched))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]


@lru_cache(maxsize=1)
def load_indexes() -> dict[str, BM25Index]:
    indexes = {}
    for lang_dir in sorted(p for p in KB_DIR.iterdir() if p.is_dir()):
        chunks = [c for path in sorted(lang_dir.glob("*.md")) for c in parse_markdown(path, lang_dir.name)]
        if chunks:
            indexes[lang_dir.name] = BM25Index(chunks)
    return indexes


def search(query: str, lang: str | None = None, top_k: int = 4) -> list[Hit]:
    """Search the documents written in the question's language."""
    indexes = load_indexes()
    lang = lang or detect_language(query)
    index = indexes.get(lang) or indexes.get("en")
    return index.search(query, top_k) if index else []


def all_documents() -> dict[str, dict[str, list[Chunk]]]:
    """{lang: {doc: [chunks]}} for the knowledge-base pages."""
    out: dict[str, dict[str, list[Chunk]]] = {}
    for lang, index in load_indexes().items():
        for chunk in index.chunks:
            out.setdefault(lang, {}).setdefault(chunk.doc, []).append(chunk)
    return out
