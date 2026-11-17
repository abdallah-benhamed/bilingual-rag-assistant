"""Pipeline guardrails, with Gemini mocked (no network in tests)."""
import pytest

from assistant import gemini, pipeline


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    # Embeddings unavailable: retrieval falls back to keyword search.
    monkeypatch.setattr(gemini, "embed", lambda *a, **k: (_ for _ in ()).throw(gemini.GeminiError("offline")))


def test_without_key_returns_the_best_section_with_its_citation():
    result = pipeline.answer("How much does delivery cost?")
    assert result.mode == "extractive" and not result.handoff
    assert result.sources[0]["id"] == "en/delivery-zones-and-fees#delivery-fees"
    assert "5 TND" in result.answer


def test_off_topic_question_hands_off_without_calling_the_model(with_key, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("the model must not be called")

    monkeypatch.setattr(gemini, "generate_json", fail)
    result = pipeline.answer("What will the weather be tomorrow?")
    assert result.handoff and result.reason == "no_matching_document"


def test_asking_for_a_person_hands_off_in_arabic():
    result = pipeline.answer("أريد التحدث مع موظف")
    assert result.handoff and result.lang == "ar" and result.reason == "customer_asked_for_person"


def test_generated_answer_keeps_only_valid_citations(with_key, monkeypatch):
    seen = {}

    def fake(system, prompt):
        seen["system"], seen["prompt"] = system, prompt
        return {"status": "answered", "answer": "Delivery costs 5 TND in the same city [S1].",
                "citations": ["S1", "S9"]}

    monkeypatch.setattr(gemini, "generate_json", fake)
    result = pipeline.answer("How much does delivery cost?")
    assert result.mode == "generated" and not result.handoff
    assert [s["label"] for s in result.sources] == ["S1"]
    assert "Reply in English" in seen["system"]
    assert "[S1] Delivery zones and fees > Delivery fees" in seen["prompt"]


def test_answer_without_citation_is_discarded(with_key, monkeypatch):
    monkeypatch.setattr(gemini, "generate_json",
                        lambda s, p: {"status": "answered", "answer": "It is free.", "citations": []})
    result = pipeline.answer("How much does delivery cost?")
    assert result.handoff and result.reason == "answer_without_citation"


@pytest.mark.parametrize("status, reason", [("not_found", "model_not_found"), ("needs_human", "model_needs_human")])
def test_model_can_refuse(with_key, monkeypatch, status, reason):
    monkeypatch.setattr(gemini, "generate_json", lambda s, p: {"status": status, "answer": "", "citations": []})
    result = pipeline.answer("How much does delivery cost?")
    assert result.handoff and result.reason == reason


def test_model_error_falls_back_to_extractive(with_key, monkeypatch):
    def broken(*args, **kwargs):
        raise gemini.GeminiError("HTTP 429")

    monkeypatch.setattr(gemini, "generate_json", broken)
    result = pipeline.answer("كم سعر التوصيل إلى جربة؟")
    assert result.mode == "extractive" and result.reason == "model_unavailable"
    assert result.answer.startswith("من وثائقنا:")


def test_hybrid_retrieval_uses_embeddings_when_available(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(pipeline, "_doc_vectors", None)
    index = pipeline.load_indexes()["en"]
    target = "en/business-accounts#cash-collection-for-businesses"

    def fake_embed(texts, task_type):
        if task_type == "RETRIEVAL_DOCUMENT":
            return [[1.0, 0.0] if c.id == target else [0.0, 1.0] for c in index.chunks]
        return [[1.0, 0.0]]

    monkeypatch.setattr(gemini, "embed", fake_embed)
    retrieved, mode = pipeline.retrieve("When do shops receive the money customers paid?", "en")
    assert mode == "hybrid"
    assert retrieved[0].chunk.id == target and retrieved[0].similarity == 1.0
    monkeypatch.setattr(pipeline, "_doc_vectors", None)


def test_generation_falls_back_to_the_next_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "retired-model")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "working-model")
    calls = []

    def fake_post(path, body, timeout):
        calls.append(path)
        if path.startswith("retired-model"):
            raise gemini.GeminiError("HTTP 503: high demand", 503)
        return {"candidates": [{"content": {"parts": [{"text": '{"status": "not_found"}'}]}}]}

    monkeypatch.setattr(gemini, "_post", fake_post)
    assert gemini.generate_json("system", "prompt") == {"status": "not_found"}
    assert calls == ["retired-model:generateContent", "working-model:generateContent"]


def test_generation_does_not_retry_on_other_errors(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = []

    def fake_post(path, body, timeout):
        calls.append(path)
        raise gemini.GeminiError("HTTP 400: bad request", 400)

    monkeypatch.setattr(gemini, "_post", fake_post)
    with pytest.raises(gemini.GeminiError):
        gemini.generate_json("system", "prompt")
    assert len(calls) == 1
