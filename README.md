# Bilingual RAG assistant (English / Arabic): Django + Gemini

A customer-support assistant for a fictional delivery company in southern Tunisia. It answers in English or Arabic, only from the company's documents, cites the section it used, and hands the customer to a person when it is not sure. Every message, source and handoff is logged in the Django admin.

**Live demo:** _add your Render link here_

## What it shows

- **Answers grounded in documents.** Retrieval finds the relevant sections; Gemini answers only from them and must cite them (`S1`, `S2`…). Each citation links to the exact section in the knowledge base.
- **Arabic handled properly.** Language detection, Arabic normalisation (diacritics, alef forms, taa marbuta) and light stemming for search; right-to-left layout in the chat and the documents.
- **Guardrails against wrong answers.**
  1. The customer asks for a person: hand off at once.
  2. No section scores above the keyword and embedding thresholds: hand off without calling the model.
  3. The model may reply `not_found` or `needs_human`: hand off.
  4. Citations are checked against the sources actually sent; an answer with no valid citation is discarded.
  5. If Gemini is down or rate-limited, the assistant returns the best matching section word for word instead of failing.
- **Handoff and logs.** The customer leaves a name and phone or email; the request lands in the admin (and optionally by email) with the question attached. Each answer stores its mode, sources, scores and latency.
- **Cost control.** Per-session rate limit; the model is not called for off-topic questions; runs without any API key in extractive mode.

## How it works

```
question ─► language detection ─► hybrid search (BM25 + Gemini embeddings, rank fusion)
                                        │
                     nothing relevant ──┴──► handoff form
                                        │
                        top 4 sections ─► Gemini (JSON: status, answer, citations)
                                        │
                  invalid / not found ──┴──► handoff form
                                        │
                                        ▼
                          answer + cited sections ─► logged in admin
```

| Part | File |
| --- | --- |
| Knowledge base (Markdown, one `##` section = one chunk) | `assistant/kb/en`, `assistant/kb/ar` |
| Keyword search, Arabic normalisation | `assistant/retrieval.py` |
| Hybrid retrieval, prompt, guardrails | `assistant/pipeline.py` |
| Gemini REST client (no SDK) | `assistant/gemini.py` |
| Chat API, handoff API, rate limit | `assistant/views.py` |
| Logs and handoffs | `assistant/models.py`, Django admin |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export DJANGO_DEBUG=1                # local only
export GEMINI_API_KEY=...            # optional: without it, extractive mode
python manage.py migrate
python manage.py createsuperuser     # to read the logs at /admin/
python manage.py runserver
```

Tests (Gemini is mocked, no network needed):

```bash
pytest
```

## Deploy on Render (free plan)

1. Push this folder to a GitHub repository.
2. In Render: **New > Blueprint**, pick the repository. `render.yaml` creates the web service.
3. Fill in the two secret values Render asks for: `GEMINI_API_KEY` (from Google AI Studio) and `DJANGO_SUPERUSER_PASSWORD` (for `/admin/`, user `admin`).
4. Open the URL Render gives you.

Free-plan limits: the service sleeps after 15 minutes without traffic (the first visit then takes about a minute), and the SQLite database is reset on each restart or deploy. For lasting logs, point the app at a managed PostgreSQL database.

## Use it for another business

Replace the Markdown files in `assistant/kb/en` and `assistant/kb/ar` (same file names in both folders), edit the company name in `SYSTEM_PROMPT` and the templates, and redeploy. For PDFs, convert them to Markdown first and keep one topic per `##` section so citations stay precise.

## Configuration

See `.env.example`. Main settings: `GEMINI_MODEL` (default `gemini-2.5-flash`), `MIN_BM25_SCORE` and `MIN_EMBED_SIMILARITY` (thresholds before the model is called), `CHAT_MESSAGES_PER_HOUR`, `HANDOFF_EMAIL`.

Note: on Gemini's free tier, Google may use prompts to improve its products. Use a paid key before loading real customer data.
