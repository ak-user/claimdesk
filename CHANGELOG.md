# Changelog

## 0.2.0 - 2026-10-24

- Follow-up questions: `/api/ask` takes the recent conversation turns.
- Examiner queue: `POST /api/claims`, `GET /api/claims`, `POST /api/claims/{id}/review`, stored in
  SQLite (`CLAIMDESK_DB_PATH`), and an Examiner queue view in the UI.
- Rate limits on the routes that can call the model, and `X-Request-ID` on every response and
  log line.
- Optional Azure AI Search index (`AZURE_SEARCH_ENDPOINT`) behind a `BookletIndex` protocol, with
  `python -m claimdesk.azure_search` to build it.
- Eval: 37 rows including follow-ups and guardrail cases; refusal accuracy and latency metrics.
- Fix: "If I leave my job..." returned the appeal deadline instead of the 90-day deadline after
  coverage ends.

## 0.1.0 - 2026-09-08

- Benefits Q&A with citations: section chunking, BM25 plus embeddings with RRF.
- Tool-calling agent on Azure OpenAI, with an offline planner.
- Claim intake: text and PDF extraction, plan rule checks and payment estimates.
- Guardrails: citation check, prompt injection flags, scope refusals and PII masking.
- Eval script with a CI quality gate, React UI, Docker image and Azure deployment guide.
