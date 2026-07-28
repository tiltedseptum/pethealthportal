# pethealthsystem

An agentic ingestion + retrieval pipeline that turns a pet's exported vet medical records into a grounded, chat-queryable knowledge base. Built for my dog Theo; designed to generalize further (see [Roadmap](#roadmap)).

## Why this exists

This started as a Next.js/Supabase pet health portal. Partway through, I re-scoped it: rather than a manual record-keeping UI, I wanted an agent that pulls Theo's medical history on its own and lets me ask plain-language questions about it ("when was his last rabies shot?", "what's he been treated for this year?") with answers grounded in the actual record, not a language model's guesswork. This repo is that rebuild.

## Architecture

Two agents, chained by a shared MongoDB Atlas collection:

**Agent 1 — ingestion pipeline (`health_agent.py`)**
| Step | What it does | Tech |
|---|---|---|
| A. Download | Logs into the vet portal, navigates to the medical record export flow, and downloads the full visit history as a PDF | Playwright (async, Python) |
| B. Parse | Converts the PDF to structured markdown, preserving tables (labs, vitals history) | LlamaParse (`llama-cloud` SDK, `agentic` tier) |
| C. Chunk | Splits the markdown into ~512-token chunks (50-token overlap) along header boundaries | LangChain `MarkdownTextSplitter` + tiktoken |
| D. Embed + store | Embeds each chunk and upserts it into MongoDB Atlas, keyed so re-syncing the same visit doesn't duplicate it | OpenRouter (`nvidia/nemotron-3-embed-1b:free`) + MongoDB Atlas |

**Agent 2 — API server (`main.py`)**

A FastAPI app with two endpoints:
- `POST /api/sync` — runs Agent 1's full pipeline as a background task.
- `POST /api/chat` — embeds a raw user question, runs a MongoDB `$vectorSearch` for the top 3 relevant chunks, and asks an LLM to answer using only that retrieved context (grounded / hallucination-resistant by design).

## Status

Built and verified end-to-end against a real vet portal account and a live MongoDB Atlas cluster:

- Step A (download): confirmed working against the live portal.
- Step B (parse): confirmed producing clean structured markdown, including table extraction for lab panels.
- Step C (chunk): confirmed token-accurate chunking within the configured size/overlap.
- Step D (embed + upsert): confirmed real embeddings written to Atlas, with content-hash-based dedup so re-running the pipeline doesn't duplicate unchanged visits.
- Atlas Search vector index: created and queryable.

Not yet tested live: the `/api/sync` and `/api/chat` FastAPI endpoints themselves (the pipeline functions they call have been verified individually, but not through the HTTP layer).

## Known limitations

- **Single portal, single pet by design.** The download step (Step A) is written against one specific vet portal's login flow and DOM. It won't work against a different clinic's system without rewriting the selectors.
- **Dedup is text-hash-based**, and the parser (LlamaParse's `agentic` tier) isn't guaranteed to render the exact same PDF into byte-identical markdown across separate parses. In practice this means re-syncing can occasionally re-insert a visit under new dedup keys instead of recognizing it as already-seen. A more robust fix (dedup keyed on stable extracted facts -- visit date/time/doctor -- rather than raw text) is scoped but not yet implemented.
- **Free-tier LLM APIs.** OpenRouter's free-tier models log prompts to improve their services. Fine for a personal project; not appropriate for other people's data without disclosure or a paid tier.

## Roadmap

The current design is intentionally narrow (one pet, one portal, credentials stored locally) because it was built to solve a real, specific problem first. The next phase I'm considering:

- **Move from portal automation to PDF upload.** Instead of Playwright-scraping one specific vet portal (which requires storing that portal's login credentials), let the user export their own record from *any* vet system and upload the PDF directly. This removes the credential-custody problem entirely and makes the ingestion step portal-agnostic by construction.
- **Replace regex/text-hash logic with LLM-based structured extraction.** Extract a normalized schema (visit date, time, doctor, event type, treatment) from the parsed markdown using a schema-constrained LLM call, rather than pattern-matching one portal's specific formatting. This fixes the dedup fragility above and generalizes to any vet's PDF layout, not just one clinic's.
- **Multi-pet / multi-user support.** Real accounts and per-user data isolation, once ingestion no longer assumes a single hardcoded pet and portal.

## Setup

```bash
cd pethealthsystem
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
playwright install chromium

cp .env.example .env
# fill in real values in .env
```

You'll also need a MongoDB Atlas Search vector index on the `medical_record_chunks` collection, named `vector_index`:

```json
{
  "fields": [
    {
      "numDimensions": 1024,
      "path": "vector_embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

### Running the pipeline

```bash
python health_agent.py            # full pipeline (Steps A-D), visible browser for diagnostics
```

Individual steps can be exercised in isolation with the included test scripts (each is read-only except where noted, and documents what it does in its own header comment):

```bash
python test_step_b.py             # PDF -> markdown, from an existing latest_visit.pdf
python test_step_c.py             # markdown -> chunks, from an existing parsed_output.md
python test_step_d.py             # chunks -> embeddings + Mongo upsert (real write)
python check_mongo_state.py       # read-only inspection of what's stored in Atlas
python cleanup_test_data.py       # deletes all documents for PET_ID (asks for confirmation)
```

### Running the API

```bash
uvicorn main:app --reload
```

## Design notes

[`AGENTS.md`](./AGENTS.md) in this repo is the original technical blueprint this system was built from -- worth a look if you want the full rationale behind specific choices (chunk size, embedding model, dedup schema).
