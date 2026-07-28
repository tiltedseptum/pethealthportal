# Agent Implementation & System Architecture Blueprint

## System Overview
This project builds an automated, full-cycle pet health portal. The system uses a headless automation agent to bypass manual scraping/download workflows on VetBuddy, processes medical records with high fidelity, indexes semantic chunks into a vector database, and exposes endpoints for an Emergent-based frontend chat interface.

## Rules
All execution must be on my approval. You will not touch any file or code or edit anything without seeking my permissions. You will not bypass this rule at any cost. 

## Tech Stack & Architecture
- **Frontend Layer:** Emergent (UI Component Engine, zero-boilerplate web panel)
- **API Middleware Layer:** FastAPI (Python, asynchronous endpoint execution)
- **Extraction Agent:** Playwright (Python, dynamic headless browser automation)
- **Document Processing:** LlamaParse (High-fidelity PDF-to-Markdown document parsing)
- **Orchestration & Chunking:** LangChain (Semantic `MarkdownTextSplitter`)
- **AI Gateway & LLM Models:** OpenRouter (Unified API Key, utilizing Free-Tier infrastructure)
  - *Embedding Model:* `nvidia/nemotron-3-embed-1b:free` (**1024 Dimensions**)
  - *Chat LLM Model:* `meta-llama/llama-3-8b-instruct:free` (or `openrouter/free`)
- **Database Engine:** MongoDB Atlas (M0 Shared Free Tier, storing vectors + raw markdown side-by-side)

---

## Workspace Directory Structure
The workspace must match the following layout exactly:
```text
├── .env                  # Secret API keys and database credentials (git-ignored)
├── main.py               # FastAPI server application and user-facing endpoints
├── health_agent.py       # Asynchronous ingestion pipeline (Playwright -> Parse -> Mongo)
└── requirements.txt      # Dependency configurations
```

---

## 🛠️ Step-by-Step Technical Execution Requirements

### 1. Environment Configurations (`.env`)
The system must read credentials strictly from a local `.env` configuration file using `python-dotenv`:
- `OPENROUTER_API_KEY`: Active master OpenRouter token string (`sk-or-...`).
- `LLAMA_CLOUD_API_KEY`: Standalone token from LlamaIndex dashboard (`llx-...`).
- `MONGO_URI`: Standard python driver connection string from MongoDB Atlas console dashboard.

### 2. Core Automation Ingestion Pipeline (`health_agent.py`)
This script must run sequentially when called by the server background worker:

- **Step A: Automated Download (Playwright):** 
  - Launch a hidden Chromium browser context (`headless=True` in production, `False` for local diagnostic verification).
  - Automate form navigation on VetBuddy: Log in securely, navigate to the export section, dynamically calculate date inputs (from 30 days ago to current date), select "Visit Details" from dropdown components, and handle checkboxes (like "Include Labs").
  - Intercept the file download explicitly using `page.expect_response()` to capture the PDF network response, then perform a separate direct fetch via `context.request.get()` (reusing the authenticated session) to retrieve the full file, avoiding Chrome's inline PDFium viewer and its partial/range responses. Save the output locally as `./latest_visit.pdf`.

- **Step B: Parsing & Markdown Transformation (LlamaParse):**
  - Pass the raw binary file directly into `LlamaParse` with `result_type="markdown"`. This maintains structural formatting, grids, and medical tabular context cleanly.

- **Step C: Chunking Strategy (LangChain):**
  - Initialize LangChain's native `MarkdownTextSplitter` with parameters explicitly tailored for medical data layouts:
    - **Chunk Size:** 512 tokens
    - **Chunk Overlap:** 50 tokens
  - The splitter must segment based on markdown header hierarchies (`#`, `##`, `###`) to ensure clinical events and lab modules are not cut in half mid-sentence.

- **Step D: 1024-Dimension Vectoring & Database Upsertion (MongoDB):**
  - Loop through generated text blocks and send the *entire raw sentence* (retaining connecting context tokens like "is" or "his") to OpenRouter's free `nvidia/nemotron-3-embed-1b:free` model.
  - The returned vector array has exactly **1024 dimensions**.
  - Bulk upsert documents into the MongoDB Atlas collection matching this JSON document model schema structure:
    ```json
    {
      "pet_id": "dog_rocky_01",
      "extracted_at": "ISO-TIMESTAMP",
      "chunk_index": 0,
      "text_content": "### LOGICAL BLOCK TEXT...",
      "vector_embedding": [0.123, -0.456, ..., 1024 total floats]
    }
    ```

### 3. API Middleware Layer & Service Endpoints (`main.py`)
Expose a lightweight **FastAPI** server containing exactly two cross-origin resource sharing (CORS) safe endpoints for the **Emergent** UI layout:

- **`POST /api/sync` (Trigger Sync Button Task):**
  - Triggers the complete `health_agent.py` background sequence asynchronously using FastAPI's `BackgroundTasks` tool, returning an immediate status message to the UI to keep components responsive.

- **`POST /api/chat` (Interactive Grounded Chat Window):**
  - Accepts a raw JSON request parameter payload: `{"user_question": "text content here"}`.
  - **Do not modify or clean stop-words.** Send the raw untouched query string straight to `nvidia/nemotron-3-embed-1b:free` to compile its 1024-dimension matrix array.
  - Query MongoDB Atlas using the `$vectorSearch` aggregation operator against the custom search index layout to isolate the top 3 highly relevant medical document text blocks.
  - Formulate an absolute, hallucination-free system context wrapper prompt using the isolated context blocks, passing it to `meta-llama/llama-3-8b-instruct:free` (via OpenRouter gateway base URL) to stream a grounded answer back to the Emergent screen interface.

---

## 🚨 Database Security & Manual Index Alert
*Before executing queries against MongoDB Atlas*, you must manually click **Create Search Index** inside your Atlas browser panel using the **JSON Editor** on your collection, and paste this exact dimension-matching object mapping configuration:
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
