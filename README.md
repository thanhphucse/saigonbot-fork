# SaigonBot — Multi-Tenant Chatbot API

A multi-tenant RAG conversational AI chatbot API built with FastAPI, ChromaDB, Ollama embeddings, and OpenAI (default) or local Ollama generation.

One codebase serves two assistants selected via `TENANT` in `.env`:
- **`chikku`** *(default)* — Chikku Robotics support assistant (`chikku_db/` / `chikku_kb`)
- **`saigon`** — Saigon Indian Restaurant host & menu assistant (`chroma_db/` / `hotel_saigon_menu`)

Questions can be asked in **ten languages** — English, Hindi, Tamil, Bengali, Gujarati, Malayalam, Kannada, Telugu, Punjabi, and Vietnamese — against the same English vector index (see [Multilingual](#multilingual)).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Tenants](#tenants)
- [Data Ingestion](#data-ingestion)
- [Running the API](#running-the-api)
- [API Endpoints](#api-endpoints)
- [Switching LLM Provider](#switching-llm-provider)
- [Multilingual](#multilingual)
- [Testing](#testing)
- [Windows Standalone Package](#windows-standalone-package)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

---

## Prerequisites

1. **Git** (`git --version`)

2. **Python 3.10+** (verified on Python 3.13; `pip` and `venv` included)
   - **Windows (PowerShell):**
     ```powershell
     python --version
     ```
   - **Linux / macOS (bash):**
     ```bash
     python3 --version
     ```

3. **Ollama** — always required for local embeddings (`mxbai-embed-large`), even when `LLM_PROVIDER=openai`
   - **Windows (PowerShell):**
     ```powershell
     winget install Ollama.Ollama
     ```
     *(Or download the installer from [ollama.com/download](https://ollama.com/download).)*
   - **Linux / macOS (bash):**
     ```bash
     curl -fsSL https://ollama.com/install.sh | sh
     ```
   - Start the Ollama server if it is not already running in the background:
     ```bash
     ollama serve
     ```
   - Pull the embedding model (**always required**):
     ```bash
     ollama pull mxbai-embed-large
     ```
   - *(Optional)* Pull a local chat model only if you plan to set `LLM_PROVIDER=ollama`:
     ```bash
     ollama pull llama3:8b-instruct-q4_0
     ```

4. **OpenAI API Key** — required when using the default `LLM_PROVIDER=openai`.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/thanhphucse/saigonbot-fork.git
cd saigonbot-fork
```

*(If using SSH: `git clone git@github.com:thanhphucse/saigonbot-fork.git`.)*

### 2. Create and activate a virtual environment (`.venv`)

- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
  *(If PowerShell blocks script execution, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt: `.venv\Scripts\activate.bat`.)*
- **Linux / macOS (bash):**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install dependencies

With `.venv` activated (`python` now refers to the virtual environment on both OSes):

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Linux / macOS shortcut:** You can also run `./setup.sh` (`chmod +x setup.sh && ./setup.sh`), which creates `.venv`, installs `requirements.txt`, copies `config.env` to `.env`, checks Ollama models, and optionally runs tenant ingestion. `setup.sh` is for Linux/macOS bash only; Windows users should follow the PowerShell steps above.

---

## Configuration

Copy `config.env` to `.env`:

- **Windows (PowerShell):**
  ```powershell
  Copy-Item config.env .env
  ```
- **Linux / macOS (bash):**
  ```bash
  cp config.env .env
  ```

Edit `.env` and add your `OPENAI_API_KEY` (or set `LLM_PROVIDER=ollama` to run locally without OpenAI).

### Core `.env` settings

```env
# ── Tenant ────────────────────────────────────
TENANT=chikku                # chikku (default) | saigon

# ── LLM Provider ──────────────────────────────
LLM_PROVIDER=openai          # openai (default) | ollama
OPENAI_API_KEY=sk-...        # required when LLM_PROVIDER=openai
OPENAI_LLM_MODEL=gpt-4o-mini

# ── Ollama (embeddings always use Ollama) ─────
EMBEDDING_MODEL=mxbai-embed-large
LLM_MODEL=llama3:8b-instruct-q4_0   # used when LLM_PROVIDER=ollama
FORCE_CPU=false

# ── Retrieval & Reranking ─────────────────────
TOP_K=7
RERANK_K=12
USE_RERANKING=false          # set true for FlagEmbedding BGE reranker

# ── Multilingual ──────────────────────────────
MULTILINGUAL_ENABLED=true    # false = English only, no translation
DEFAULT_LANGUAGE=en
TRANSLATION_GLOSSARY=Chikku Robotics|S-Robot|WatchGuard6S|Surender Rangaraju

# ── API Server ────────────────────────────────
API_HOST=0.0.0.0
API_PORT=8000
ALLOWED_ORIGINS=*
```

> **Note on `USE_RERANKING=true`:** Enabling reranking loads `BAAI/bge-reranker-v2-m3` via `FlagEmbedding` (~1–2 GB download on first startup) and adds latency. Keep `USE_RERANKING=false` for fast fresh-system setup and turn it on when you need higher ranking precision.

<details>
<summary><strong>Advanced & tuning environment variables (complete reference)</strong></summary>

| Variable | Default | Where used | Description |
|---|---|---|---|
| `CHROMA_DIR` | tenant default | `src/tenant_config.py` | Override vector store path (`chikku_db` or `chroma_db`) |
| `COLLECTION_NAME` | tenant default | `src/tenant_config.py` | Override Chroma collection (`chikku_kb` or `hotel_saigon_menu`) |
| `SYSTEM_PROMPT_FILE` | tenant default | `src/tenant_config.py` | Override prompt file (`data/chikku_system_prompt.txt` or `data/system_prompt.txt`) |
| `QUERY_NORMALIZATION` | `true` | `src/rag.py` | Normalize speech-to-text Indian food terms (`saigon` path) |
| `OLLAMA_NUM_GPU` | unset | `src/rag.py`, `src/llm_provider.py` | Passed to Ollama `num_gpu`; `FORCE_CPU=true` sets `OLLAMA_NUM_GPU=0` |
| `TRANSLATION_CACHE_MAX_SIZE` | `300` | `src/language.py` | LRU cache size for translated strings (min `20`) |
| `RAG_CACHE_ENABLED` | `true` | `src/rag.py` | Cache retrieval hits |
| `RAG_CACHE_MAX_SIZE` | `100` | `src/rag.py` | Retrieval LRU cache size (min `10`) |
| `EMBEDDING_CACHE_ENABLED` | `true` | `src/rag.py` | Cache query embeddings |
| `EMBEDDING_CACHE_MAX_SIZE` | `200` | `src/rag.py` | Embedding LRU cache size (min `50`) |
| `ENABLE_RESPONSE_VALIDATION` | `true` | `src/response_validator.py` | Enforce persona rules and strip AI disclaimers |
| `RATE_LIMIT_ENABLED` | `false` *(code)* / `true` *(`config.env`)* | `src/rate_limit.py` | In-memory IP rate limiter |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | `src/rate_limit.py` | Rate limit sliding window |
| `RATE_LIMIT_MAX_REQUESTS` | `60` | `src/rate_limit.py` | Max requests per client IP per window |
| `LOG_LEVEL` | `INFO` | `src/api.py` | Application log level |
| `USE_JSON_LOGS` | `false` | `src/api.py` | Structured JSON logging output |
| `CONV_SEMANTIC_ENABLED` | `true` | `src/company_conversational.py` | Semantic small-talk detector for `chikku` |
| `CONV_ACCEPT_TAU` | `0.55` | `src/company_conversational.py` | Minimum cosine similarity for conversational match |
| `CONV_HIGH_TAU` | `0.75` | `src/company_conversational.py` | High-confidence conversational threshold |
| `CONV_MARGIN` | `0.06` | `src/company_conversational.py` | Minimum margin over runner-up conversational class |
| `CONV_SHORT_QUERY_WORDS` | `5` | `src/company_conversational.py` | Word-count limit for short-query conversational rule |
| `CONV_MAX_TOKENS` | `100` | `src/company_rag.py` | Max tokens for `chikku` conversational replies |
| `CONV_TEMPERATURE` | `0.7` | `src/company_rag.py` | Temperature for `chikku` conversational replies |
| `API_TIMEOUT` | `60` | `test_api.py`, `simple_test.py` | HTTP request timeout (seconds) for client test scripts only |

</details>

---

## Tenants

One codebase serves two assistants. `TENANT` in `.env` (or in your shell environment) selects which:

| `TENANT` | Assistant | Vector store | Collection | System prompt |
|---|---|---|---|---|
| `chikku` *(default)* | Chikku Robotics support agent | `chikku_db/` | `chikku_kb` | `data/chikku_system_prompt.txt` |
| `saigon` | Saigon Indian Restaurant host | `chroma_db/` | `hotel_saigon_menu` | `data/system_prompt.txt` |

The tenant selects the vector store, system prompt, intent taxonomy, retrieval filters, and response-validator rules (see `src/tenant_config.py`). Switching is a restart — no code change needed:

- **Windows (PowerShell):**
  ```powershell
  $env:TENANT = "saigon"
  python run_api.py
  # Later, clear the shell override so .env takes over again:
  Remove-Item Env:TENANT
  ```
- **Linux / macOS (bash):**
  ```bash
  TENANT=saigon python run_api.py
  ```

Individual fields can be overridden with `CHROMA_DIR`, `COLLECTION_NAME`, and `SYSTEM_PROMPT_FILE`.

**How the two paths differ**

- `chikku` (`domain=company`) runs `src/company_rag.py`: a B2B support taxonomy (`identity`, `company_info`, `product`, `service`, `solution`, `technology`, `partnership`, `support`), semantic small-talk detection (`src/company_conversational.py`), metadata-scoped retrieval by `doc_type`, and a constant `audience='customer'` filter on every lookup. The markdown files in `data/Chikku-conpany-data/Chikku_docs/` are developer docs from the `chikku_mobile` frontend repo and are **not indexed at all**, so internal code material never surfaces in answers.
- `saigon` (`domain=restaurant`) runs the menu pipeline in `src/rag.py`: dietary/protein filtering, dish-name extraction for orders, speech-to-text normalisation tuned to Indian food, and the structured restaurant knowledge base (`data/knowledge_base/`) for hours/location/policies.

The response validator is tenant-aware too. Its default rules ban phrases like `"machine learning"` and `"artificial intelligence"` as persona leaks — correct for a restaurant host, wrong for a robotics company. `validator_allow` in `src/tenant_config.py` exempts those terms for `chikku` while still blocking real persona leaks (`"as an AI language model"`).

---

## Data Ingestion

> **Pre-built vector stores are already committed in the repository** (`chikku_db/` with 330 chunks and `chroma_db/` with 202 menu items). On a fresh clone, you can skip directly to [Running the API](#running-the-api) once `.env` and `ollama pull mxbai-embed-large` are ready. Re-run ingestion only when source files change or you switch embedding models.

> **Note:** Embeddings always use Ollama (`mxbai-embed-large`). Switching `LLM_PROVIDER` between `openai` and `ollama` does **not** require re-ingestion. Make sure `ollama serve` is running (and `OPENAI_API_KEY` is set in `.env` if `LLM_PROVIDER=openai`) before running embedding scripts.

### Chikku Robotics (`TENANT=chikku`)

Rebuild the index and portable export from `data/Chikku-conpany-data/`:

```bash
python src/ingest_chikku.py --reset
```

This indexes all **16 customer-facing source files** in `data/Chikku-conpany-data/` — **8 JSON** and **8 PDF** (via `pdfplumber`) — into `chikku_db/chikku_kb` (**330 chunks**, `audience="customer"`), and writes a portable export to `data/processed/chikku_kb/` (JSONL, CSV, per-doc-type splits, and pre-computed vectors). See [data/processed/chikku_kb/README.md](data/processed/chikku_kb/README.md).

Useful flags for `src/ingest_chikku.py`:
- `--dry-run` — print chunk plan without calling Ollama or modifying ChromaDB
- `--no-pdfs` — index structured JSON files only
- `--export-only` / `--no-embeddings` — regenerate `data/processed/chikku_kb/` exports without writing to ChromaDB

Verify nothing was lost in chunking (exits with code `1` if any fragments are missing):

```bash
python src/audit_chikku_coverage.py            # per-file coverage table
python src/audit_chikku_coverage.py --verbose  # list every missing fragment
```

Upload the pre-computed export into `chikku_db/chikku_kb` **without** calling Ollama:

```bash
python src/load_chikku_export.py --reset
```

*(Pass `--embed` to recompute embeddings from text instead of reusing stored vectors.)*

### Saigon Indian Restaurant (`TENANT=saigon`)

1. *(Optional — only needed if `saigon.pdf` changes)* Re-extract menu items from `saigon.pdf` into `data/processed/menu_items.json`:
   ```bash
   python src/extract_menu.py
   ```
2. Build or rebuild `chroma_db/hotel_saigon_menu`:
   ```bash
   python src/ingestion.py
   ```
   *(Because `chroma_db/` already contains 202 items, `src/ingestion.py` prompts `Re-index? This will replace existing data (y/n):` — enter `y` to wipe and rebuild, or `n` to run validation queries only.)*
3. *(Optional)* Index/update restaurant "about" paragraphs from `data/processed/about.txt` into the existing `hotel_saigon_menu` collection:
   ```bash
   python src/ingest_about.py
   ```

---

## Running the API

Start the development server (honors `API_HOST` and `API_PORT` from `.env`, auto-reload enabled):

```bash
python run_api.py
```

Or run Uvicorn directly from the repository root:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

- API root: `http://localhost:8000`
- Interactive Swagger docs: `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service metadata (`tenant`, `domain`, `version`, endpoint map) |
| `GET` | `/health` | Health check and active Chroma collection count |
| `GET` | `/stats` | Collection stats, active models, and multilingual config |
| `POST` | `/query` | One-shot RAG query (uses `session_id` lookup for `conversation_id`) |
| `POST` | `/chat/text` | Non-streaming chat query (`conversation_id` used directly as internal ID; `QueryResponse` does not return `conversation_id`) |
| `POST` | `/chat/stream` | Streaming response (SSE) |
| `POST` | `/chat/stop` | Acknowledge stop request *(placeholder endpoint)* |
| `POST` | `/conversation/new` | Create a new conversation (`{"conversation_id": "<uuid>", "internal_id": "<id>"}`) |
| `GET` | `/conversation/{conv_id}/history` | Get conversation history |
| `DELETE` | `/conversation/{conv_id}` | Clear conversation history |

`/query`, `/chat/text`, and `/chat/stream` all accept an optional `language` field (`"ta"`, `"ta-IN"`, `"vi"`, …). Omit it and the language is inferred from the script. `/query` and `/chat/text` echo back `language` and, when a translation happened, `query_english`.

`/chat/stream` emits Server-Sent Events (`data: <json>\n\n`) in order:
1. `{"type": "language", "language": "<lang>"}` (before the first token)
2. `{"type": "items", "items": [...], "count": <n>, "language": "<lang>"}`
3. `{"type": "token", "content": "<chunk>"}` (per streamed chunk) and `{"type": "text", "content": "<accumulated>"}` (every 10 tokens + final)
4. `{"type": "done", "conversation_id": "<id>", "full_response": "<text>", "language": "<lang>"}` (or `{"type": "error", "content": "<msg>"}` on failure)

### Examples

- **Windows (PowerShell):**
  ```powershell
  # Health check
  Invoke-RestMethod http://localhost:8000/health

  # Query (use active tenant's domain — e.g., Chikku Robotics by default)
  $body = @{ query = "What products does Chikku Robotics offer?" } | ConvertTo-Json
  Invoke-RestMethod -Uri http://localhost:8000/query -Method Post -ContentType "application/json" -Body $body

  # Streaming SSE (pipe JSON via stdin to avoid PowerShell quoting issues)
  '{"query":"Tell me about Chikku Robotics."}' | curl.exe -N -X POST http://localhost:8000/chat/stream `
    -H "Content-Type: application/json" --data-binary "@-"
  ```

- **Linux / macOS (bash):**
  ```bash
  # Health check
  curl http://localhost:8000/health

  # Query
  curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "What products does Chikku Robotics offer?"}'

  # Streaming SSE
  curl -X POST http://localhost:8000/chat/stream \
    -H "Content-Type: application/json" \
    -d '{"query": "Tell me about Chikku Robotics."}' \
    --no-buffer
  ```

---

## Switching LLM Provider

Embeddings are always handled by Ollama (`mxbai-embed-large`) so ChromaDB indexes stay consistent. Only the **generation** step switches.

### Use OpenAI *(default)*

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_LLM_MODEL=gpt-4o-mini    # or gpt-4o
```

### Use Ollama *(local / offline)*

Pull your chosen model first (e.g., `ollama pull llama3:8b-instruct-q4_0`; lighter alternative `llama3.2`, or reasoning model `deepseek-r1:latest`), then update `.env`:

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3:8b-instruct-q4_0
```

Restart the API after editing `.env`. No re-ingestion is needed when switching providers.

---

## Multilingual

A question can be asked in Tamil (or any supported language) and answered in that language, without a second vector index.

### Why it works this way

Every retrieval component here is English: the ChromaDB index was embedded with an English-trained model (`mxbai-embed-large`), the intent taxonomies are English regex patterns, and the chunk metadata (`doc_type`, `audience`, section titles) is English.

Rather than re-embedding the corpus per language, the language boundary sits at the edges (`src/language.py`):

```
Tamil question
  → translate to English         (LLM, temperature 0, cached)
  → retrieve / classify / rerank (unchanged English pipeline)
  → LLM instructed to answer in Tamil
  → Tamil answer
```

### Supported languages

| code | language | script block |
|---|---|---|
| `en` | English | Latin |
| `hi` | Hindi | Devanagari |
| `ta` | Tamil | Tamil |
| `bn` | Bengali | Bengali |
| `gu` | Gujarati | Gujarati |
| `ml` | Malayalam | Malayalam |
| `kn` | Kannada | Kannada |
| `te` | Telugu | Telugu |
| `pa` | Punjabi | Gurmukhi |
| `vi` | Vietnamese | Latin |

`GET /stats` reports `supported_languages`, `multilingual_enabled`, and `default_language`. Adding a language takes one entry in `LANGUAGES` plus its Unicode block in `_SCRIPT_RANGES` in `src/language.py` — no re-ingestion and no second index.

Two blocks serve more than one language: Devanagari is also Marathi and Nepali, and Bengali is also Assamese (reported as `hi` and `bn`). Because `vi` shares the Latin script with English, `detect_language()` returns `"en"` for Latin text — Vietnamese requests must send `"language": "vi"`, and `DEFAULT_LANGUAGE` is only surfaced in `GET /stats` (Latin-script fallback resolves to `"en"`). Because `localize()` checks `detect_language(text) == target`, non-Indic target languages such as `"vi"` also pass through the cached English-to-target translator check.

### How the language of a turn is decided

1. The `language` field on the request, if provided (e.g., `"ta"`, `"ta-IN"`, `"vi"`).
2. Otherwise the script of the query text (`_SCRIPT_RANGES` with threshold `0.15`), falling back to `"en"`.

### Proper nouns

An Indic-script question carries no Latin spelling, so the translator transliterates by ear (`சிக்கு ரோபோட்டிக்ஸ்` → *"Siku Robotics"*). `TRANSLATION_GLOSSARY` pins the spellings that matter:

```env
TRANSLATION_GLOSSARY=Chikku Robotics|S-Robot|WatchGuard6S|Surender Rangaraju
```

### Cost & validation

- Non-English turns add one translation call at `temperature=0` (400-token budget), cached up to `TRANSLATION_CACHE_MAX_SIZE` entries (`300` by default). Translation fails open if the provider errors.
- `src/response_validator.py` enforces English persona regexes for English replies; non-English replies undergo structural checks (length/truncation) without whitespace flattening.
- Set `MULTILINGUAL_ENABLED=false` in `.env` to disable translation and answer directives completely.

### Multilingual smoke test

- **Windows (PowerShell):**
  ```powershell
  $utf8Bytes = [System.Text.Encoding]::UTF8.GetBytes('{"query":"சிக்கு ரோபோட்டிக்ஸ் நிறுவனத்தின் நிறுவனர் யார்?","language":"ta"}')
  Invoke-RestMethod -Uri http://localhost:8000/chat/text -Method Post -ContentType "application/json; charset=utf-8" -Body $utf8Bytes
  ```
- **Linux / macOS (bash):**
  ```bash
  curl -X POST http://localhost:8000/chat/text \
    -H "Content-Type: application/json" \
    -d '{"query":"சிக்கு ரோபோட்டிக்ஸ் நிறுவனத்தின் நிறுவனர் யார்?","language":"ta"}'
  ```

---

## Testing

### Offline tests *(no Ollama, OpenAI, or running server needed)*

```bash
python test_language.py                # translation boundaries, script detection, validator (stubbed LLM)
python test_company_conversational.py  # Chikku small-talk detector & company intent rules (fake embedder)
python test_knowledge_base.py          # Saigon restaurant JSON knowledge base loader
```

### Retrieval & live API tests *(`TENANT=saigon`)*

```bash
# Requires TENANT=saigon, chroma_db/, and Ollama embeddings (writes data/fine_tuning_analysis.json)
python test_fine_tuning.py

# Requires API server running (python run_api.py) with TENANT=saigon
python test_api.py
python simple_test.py     # interactive (press Enter between queries)
```

> **Note:** `test_phase1.py` is a legacy Phase 1 development script left in the repo (it expects older `intent_classifier` labels and fixed `test_session_123`/`test_session_456` rows in `conversations.db`). Use the suites above for current verification.

---

## Windows Standalone Package

For distributing a self-contained Windows x64 executable (`dist/backend/SaigonBot-Backend.exe`) that bundles Python, FastAPI, a private Ollama 0.34.1 CPU runtime, `mxbai-embed-large`, seeded ChromaDB data, and `.env` OpenAI config, see [docs/backend-windows.md](docs/backend-windows.md).

`build/` and `dist/` are excluded in `.gitignore` — generated executables and build artifacts (>50 MB) must never be committed to Git.

---

## Troubleshooting

**Ollama connection error / Ollama not running**
- Start the daemon in a terminal (`ollama serve`) and verify models are pulled:
  ```bash
  ollama list
  ```
- On Windows in headless/scripted shells, run `ollama serve` directly in a background terminal rather than relying on the desktop GUI auto-start.

**`OPENAI_API_KEY is required when LLM_PROVIDER=openai`**
- Set `OPENAI_API_KEY=sk-...` in `.env` (or switch `LLM_PROVIDER=ollama` in `.env` and pull `llama3:8b-instruct-q4_0`).

**ChromaDB collection not found**
- For `TENANT=chikku`: `python src/ingest_chikku.py --reset` (or `python src/load_chikku_export.py --reset` without Ollama)
- For `TENANT=saigon`: `python src/ingestion.py`

**Port already in use (`8000`)**
- Change `API_PORT=8001` in `.env` and restart `python run_api.py`
- Or terminate the process listening on port `8000`:
  - **Windows (PowerShell):**
    ```powershell
    Get-NetTCPConnection -LocalPort 8000 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
    ```
  - **Linux / macOS (bash):**
    ```bash
    lsof -ti:8000 | xargs kill -9
    ```

**PowerShell blocks `.venv\Scripts\Activate.ps1`**
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Slow responses or GPU/CUDA out of memory**
- Keep `USE_RERANKING=false` (or reduce `TOP_K` / `RERANK_K`)
- Set `FORCE_CPU=true` in `.env` if local Ollama runs out of VRAM
- Use `LLM_PROVIDER=openai` (`gpt-4o-mini`) for fast cloud generation

**Answers in English when asking in another language / poor non-English retrieval**
- Verify `MULTILINGUAL_ENABLED=true` in `.env`
- For Vietnamese (`vi`) or Romanized queries, pass `"language": "vi"` (or the ISO 639-1 tag) explicitly on the request
- Inspect `query_english` in `/query` or `/chat/text` responses and add brand/product terms to `TRANSLATION_GLOSSARY`

---

## Project Structure

```
saigonbot-fork/
├── src/
│   ├── api.py                      # FastAPI server & SSE streaming endpoints
│   ├── tenant_config.py            # Tenant registry (chikku | saigon) & env overrides
│   ├── rag.py                      # Core RAGPipeline (restaurant path + company delegation)
│   ├── company_rag.py              # Chikku Robotics B2B RAG path
│   ├── company_intents.py          # Chikku B2B intent taxonomy & doc_type mapping
│   ├── company_conversational.py   # Regex + semantic small-talk detector for Chikku
│   ├── intent_classifier.py        # Restaurant intent patterns (27 intent types)
│   ├── knowledge_base.py           # Structured restaurant info loader
│   ├── language.py                 # Translation boundaries + answer-language directives
│   ├── llm_provider.py             # OpenAI / Ollama abstraction (Ollama always for embeddings)
│   ├── response_validator.py       # Tenant-aware persona enforcement
│   ├── response_templates.py       # Deterministic restaurant responses
│   ├── chat.py                     # Conversation session manager
│   ├── database.py                 # SQLite conversation storage
│   ├── errors.py                   # Structured error types & user-friendly messages
│   ├── logging_config.py           # Console / JSON logging setup
│   ├── rate_limit.py               # Fixed-window IP rate limiter
│   ├── extract_menu.py             # Extracts saigon.pdf -> data/processed/menu_items.json
│   ├── ingestion.py                # Saigon menu ingestion -> chroma_db/
│   ├── ingest_about.py             # Saigon about.txt ingestion -> chroma_db/
│   ├── ingest_chikku.py            # Chikku JSON+PDF ingestion -> chikku_db/ + exports
│   ├── audit_chikku_coverage.py    # Audits Chikku source fragments against chikku_db/
│   └── load_chikku_export.py       # Loads data/processed/chikku_kb/ export into ChromaDB
├── data/
│   ├── Chikku-conpany-data/        # 8 JSON + 8 PDF company KB files (+ unindexed Chikku_docs/ frontend docs)
│   ├── processed/
│   │   ├── chikku_kb/              # Portable Chikku KB exports + manifest.json + README.md
│   │   ├── menu_items.json         # 202 extracted Saigon menu items
│   │   └── about.txt               # Restaurant background text
│   ├── knowledge_base/             # Saigon FAQ, hours, location, policies, services JSONs
│   ├── calibration/                # conversational_eval.json eval set
│   ├── chikku_system_prompt.txt    # Chikku tenant system prompt
│   └── system_prompt.txt           # Saigon tenant system prompt
├── docs/
│   └── backend-windows.md          # Windows standalone PyInstaller EXE build & run guide
├── scripts/
│   ├── build-backend.py            # Packages Windows standalone EXE into dist/backend/
│   ├── verify-backend.py           # End-to-end smoke verifier for the built EXE
│   └── backend/                    # PyInstaller launcher, bundled Ollama & lockfile
├── documents/                      # Archived Phase 1–4 restaurant dev notes (historical/stale)
├── chikku_db/                      # Pre-built Chikku ChromaDB vector store (330 chunks)
├── chroma_db/                      # Pre-built Saigon ChromaDB vector store (202 items)
├── conversations.db                # SQLite conversation history
├── saigon.pdf                      # Source Saigon restaurant menu PDF
├── config.env                      # Canonical environment configuration template
├── requirements.txt                # Python dependencies
├── run_api.py                      # API server launcher (honors API_HOST & API_PORT)
├── setup.sh                        # Automated setup script for Linux/macOS bash
├── test_language.py                # Offline multilingual unit tests
├── test_company_conversational.py  # Offline Chikku conversational/intent unit tests
├── test_knowledge_base.py          # Offline restaurant knowledge base check
├── test_phase1.py                  # Legacy Phase 1 dev script (superseded by suites above)
├── test_fine_tuning.py             # Saigon retrieval coverage analyzer
├── test_api.py                     # Live HTTP endpoint suite (TENANT=saigon)
├── simple_test.py                  # Interactive HTTP API test (TENANT=saigon)
└── README.md
```
