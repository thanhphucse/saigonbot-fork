# SaigonBot — Chatbot API

A RAG-based conversational AI chatbot for Saigon Indian Restaurant. Uses ChromaDB for vector search, Ollama for local embeddings, and supports **Ollama or OpenAI** for response generation — switchable via a single `.env` variable. Questions can be asked in **ten languages** — English, Hindi, Tamil, Bengali, Gujarati, Malayalam, Kannada, Telugu, Punjabi, Vietnamese — against the same English index — see [Multilingual](#multilingual).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Data Ingestion](#data-ingestion)
- [Running the API](#running-the-api)
- [API Endpoints](#api-endpoints)
- [Switching LLM Provider](#switching-llm-provider)
- [Multilingual](#multilingual)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

---

## Prerequisites

1. **Python 3.10+**
   ```bash
   python3 --version
   ```

2. **pip** (comes with Python — no extra install needed)

3. **Ollama** (required for embeddings regardless of LLM provider)
   ```bash
   # Install
   curl -fsSL https://ollama.ai/install.sh | sh

   # Pull embedding model (always required)
   ollama pull mxbai-embed-large

   # Pull LLM model (only if using LLM_PROVIDER=ollama)
   ollama pull deepseek-r1:latest
   ```

4. **OpenAI API Key** — only needed if using `LLM_PROVIDER=openai`

---

## Installation

```bash
# 1. Navigate to project
cd saigonbot

# 2. (Optional) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Copy the template and edit:
```bash
cp config.env .env
```

Key settings in `.env`:

```env
# ── Tenant ────────────────────────────────────
TENANT=saigon                # saigon | chikku

# ── LLM Provider ──────────────────────────────
LLM_PROVIDER=ollama          # ollama | openai

# ── Ollama (embeddings always use Ollama) ─────
EMBEDDING_MODEL=mxbai-embed-large
LLM_MODEL=deepseek-r1:latest

# ── OpenAI (only when LLM_PROVIDER=openai) ────
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-4o-mini

# ── Multilingual ──────────────────────────────
MULTILINGUAL_ENABLED=true    # false = English only, no translation
DEFAULT_LANGUAGE=en
# Proper nouns that must survive translation with their exact spelling
TRANSLATION_GLOSSARY=Chikku Robotics|S-Robot|WatchGuard6S

# ── Retrieval ─────────────────────────────────
TOP_K=7
RERANK_K=12
USE_RERANKING=true

# ── API Server ────────────────────────────────
API_HOST=0.0.0.0
API_PORT=8000
API_TIMEOUT=60
ALLOWED_ORIGINS=*
```

---

## Tenants

One codebase serves two assistants. `TENANT` in `.env` selects which:

| TENANT | Assistant | Vector store | Collection | System prompt |
|---|---|---|---|---|
| `saigon` | Saigon Indian Restaurant host | `chroma_db/` | `hotel_saigon_menu` | `data/system_prompt.txt` |
| `chikku` | Chikku Robotics support agent | `chikku_db/` | `chikku_kb` | `data/chikku_system_prompt.txt` |

The tenant decides the vector store, system prompt, intent taxonomy, retrieval
filters and response-validator rules — see `src/tenant_config.py`. Switching is a
restart, not a code change:

```bash
TENANT=chikku python run_api.py
```

Individual fields can be overridden with `CHROMA_DIR`, `COLLECTION_NAME` and
`SYSTEM_PROMPT_FILE` if a tenant needs to point somewhere else.

**How the two paths differ**

- `saigon` (`domain=restaurant`) runs the menu pipeline: dietary/protein filtering,
  dish-name extraction for orders, speech-to-text normalisation tuned to Indian food,
  and the restaurant knowledge base for hours/location/policies.
- `chikku` (`domain=company`) runs `src/company_rag.py`: a B2B support taxonomy
  (product, solution, service, technology, partnership, sales, support, company info),
  metadata-scoped retrieval by `doc_type`, and a constant `audience='customer'`
  filter on every lookup. The developer-facing docs in `Chikku_docs/` are not
  indexed at all, so source-code material can never surface in answers.

The response validator is tenant-aware too. Its default rules ban phrases like
"machine learning" and "artificial intelligence" as persona leaks — correct for a
restaurant host, wrong for a robotics company. `validator_allow` in the tenant config
exempts those terms for `chikku` while still blocking real persona leaks
("as an AI language model").

---

## Data Ingestion

### Restaurant (TENANT=saigon)

Run once (or whenever the menu changes) to build the ChromaDB vector index:

```bash
python src/ingestion.py
```

> Embeddings always use Ollama (`mxbai-embed-large`). Switching to OpenAI for generation does **not** require re-ingestion.

To also index restaurant "about" content:
```bash
python src/ingest_about.py
```

### Chikku Robotics (TENANT=chikku)

```bash
python src/ingest_chikku.py --include-docs --reset
```

Indexes **all 24 files** in `data/Chikku-conpany-data/` — 8 JSON, 8 PDF (via
`pdfplumber`), and 8 internal markdown docs — into `chikku_db/chikku_kb` (559 chunks),
and writes a portable export to `data/processed/chikku_kb/` (JSONL, CSV, per-doc-type
splits, and records with pre-computed vectors). See `data/processed/chikku_kb/README.md`.

Verify nothing was lost in chunking:

```bash
python src/audit_chikku_coverage.py            # per-file coverage table
python src/audit_chikku_coverage.py --verbose  # list every missing fragment
```

It checks every JSON leaf value and every non-empty PDF/markdown line against the
indexed text and fails loudly on gaps — a chunk count alone cannot catch a chunker
bug that silently drops a section.

To upload an existing export without re-embedding:

```bash
python src/load_chikku_export.py
```

---

## Running the API

```bash
python run_api.py
```

Or directly with uvicorn:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

API is available at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Collection stats + active models |
| `POST` | `/query` | One-shot menu query |
| `POST` | `/chat/text` | Conversational query (with history) |
| `POST` | `/chat/stream` | Streaming response (SSE) |
| `POST` | `/conversation/new` | Create a new conversation session |
| `GET` | `/conversation/{id}/history` | Get conversation history |
| `DELETE` | `/conversation/{id}` | Clear conversation |

`/query`, `/chat/text` and `/chat/stream` all accept an optional `language` field
(`"ta"`, `"ta-IN"`, …). Omit it and the language is inferred from the script.
`/chat/text` and `/query` echo back `language` and, when a translation happened,
`query_english`. `/chat/stream` emits a `language` event before the first token.

### Examples

```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "vegetarian options"}'

# Streaming
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "what breakfast items do you have"}' \
  --no-buffer
```

---

## Switching LLM Provider

Embeddings are always handled by Ollama (keeps the ChromaDB index stable).
Only the **generation** step switches.

### Use OpenAI
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_LLM_MODEL=gpt-4o-mini    # or gpt-4o, gpt-3.5-turbo
```

### Use Ollama (local)
```env
LLM_PROVIDER=ollama
LLM_MODEL=deepseek-r1:latest    # any model you've pulled
```

Restart the API after changing `.env`. No re-ingestion needed when switching providers.

---

## Multilingual

A question can be asked in Tamil and is answered in Tamil, without a second
vector index.

### Why it works this way

Every retrieval component here is English: the ChromaDB index was embedded with
an English-trained model (`mxbai-embed-large`), the intent taxonomies are English
regex patterns, and the chunk metadata (`doc_type`, `audience`, section titles) is
English. A Tamil query embedded directly scores near zero against that index, so
retrieval returns noise no matter how good the LLM is.

Rather than re-embedding the corpus per language, the language boundary sits at
the edges (`src/language.py`):

```
Tamil question
  → translate to English         (LLM, temperature 0, cached)
  → retrieve / classify / rerank (unchanged English pipeline)
  → LLM instructed to answer in Tamil
  → Tamil answer
```

Adding a language costs one entry in `LANGUAGES` — no re-ingestion, no second
index, no second intent taxonomy.

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

`GET /stats` reports the live list. Adding one is a single entry in `LANGUAGES`
plus its Unicode block in `_SCRIPT_RANGES` — no re-ingestion, no second index.

Two blocks serve more than one language: Devanagari is also Marathi and Nepali,
Bengali is also Assamese. Those are reported as the majority language, which is
the most a script test can determine. `vi` shares the Latin block with English and
so is never script-detected — a Vietnamese turn has to declare `language`.

### How the language of a turn is decided

1. The `language` field on the request, if the client sent one. The kiosk sends
   what Whisper detected during speech recognition.
2. Otherwise the script of the query text. The Indic languages each own a Unicode
   block, so this is exact rather than probabilistic — it is what covers typed
   queries, which carry no speech signal.

### Two output paths

LLM-generated answers are produced in the target language directly, via a
directive appended to the user prompt (`answer_language_directive`). Only one LLM
call, and the model writes natively rather than translating its own English.

Fixed strings — the welcome message, `"I couldn't find that"`, the ~30 hardcoded
templates in the restaurant path — never reach the LLM, so they are translated on
the way out by `localize()` and cached. `localize()` is idempotent: it detects
that an LLM answer is already Tamil and passes it through untouched, which is why
it can sit on the single exit boundary in `RAGPipeline.query` and cover every
branch without editing any of them.

### Proper nouns

A Tamil-script question carries no Latin spelling, so the translator
transliterates by ear — `சிக்கு ரோபோட்டிக்ஸ்` came back as *"Siku Robotics"*,
which then has to match *"Chikku Robotics"* in the index. `TRANSLATION_GLOSSARY`
pins the spellings that matter:

```env
TRANSLATION_GLOSSARY=Chikku Robotics|S-Robot|WatchGuard6S|Surender Rangaraju
```

For a company assistant the brand name is often the whole query, so this is not
cosmetic.

### Cost

One extra LLM call per non-English turn (the query translation), at
`temperature=0` with a 400-token budget. Translations and canned strings are
cached (`TRANSLATION_CACHE_MAX_SIZE`), so a repeated question costs nothing.
English turns are byte-for-byte the same pipeline as before — no translation call
and no directive.

Translation fails open: if the provider is unreachable the untranslated query
goes to retrieval, which degrades answer quality instead of taking the kiosk down.

### Response validation

`src/response_validator.py` enforces persona with English regexes ("as an AI
model", "training data"). Those cannot match Tamil text, so a non-English reply
is checked for structure only (length, truncation) and returned unmodified — a
clean bill of health from rules that cannot fire would be meaningless, and the
cleaner's whitespace collapse would flatten Markdown it never needed to touch.

### Testing

```bash
python test_language.py    # no Ollama/OpenAI/ChromaDB needed — stubbed provider
```

```bash
# Tamil in, Tamil out
curl -X POST http://localhost:8000/chat/text \
  -H "Content-Type: application/json" \
  -d '{"query":"சிக்கு ரோபோட்டிக்ஸ் நிறுவனத்தின் நிறுவனர் யார்?","language":"ta"}'
```

### Turning it off

```env
MULTILINGUAL_ENABLED=false
```

Every turn is then treated as English: no translation, no directive, no script
detection.

---

## Troubleshooting

**Ollama connection error**
```bash
ollama serve        # start Ollama
ollama list         # verify models are pulled
```

**ChromaDB collection not found**
```bash
python src/ingestion.py
```

**Port already in use**
```bash
# Change port in .env
API_PORT=8001
# Or kill the process
lsof -ti:8000 | xargs kill -9
```

**Answers come back in English even though the question was Tamil**
- Check `MULTILINGUAL_ENABLED=true` in `.env`
- Check the client is sending `language`, or that the query really is in Tamil
  script (transliterated Tamil written in Latin letters detects as English)
- `GET /stats` shows `multilingual_enabled` and `supported_languages`

**Tamil question returns irrelevant chunks**
- Look at `query_english` in the response — if a brand name was mistransliterated,
  add it to `TRANSLATION_GLOSSARY`
- Check the server log for `🌐 Tamil → English:` to see what retrieval actually saw

**Slow responses**
- Reduce `TOP_K` / `RERANK_K`
- Set `USE_RERANKING=false`
- Switch to `LLM_PROVIDER=openai` for faster cloud inference

**OpenAI key error**
- Ensure `OPENAI_API_KEY` is set in `.env`
- Verify key is valid at platform.openai.com

---

## Project Structure

```
saigonbot/
├── src/
│   ├── api.py                  # FastAPI server
│   ├── rag.py                  # RAG pipeline
│   ├── llm_provider.py         # Ollama / OpenAI abstraction layer
│   ├── language.py             # Translation boundaries + answer-language directive
│   ├── ingestion.py            # Menu data ingestion
│   ├── ingest_about.py         # Restaurant info ingestion
│   ├── intent_classifier.py    # 45+ intent patterns
│   ├── response_validator.py   # Persona enforcement
│   ├── response_templates.py   # Deterministic responses
│   ├── knowledge_base.py       # Structured restaurant info
│   ├── chat.py                 # Conversation session manager
│   ├── database.py             # SQLite conversation storage
│   └── extract_menu.py         # PDF menu extraction
├── data/
│   ├── processed/
│   │   └── menu_items.json     # 202 menu items
│   ├── knowledge_base/         # Restaurant info JSON files
│   └── saigon.pdf              # Source menu PDF
├── chroma_db/                  # Vector database (generated)
├── conversations.db            # Conversation history (generated)
├── requirements.txt            # pip dependencies
├── config.env                  # Configuration template
├── .env                        # Active configuration (git-ignored)
├── run_api.py                  # API startup script
└── README.md
```

---

**Version:** 2.1.0 | **Updated:** 2026-08-19
