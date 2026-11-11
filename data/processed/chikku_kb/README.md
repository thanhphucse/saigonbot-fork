# Chikku Robotics — Knowledge Base Export

Vendor-neutral RAG payload generated from `data/Chikku-conpany-data/`.
Every record is a ready-to-upload chunk: stable id, embedding text, and full metadata.

**All 16 source files are indexed** — 8 JSON + 8 PDF — into 330 chunks, all
`audience=customer`. Coverage is verified: `python src/audit_chikku_coverage.py`
checks every JSON leaf value and every non-empty PDF line against the store.

The developer-facing markdown docs in `data/Chikku-conpany-data/Chikku_docs/`
are **deliberately not indexed**: they describe the source code and must never
surface in customer answers.

Regenerate with:

```bash
python src/ingest_chikku.py --reset          # index + export
python src/ingest_chikku.py --export-only    # export only
python src/ingest_chikku.py --export-only --no-embeddings
```

## Contents

| File | What it is |
|---|---|
| `chikku_kb.jsonl` | **Canonical.** All 330 chunks, one JSON record per line |
| `chikku_kb.json` | Same records as a single JSON array |
| `chikku_kb.csv` | Flat table, metadata as columns — for tools that only take CSV |
| `chikku_kb.embeddings.jsonl` | Records **plus** 1024-d vectors — upload without an embedding model |
| `chikku_kb.customer.jsonl` | The whole corpus (all chunks are customer-facing) |
| `by_doc_type/*.jsonl` | One file per `doc_type` — load only FAQ, only products, etc. |
| `manifest.json` | Counts, schema, embedding model, chunking parameters |

## Record shape

```json
{
  "id": "solution:agriculture:agriculture-agritech",
  "text": "Agriculture & Agritech\nPrecision agriculture powered by robotics and AI ...",
  "metadata": {
    "doc_type": "solution",
    "category": "agriculture",
    "category_title": "Agriculture & Agritech",
    "title": "Agriculture & Agritech",
    "entity": "",
    "audience": "customer",
    "source_file": "solutions.json",
    "keywords": "Agriculture & Agritech | Crop Monitoring | Autonomous Weeding & Spraying",
    "company": "Chikku Robotics",
    "char_len": 219,
    "part_index": 0,
    "part_total": 1
  }
}
```

## Metadata fields

| Field | Meaning |
|---|---|
| `doc_type` | `identity`, `company`, `partnership`, `product`, `service`, `solution`, `technology`, `faq`, `pdfreport` |
| `category` | Section key inside the source file (`healthcare`, `safety_systems`, `training`, …) |
| `category_title` | Human-readable section title |
| `title` | Chunk title |
| `entity` | Named product / partner / person / program; empty when not applicable |
| `audience` | Constant `customer` (no internal docs in this corpus) |
| `source_file` | Originating file inside `data/Chikku-conpany-data` |
| `keywords` | Pipe-separated search aids |
| `company` | Constant `Chikku Robotics` |
| `source_format` | `json` \| `pdf` — how the chunk was extracted |
| `char_len` | Length of the chunk text |
| `part_index` / `part_total` | Set when a long chunk was split across parts |

All metadata values are scalars — lists are pipe-joined, because Chroma rejects
list-valued metadata. Split on `" | "` if your store supports arrays.

## Chunk counts

| doc_type | count | from |
|---|---|---|
| pdfreport | 197 | `*.pdf` |
| faq | 33 | `faq.json` |
| service | 26 | `services.json` |
| solution | 25 | `solutions.json` |
| technology | 25 | `technologies.json` |
| product | 8 | `products.json` |
| company | 7 | `company_overview.json` |
| identity | 5 | `identity.json` |
| partnership | 4 | `partnerships.json` |
| **total** | **330** | 16 files |

By source format: json 133 · pdf 197
By audience: customer 330

## Uploading

### ChromaDB (helper script included)

```bash
# uses the shipped vectors — no Ollama required
python src/load_chikku_export.py

# customer-facing subset only
python src/load_chikku_export.py \
  --input data/processed/chikku_kb/chikku_kb.customer.jsonl --embed

# re-embed from text instead of reusing stored vectors
python src/load_chikku_export.py --embed
```

### Any other vector store

```python
import json

records = [json.loads(l) for l in open("chikku_kb.embeddings.jsonl")]

ids   = [r["id"] for r in records]
texts = [r["text"] for r in records]
metas = [r["metadata"] for r in records]
vecs  = [r["embedding"] for r in records]
```

Vectors are `mxbai-embed-large`, 1024-d, intended for **cosine** distance. Reuse
them only if your query side embeds with the same model — otherwise drop the
`embedding` field and re-embed `text` with your own model.

## Notes

- Chunk ids are deterministic, so re-uploading **upserts** rather than duplicating.
- Retrieval always scopes to `audience == "customer"` (every chunk in this corpus has `audience="customer"`, and `Chikku_docs/` internal technical docs are not indexed).
- Chunks are capped at 1100 chars with 120-char overlap — sized for the
  512-token context of `mxbai-embed-large`. Raise `MAX_CHUNK_CHARS` in
  `src/ingest_chikku.py` if you move to a longer-context embedding model.
- Oversized sections split on line boundaries, so a table row or bullet is never
  cut in half. Only a single line longer than the window is hard-wrapped.
- The PDFs restate most of the JSON content but also carry summary tables, ROI
  sections and brand details the JSON lacks, so they are indexed in full. Answer-time
  near-duplicate suppression in `src/company_rag.py` stops a fact appearing twice in
  one context window.
