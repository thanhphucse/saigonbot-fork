"""
Ingestion pipeline for the Chikku Robotics knowledge base.

Builds a SEPARATE vector store from the hotel one:
    - persist dir : <project_root>/chikku_db
    - collection  : chikku_kb

Source: data/Chikku-conpany-data/
    *.json              structured company data      (audience=customer)
    *.pdf               designed report renderings   (audience=customer, source_format=pdf)

The developer-facing markdown docs in Chikku_docs/ are deliberately NOT
indexed: they describe the source code and must never surface in customer
answers. (Historically this was behind --include-docs; removed 2026-09.)

Chunking is structure-aware: each JSON file has its own splitter so a chunk is
one semantically complete unit (one product, one FAQ pair, one capability),
never an arbitrary character window.

Chunk ids are deterministic (`doc_type:category:slug`) so re-running upserts
instead of duplicating.

Usage:
    python src/ingest_chikku.py                 # JSON + PDFs
    python src/ingest_chikku.py --reset         # full rebuild
    python src/ingest_chikku.py --no-pdfs       # structured JSON only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import chromadb
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from llm_provider import get_provider  # noqa: E402


COLLECTION_NAME = "chikku_kb"

# mxbai-embed-large has a 512-token context (~1600 chars). Anything longer is
# split into overlapping parts so no content is silently dropped.
# Code/table-heavy markdown tokenizes far worse than prose, so the cap is set
# conservatively rather than at the ~1600-char prose equivalent.
MAX_CHUNK_CHARS = 1100
CHUNK_OVERLAP = 120

_provider = None


def _get_provider():
    global _provider
    if _provider is None:
        _provider = get_provider()
    return _provider


# ---------------------------------------------------------------- helpers

def slug(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return s[:max_len] or "x"


def join_list(values: Iterable[Any]) -> str:
    """Chroma metadata only accepts scalars -> flatten lists to a pipe string."""
    return " | ".join(str(v) for v in values)


def bullets(values: Iterable[Any]) -> str:
    return "\n".join(f"- {v}" for v in values)


class Chunk:
    def __init__(self, cid: str, text: str, meta: Dict[str, Any]):
        self.id = cid
        self.text = text.strip()
        self.meta = meta


def make_chunk(
    doc_type: str,
    category: str,
    title: str,
    text: str,
    source_file: str,
    *,
    category_title: str = "",
    entity: str = "",
    keywords: Iterable[str] = (),
    audience: str = "customer",
    source_format: str = "json",
) -> Chunk:
    meta = {
        "doc_type": doc_type,
        "category": category,
        "category_title": category_title or category.replace("_", " ").title(),
        "title": title,
        "entity": entity,
        "audience": audience,
        "source_file": source_file,
        "source_format": source_format,
        "keywords": join_list(keywords),
        "company": "Chikku Robotics",
        "char_len": len(text.strip()),
    }
    return Chunk(f"{doc_type}:{category}:{slug(title)}", text, meta)


def load(data_dir: Path, name: str) -> Dict[str, Any]:
    with open(data_dir / name, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- splitters

def chunk_identity(data_dir: Path) -> List[Chunk]:
    src = "identity.json"
    d = load(data_dir, src)
    out: List[Chunk] = []

    b = d["brand"]
    out.append(make_chunk(
        "identity", "brand", "Brand identity",
        f"Chikku Robotics brand identity.\nName: {b['name']}\nTagline: {b['tagline']}\n{b['description']}",
        src, entity=b["name"], keywords=["brand", "tagline", "about", "who are you"],
    ))

    for key, person in d["leadership"].items():
        out.append(make_chunk(
            "identity", "leadership", f"{person['title']}: {person['name']}",
            f"{person['name']} — {person['title']} at Chikku Robotics.\n{person['description']}",
            src, category_title="Leadership", entity=person["name"],
            keywords=["leadership", "team", person["title"], person["name"], key],
        ))

    n = d["name_origin"]
    out.append(make_chunk(
        "identity", "name_origin", "Origin of the name Chikku",
        f"Why the company is called Chikku.\nInspiration: {n['inspiration']}\n{n['meaning']}",
        src, keywords=["name", "origin", "chikku fruit", "why chikku"],
    ))

    i = d["bot_introduction"]
    out.append(make_chunk(
        "identity", "bot_introduction", "Bot self-introduction",
        f"{i['greeting']}\n{i['mission']}",
        src, keywords=["greeting", "introduction", "hello", "who are you"],
    ))
    return out


def chunk_company_overview(data_dir: Path) -> List[Chunk]:
    src = "company_overview.json"
    d = load(data_dir, src)
    out: List[Chunk] = []

    c = d["company"]
    out.append(make_chunk(
        "company", "profile", "Company profile",
        (
            f"Company: {c['name']}\nTagline: {c['tagline']}\n"
            f"Headquarters: {c['headquarters']}\n"
            f"Global offices: {join_list(c['global_offices'])}\n"
            f"Mission: {c['mission']}\nVision: {c['vision']}"
        ),
        src, entity=c["name"],
        keywords=["company", "headquarters", "offices", "location", "mission", "vision"],
    ))

    for v in c["values"]:
        out.append(make_chunk(
            "company", "values", f"Core value: {v['value']}",
            f"Chikku Robotics core value — {v['value']}.\n{v['description']}",
            src, category_title="Core Values", entity=v["value"],
            keywords=["values", "culture", v["value"]],
        ))

    n = d["name_origin"]
    bc = n.get("brand_colors", {})
    out.append(make_chunk(
        "company", "name_origin", "Name origin and brand story",
        (
            f"Inspiration: {n['inspiration']}\nPronunciation: {n['pronunciation']}\n"
            f"{n['story']}\n"
            f"Brand colors — primary: {bc.get('primary','')}, accent: {bc.get('accent','')}"
        ),
        src, keywords=["name origin", "brand story", "pronunciation", "brand colors"],
    ))
    return out


def chunk_partnerships(data_dir: Path) -> List[Chunk]:
    src = "partnerships.json"
    d = load(data_dir, src)
    out: List[Chunk] = []
    for cat_key, block in d.items():
        if block.get("title") or block.get("description"):
            out.append(make_chunk(
                "partnership", cat_key, block.get("title", cat_key),
                f"{block.get('title', '')}\n{block.get('description', '')}\n"
                f"Partners: {join_list(p['name'] for p in block.get('partners', []))}",
                src, category_title=block.get("title", cat_key),
                keywords=["partners", "partnerships", "overview"],
            ))
        for p in block.get("partners", []):
            out.append(make_chunk(
                "partnership", cat_key, f"Partner: {p['name']}",
                (
                    f"{p['name']} — {p['type']} partner of Chikku Robotics.\n"
                    f"{p['description']}\nCollaboration: {p['collaboration']}"
                ),
                src, category_title=block.get("title", cat_key), entity=p["name"],
                keywords=["partner", "partnership", p["name"], p["type"]],
            ))
    return out


def chunk_products(data_dir: Path) -> List[Chunk]:
    src = "products.json"
    d = load(data_dir, src)
    out: List[Chunk] = []

    hw = d["hardware"]
    out.append(make_chunk(
        "product", "hardware", hw["title"],
        f"{hw['title']}\n{hw['description']}\nProduct lines:\n{bullets(hw['products'])}",
        src, category_title=hw["title"],
        keywords=["hardware", "robots", "catalog"] + list(hw["products"]),
    ))

    sw = d["software"]
    out.append(make_chunk(
        "product", "software", sw["title"],
        f"{sw['title']}\n{sw['description']}\nPlatforms:\n{bullets(sw['technologies'])}",
        src, category_title=sw["title"],
        keywords=["software", "ai platform"] + list(sw["technologies"]),
    ))

    for p in d["featured_products"]:
        # Featured products do not share one schema — the Kiosk has use_cases,
        # S-Robot has key_benefits, WatchGuard6S has mission. Render whatever
        # keys are present rather than hardcoding a known set, so a new field
        # in the source data is never silently dropped.
        head_keys = ("name", "category", "tagline", "description")
        sections = [
            f"{p['name']} ({p.get('category', '')})",
            p.get("tagline", ""),
            "",
            p.get("description", ""),
        ]
        for key, value in p.items():
            if key in head_keys:
                continue
            label = key.replace("_", " ").title()
            if isinstance(value, dict):
                body = "\n".join(
                    f"{k.replace('_', ' ').title()}:\n"
                    + (bullets(v) if isinstance(v, list) else f"- {v}")
                    for k, v in value.items()
                )
            elif isinstance(value, list):
                body = bullets(value)
            else:
                body = f"- {value}"
            sections.append(f"\n{label}:\n{body}")
        text = "\n".join(sections)
        out.append(make_chunk(
            "product", "featured", p["name"], text, src,
            category_title="Featured Products", entity=p["name"],
            keywords=[p["name"], p["category"]] + list(p["target_environments"]),
        ))
    return out


def _chunk_category_with_items(
    data_dir: Path, src: str, doc_type: str, item_keys: List[str], item_label: str
) -> List[Chunk]:
    """Shared shape: {category: {title, description, <items>: [{name, description, ...}]}}"""
    d = load(data_dir, src)
    out: List[Chunk] = []
    for cat_key, block in d.items():
        items: List[Dict] = []
        for k in item_keys:
            items.extend(block.get(k, []))

        names = [
            (i.get("name") or i.get("area") or "") if isinstance(i, dict) else str(i)
            for i in items
        ]
        overview = f"{block['title']}\n{block['description']}"
        if names:
            overview += f"\n\n{item_label}:\n{bullets(names)}"
        if block.get("case_study"):
            overview += f"\n\nCase study: {block['case_study']}"
        out.append(make_chunk(
            doc_type, cat_key, block["title"], overview, src,
            category_title=block["title"],
            keywords=[block["title"]] + names,
        ))

        for item in items:
            if not isinstance(item, dict):
                # Plain string entries still deserve their own chunk.
                out.append(make_chunk(
                    doc_type, cat_key, str(item),
                    f"{block['title']} — {item}", src,
                    category_title=block["title"], entity=str(item),
                    keywords=[str(item), block["title"]],
                ))
                continue
            name = item.get("name") or item.get("area") or "item"
            extras = []
            for k, v in item.items():
                if k in ("name", "area", "description"):
                    continue
                extras.append(f"{k.replace('_', ' ').title()}: {join_list(v) if isinstance(v, list) else v}")
            text = (
                f"{block['title']} — {name}\n{item.get('description', '')}"
                + ("\n" + "\n".join(extras) if extras else "")
            )
            out.append(make_chunk(
                doc_type, cat_key, name, text, src,
                category_title=block["title"], entity=name,
                keywords=[name, block["title"]],
            ))
    return out


def chunk_services(data_dir: Path) -> List[Chunk]:
    return _chunk_category_with_items(
        data_dir, "services.json", "service", ["offerings", "programs"], "Offerings"
    )


def chunk_solutions(data_dir: Path) -> List[Chunk]:
    return _chunk_category_with_items(
        data_dir, "solutions.json", "solution", ["applications"], "Applications"
    )


def chunk_technologies(data_dir: Path) -> List[Chunk]:
    src = "technologies.json"
    d = load(data_dir, src)
    out: List[Chunk] = []
    for cat_key, block in d.items():
        items: List[Dict] = []
        for k in ("capabilities", "platforms"):
            items.extend(block.get(k, []))

        names = [i.get("name") or i.get("area", "") for i in items if isinstance(i, dict)]
        overview = f"{block['title']}\n{block['description']}"
        if names:
            overview += f"\n\nCapabilities:\n{bullets(names)}"
        for extra_key in ("standards", "safety_features"):
            if block.get(extra_key):
                vals = block[extra_key]
                rendered = [
                    v if isinstance(v, str) else " — ".join(str(x) for x in v.values())
                    for v in vals
                ]
                overview += f"\n\n{extra_key.replace('_', ' ').title()}:\n{bullets(rendered)}"
        out.append(make_chunk(
            "technology", cat_key, block["title"], overview, src,
            category_title=block["title"], keywords=[block["title"]] + names,
        ))

        for item in items:
            name = item.get("name") or item.get("area", "item")
            extras = [
                f"{k.replace('_', ' ').title()}: {join_list(v) if isinstance(v, list) else v}"
                for k, v in item.items()
                if k not in ("name", "area", "description")
            ]
            text = (
                f"{block['title']} — {name}\n{item.get('description', '')}"
                + ("\n" + "\n".join(extras) if extras else "")
            )
            out.append(make_chunk(
                "technology", cat_key, name, text, src,
                category_title=block["title"], entity=name,
                keywords=[name, block["title"]],
            ))
    return out


def chunk_faq(data_dir: Path) -> List[Chunk]:
    src = "faq.json"
    d = load(data_dir, src)
    out: List[Chunk] = []
    if d.get("title") or d.get("description"):
        out.append(make_chunk(
            "faq", "general", d.get("title", "FAQ"),
            f"{d.get('title', '')}\n{d.get('description', '')}",
            src, category_title="Frequently Asked Questions",
            keywords=["faq", "overview"],
        ))
    for item in d["faq_items"]:
        q = item["question"]
        out.append(make_chunk(
            "faq", "general", q,
            f"Q: {q}\nA: {item['answer']}",
            src, category_title="Frequently Asked Questions", entity=q,
            keywords=["faq", "question"],
        ))
    return out


def chunk_pdfs(data_dir: Path) -> List[Chunk]:
    """Index the PDF reports.

    Each PDF is the designed report rendering of its sibling JSON file. The
    facts largely overlap, but the reports also carry material the JSON does
    not — summary tables, partnership tiers, section numbering and ordering —
    so they are indexed in full.

    Split on the reports' numbered headings ("2.1 Product List"); if a PDF has
    no such structure, fall back to one chunk per page. Tagged
    source_format="pdf" so PDF-derived text can be filtered in or out at query
    time, and given doc_type="pdfreport" so it does not crowd out the
    structured JSON chunks in the intent-scoped retrieval path.
    """
    import pdfplumber

    heading_re = re.compile(r'^(\d+(?:\.\d+)*)\s+(\S.*)$')
    out: List[Chunk] = []

    for path in sorted(data_dir.glob("*.pdf")):
        with pdfplumber.open(path) as pdf:
            pages = [(p.extract_text() or "") for p in pdf.pages]
        full = "\n".join(pages).strip()
        if not full:
            print(f"⚠️  {path.name}: no extractable text (scanned image?) — skipped")
            continue

        # Split into numbered sections.
        sections: List[Tuple[str, List[str]]] = []
        current_title = f"{path.stem} — overview"
        current_body: List[str] = []
        for line in full.splitlines():
            m = heading_re.match(line.strip())
            if m and len(m.group(2)) < 90:
                # Flush even when the body is empty: consecutive headings are
                # table rows in these reports and must not be dropped.
                sections.append((current_title, current_body))
                current_title = f"{m.group(1)} {m.group(2)}".strip()
                current_body = []
            else:
                current_body.append(line)
        if current_body:
            sections.append((current_title, current_body))

        # No numbered structure at all -> fall back to page-level chunks.
        if len(sections) <= 1:
            sections = [(f"page {i + 1}", text.splitlines()) for i, text in enumerate(pages) if text.strip()]

        for title, body_lines in sections:
            body = "\n".join(body_lines).strip()
            # Keep short sections too — a heading with a one-line body is still
            # content. The title is always part of the chunk text, so nothing
            # here can produce an empty document.
            out.append(make_chunk(
                "pdfreport", path.stem, f"{path.stem}: {title}",
                f"{path.stem.replace('_', ' ').title()} report — {title}\n{body}",
                path.name,
                category_title=f"{path.stem.replace('_', ' ').title()} (PDF report)",
                entity=title,
                keywords=["pdf", "report", path.stem, title],
                source_format="pdf",
            ))

    return out


# ---------------------------------------------------------------- indexing

def split_on_lines(body: str, window: int) -> List[str]:
    """Split text into <=window pieces, preferring line boundaries.

    A table row or bullet cut in half is present in the store but never
    retrievable as one statement, so parts break between lines wherever
    possible. A single line longer than the window is hard-wrapped with overlap
    — unavoidable, since the embedding model's context is the hard ceiling.

    Consecutive parts repeat the trailing lines of the previous part (up to
    CHUNK_OVERLAP characters) so a fact spanning a boundary still appears whole
    in one of them.
    """
    lines = body.split("\n")
    parts: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        if current:
            parts.append("\n".join(current))
            current, current_len = [], 0

    for line in lines:
        # A single over-long line cannot be kept whole: wrap it with overlap.
        if len(line) > window:
            flush()
            step = max(1, window - CHUNK_OVERLAP)
            for i in range(0, len(line), step):
                piece = line[i:i + window]
                parts.append(piece)
                if i + window >= len(line):
                    break
            continue

        if current_len + len(line) + 1 > window:
            # Carry the tail of this part into the next one as overlap.
            carry: List[str] = []
            carry_len = 0
            for prev in reversed(current):
                if carry_len + len(prev) + 1 > CHUNK_OVERLAP:
                    break
                carry.insert(0, prev)
                carry_len += len(prev) + 1
            flush()
            current = list(carry)
            current_len = carry_len

        current.append(line)
        current_len += len(line) + 1

    flush()

    # A final part fully contained in its predecessor adds nothing.
    if len(parts) > 1 and parts[-1] in parts[-2]:
        parts.pop()
    return parts or [body]


def split_oversized(chunks: List[Chunk]) -> List[Chunk]:
    """Split any chunk longer than the embedding model's context into parts.

    Each part keeps the original metadata plus part_index/part_total, and is
    prefixed with the chunk title so a mid-document part still carries context.
    """
    out: List[Chunk] = []
    for c in chunks:
        if len(c.text) <= MAX_CHUNK_CHARS:
            c.meta["part_index"] = 0
            c.meta["part_total"] = 1
            out.append(c)
            continue

        header = f"{c.meta['category_title']} — {c.meta['title']}\n"
        body = c.text
        window = MAX_CHUNK_CHARS - len(header)
        parts = split_on_lines(body, window)
        for n, part in enumerate(parts):
            meta = dict(c.meta)
            meta["part_index"] = n
            meta["part_total"] = len(parts)
            text = header + part
            meta["char_len"] = len(text)
            out.append(Chunk(f"{c.id}#p{n}", text, meta))
    return out


def build_chunks(data_dir: Path, include_pdfs: bool = True) -> List[Chunk]:
    chunks: List[Chunk] = []
    chunks += chunk_identity(data_dir)
    chunks += chunk_company_overview(data_dir)
    chunks += chunk_partnerships(data_dir)
    chunks += chunk_products(data_dir)
    chunks += chunk_services(data_dir)
    chunks += chunk_solutions(data_dir)
    chunks += chunk_technologies(data_dir)
    chunks += chunk_faq(data_dir)
    if include_pdfs:
        chunks += chunk_pdfs(data_dir)

    chunks = split_oversized(chunks)

    # deterministic ids must be unique; suffix collisions
    seen: Dict[str, int] = {}
    for c in chunks:
        if c.id in seen:
            seen[c.id] += 1
            c.id = f"{c.id}-{seen[c.id]}"
        else:
            seen[c.id] = 0
    return chunks


def embed_all(chunks: List[Chunk]) -> List[List[float]]:
    provider = _get_provider()
    print(f"🔄 Embedding {len(chunks)} chunks via {provider.embedding_model} ...")
    vectors = []
    for c in tqdm(chunks, desc="Embedding", mininterval=5.0):
        try:
            vectors.append(provider.embed(c.text))
        except Exception as e:
            # last-resort guard: shrink and retry rather than abort the run
            print(f"⚠️  {c.id}: {e} — retrying truncated")
            vectors.append(provider.embed(c.text[:1000]))
    return vectors


def export_chunks(chunks: List[Chunk], out_dir: Path, vectors: List[List[float]] | None = None) -> None:
    """Write the prepared chunks + metadata to portable files.

    Produces a vendor-neutral payload so the same corpus can be uploaded to any
    vector store (Chroma, Pinecone, Qdrant, Weaviate, pgvector, ...) without
    re-running the chunkers.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [{"id": c.id, "text": c.text, "metadata": c.meta} for c in chunks]

    # 1. JSONL — the canonical upload format, one record per line
    with open(out_dir / "chikku_kb.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 2. JSON array — easier to eyeball / diff
    with open(out_dir / "chikku_kb.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # 3. CSV — for tools that only accept tabular upload
    meta_keys = sorted({k for r in records for k in r["metadata"]})
    with open(out_dir / "chikku_kb.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text"] + meta_keys)
        for r in records:
            w.writerow([r["id"], r["text"]] + [r["metadata"].get(k, "") for k in meta_keys])

    # 4. Split by audience — customer-facing corpus vs internal technical docs
    for audience in ("customer", "internal"):
        subset = [r for r in records if r["metadata"]["audience"] == audience]
        if not subset:
            continue
        with open(out_dir / f"chikku_kb.{audience}.jsonl", "w", encoding="utf-8") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 5. Split by doc_type — lets you load only products, only FAQ, etc.
    by_type: Dict[str, List[Dict]] = {}
    for r in records:
        by_type.setdefault(r["metadata"]["doc_type"], []).append(r)
    type_dir = out_dir / "by_doc_type"
    type_dir.mkdir(exist_ok=True)
    for doc_type, subset in by_type.items():
        with open(type_dir / f"{doc_type}.jsonl", "w", encoding="utf-8") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 6. Pre-computed embeddings — upload without re-embedding anything
    if vectors:
        with open(out_dir / "chikku_kb.embeddings.jsonl", "w", encoding="utf-8") as f:
            for r, vec in zip(records, vectors):
                f.write(json.dumps({**r, "embedding": vec}, ensure_ascii=False) + "\n")

    # 7. Manifest — schema, counts, and the model the vectors came from
    provider = _get_provider() if vectors else None
    manifest = {
        "collection": COLLECTION_NAME,
        "company": "Chikku Robotics",
        "total_chunks": len(records),
        "counts_by_doc_type": {k: len(v) for k, v in sorted(by_type.items())},
        "counts_by_source_format": {
            fmt: sum(1 for r in records if r["metadata"].get("source_format") == fmt)
            for fmt in ("json", "pdf")
        },
        "counts_by_audience": {
            a: sum(1 for r in records if r["metadata"]["audience"] == a)
            for a in ("customer", "internal")
        },
        "embedding_model": provider.embedding_model if provider else None,
        "embedding_dim": len(vectors[0]) if vectors else None,
        "distance": "cosine",
        "max_chunk_chars": MAX_CHUNK_CHARS,
        "chunk_overlap": CHUNK_OVERLAP,
        "metadata_schema": {
            "doc_type": "identity | company | partnership | product | service | solution | technology | faq",
            "category": "section key within the source file (e.g. healthcare, safety_systems, training)",
            "category_title": "human-readable section title",
            "title": "chunk title",
            "entity": "named product / partner / person / program, empty when not applicable",
            "audience": "customer (public-facing); the corpus no longer contains internal docs",
            "source_file": "originating file inside data/Chikku-conpany-data",
            "keywords": "pipe-separated search aids",
            "company": "constant: Chikku Robotics",
            "char_len": "length of the chunk text",
            "part_index": "0-based index when a long chunk was split",
            "part_total": "number of parts the original chunk was split into",
        },
        "files": {
            "chikku_kb.jsonl": "all chunks, one JSON record per line (canonical)",
            "chikku_kb.json": "same records as a single JSON array",
            "chikku_kb.csv": "flat tabular export, metadata as columns",
            "chikku_kb.customer.jsonl": "public-facing subset",
            "by_doc_type/*.jsonl": "one file per doc_type",
            "chikku_kb.embeddings.jsonl": "records with pre-computed embedding vectors",
        },
        "notes": [
            "Metadata values are scalars only (lists are pipe-joined) for Chroma compatibility.",
            "Chunk ids are deterministic, so re-uploading upserts instead of duplicating.",
            "Retrieval always filters audience='customer'; the corpus contains no internal docs.",
        ],
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n📦 Exported {len(records)} records to {out_dir}")
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            print(f"   {p.relative_to(out_dir)}  ({p.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index Chikku Robotics KB into ChromaDB")
    parser.add_argument("--chroma-dir", default=None, help="defaults to <root>/chikku_db")
    parser.add_argument("--data-dir", default=None, help="defaults to <root>/data/Chikku-conpany-data")
    parser.add_argument("--no-pdfs", action="store_true", help="skip the PDF reports (indexed by default)")
    parser.add_argument("--reset", action="store_true", help="delete the collection before indexing")
    parser.add_argument("--dry-run", action="store_true", help="print chunk plan, no embeddings")
    parser.add_argument("--export-dir", default=None, help="defaults to <root>/data/processed/chikku_kb")
    parser.add_argument("--export-only", action="store_true", help="write export files, skip ChromaDB write")
    parser.add_argument("--no-embeddings", action="store_true", help="export text+metadata only, no vectors")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    chroma_dir = Path(args.chroma_dir) if args.chroma_dir else root / "chikku_db"
    data_dir = Path(args.data_dir) if args.data_dir else root / "data" / "Chikku-conpany-data"

    if not data_dir.exists():
        raise SystemExit(f"data dir not found: {data_dir}")

    chunks = build_chunks(data_dir, not args.no_pdfs)

    counts: Dict[str, int] = {}
    for c in chunks:
        counts[c.meta["doc_type"]] = counts.get(c.meta["doc_type"], 0) + 1
    print("📊 Chunk plan:")
    for k, v in sorted(counts.items()):
        print(f"   {k:<12} {v:>4}")
    print(f"   {'TOTAL':<12} {len(chunks):>4}")

    if args.dry_run:
        for c in chunks[:5]:
            print(f"\n--- {c.id}\n{c.text[:300]}\nmeta={c.meta}")
        return

    export_dir = Path(args.export_dir) if args.export_dir else root / "data" / "processed" / "chikku_kb"

    if args.export_only:
        vectors = None if args.no_embeddings else embed_all(chunks)
        export_chunks(chunks, export_dir, vectors)
        return

    client = chromadb.PersistentClient(path=str(chroma_dir))
    if args.reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"🗑️  Dropped existing collection {COLLECTION_NAME}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Chikku Robotics company knowledge base",
            "hnsw:space": "cosine",
        },
    )

    vectors = embed_all(chunks)

    batch = 100
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        collection.upsert(
            ids=[c.id for c in part],
            documents=[c.text for c in part],
            metadatas=[c.meta for c in part],
            embeddings=vectors[i:i + batch],
        )

    print(f"\n✅ Indexed into {chroma_dir}/{COLLECTION_NAME}")
    print(f"   Total docs in collection: {collection.count()}")

    export_chunks(chunks, export_dir, vectors)


if __name__ == "__main__":
    main()
