"""
Coverage audit for the Chikku Robotics knowledge base.

Answers one question: did every piece of text in data/Chikku-conpany-data
actually make it into the vector store?

For each source file it enumerates the atomic units of content —

    *.json  every leaf value in the object tree
    *.pdf   every non-empty extracted line

— then checks each one appears in the indexed text for that same file. Anything
missing is printed, so a silent chunker bug cannot hide behind a chunk count.

Usage:
    python src/audit_chikku_coverage.py
    python src/audit_chikku_coverage.py --verbose      # list every miss
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List

import chromadb

sys.path.insert(0, str(Path(__file__).parent))

COLLECTION_NAME = "chikku_kb"

# Fragments below this length are too generic to match meaningfully
# ("Yes", "N/A", a bare number) and would produce noise either way.
MIN_FRAGMENT_CHARS = 12


def norm(text: str) -> str:
    """Collapse whitespace and case so formatting differences don't count as loss."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def json_leaves(node, path: str = "") -> Iterator[str]:
    if isinstance(node, dict):
        for k, v in node.items():
            yield from json_leaves(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from json_leaves(v, f"{path}[{i}]")
    elif node is not None:
        yield str(node)


def source_fragments(root: Path) -> Dict[str, List[str]]:
    """Map source_file -> list of atomic content fragments."""
    out: Dict[str, List[str]] = {}

    for path in sorted(root.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        out[path.name] = list(json_leaves(data))

    for path in sorted(root.glob("*.pdf")):
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        out[path.name] = [l for l in text.splitlines() if l.strip()]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit KB coverage against source files")
    parser.add_argument("--chroma-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--verbose", action="store_true", help="print every missing fragment")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    chroma_dir = Path(args.chroma_dir) if args.chroma_dir else root / "chikku_db"
    data_dir = Path(args.data_dir) if args.data_dir else root / "data" / "Chikku-conpany-data"

    col = chromadb.PersistentClient(path=str(chroma_dir)).get_collection(COLLECTION_NAME)
    got = col.get(include=["documents", "metadatas"])

    # Concatenate the indexed text per source file.
    indexed: Dict[str, str] = defaultdict(str)
    for doc, meta in zip(got["documents"], got["metadatas"]):
        indexed[meta["source_file"]] += " " + norm(doc)

    fragments = source_fragments(data_dir)

    print(f"{'source_file':40} {'checked':>8} {'missing':>8}  coverage")
    print("-" * 74)

    total_checked = total_missing = 0
    all_misses: Dict[str, List[str]] = {}

    for name in sorted(fragments):
        haystack = indexed.get(name, "")
        misses = []
        checked = 0
        for frag in fragments[name]:
            n = norm(frag)
            if len(n) < MIN_FRAGMENT_CHARS:
                continue
            checked += 1
            if n not in haystack:
                misses.append(frag)

        pct = 100.0 if checked == 0 else 100.0 * (checked - len(misses)) / checked
        flag = "" if not misses else "  <-- gaps"
        print(f"{name:40} {checked:>8} {len(misses):>8}  {pct:6.2f}%{flag}")

        total_checked += checked
        total_missing += len(misses)
        if misses:
            all_misses[name] = misses

    print("-" * 74)
    pct = 100.0 if total_checked == 0 else 100.0 * (total_checked - total_missing) / total_checked
    print(f"{'TOTAL':40} {total_checked:>8} {total_missing:>8}  {pct:6.2f}%")
    print(f"\nCollection: {COLLECTION_NAME} @ {chroma_dir}  ({col.count()} chunks)")

    if all_misses:
        print("\nMissing fragments:")
        for name, misses in all_misses.items():
            shown = misses if args.verbose else misses[:5]
            print(f"\n  {name}  ({len(misses)} missing)")
            for m in shown:
                print(f"    - {m[:160]}")
            if not args.verbose and len(misses) > len(shown):
                print(f"    ... {len(misses) - len(shown)} more (use --verbose)")
        sys.exit(1)
    else:
        print("\n✅ Every source fragment is present in the vector store.")


if __name__ == "__main__":
    main()
