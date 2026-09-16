"""
Conversational (small-talk) detection for the company-support tenant.

Two complementary layers, both keyed to `CompanyConvKind`:

- Regex fast-path (`fast_detect`): the canonical utterances ("hi", "bye",
  "thank you") are formulaic. Matching them costs nothing, is deterministic,
  and keeps working when the embedding backend is down.
- Semantic detector (`ConversationalDetector`): embeds the query with the
  pipeline's embedding model and compares it by cosine similarity against
  prototype utterances per kind. This is the layer that catches the paraphrases
  no pattern list will ever be exhaustive over ("laterz", "gotta run",
  "that's all for today").

Precedence between these layers and the knowledge patterns lives in
`CompanyIntentClassifier.classify` — this module only answers "is this
chitchat, and of what kind?".

Thresholds are env-overridable so the operating point can be tuned without a
code change; see `data/calibration/conversational_eval.json` for the eval set
used to pick the defaults.
"""

from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from enum import Enum
from typing import Callable, List, NamedTuple, Optional


class CompanyConvKind(Enum):
    GREETING = "greeting"
    THANKS = "thanks"
    FAREWELL = "farewell"
    HOW_ARE_YOU = "how_are_you"
    ACK = "ack"


class ConvHit(NamedTuple):
    kind: CompanyConvKind
    score: float          # cosine similarity to the best-matching prototype
    margin: float         # score minus the runner-up kind's score


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


# Master switch: false falls back to regex-only detection (Rule 4).
SEMANTIC_ENABLED = os.getenv("CONV_SEMANTIC_ENABLED", "true").lower() == "true"
# Minimum similarity to accept a semantic verdict at all.
ACCEPT_TAU = _env_float("CONV_ACCEPT_TAU", 0.55)
# Similarity above which a short message is chitchat even if it contains a
# knowledge keyword ("cool robots!").
HIGH_TAU = _env_float("CONV_HIGH_TAU", 0.75)
# Best kind must beat the runner-up by this much, else the verdict is
# "not sure" and the query falls through to knowledge retrieval.
MARGIN = _env_float("CONV_MARGIN", 0.06)
# Rule-1 word limit.
SHORT_QUERY_WORDS = _env_int("CONV_SHORT_QUERY_WORDS", 5)

# --- Regex fast-path ---------------------------------------------------------
# Ordered: first match wins. HOW_ARE_YOU is deliberately end-anchored so
# "how are you useful?" (a capabilities question) does NOT match here.
class _FastPattern(NamedTuple):
    kind: CompanyConvKind
    regex: str


FAST_PATTERNS = [
    _FastPattern(CompanyConvKind.GREETING,
        r'^\s*(hi|hii+|hey|hello|yo|namaste|good\s+(morning|afternoon|evening))\b'),
    _FastPattern(CompanyConvKind.THANKS,
        r'^\s*(thanks|thank\s+you|thx|ty|cheers)\b'),
    _FastPattern(CompanyConvKind.FAREWELL,
        r'^\s*(bye|goodbye|see\s+you|good\s*night)\b'),
    _FastPattern(CompanyConvKind.ACK,
        r'^\s*(ok|okay|cool|nice|great|awesome|got\s+it)\s*[.!]?\s*$'),
    _FastPattern(CompanyConvKind.HOW_ARE_YOU,
        r"^\s*how\s+are\s+you(\s+(doing|today))?\s*[?.!]*\s*$"),
]

_FAST_COMPILED = [(k, re.compile(rx, re.IGNORECASE)) for k, rx in FAST_PATTERNS]


def fast_detect(text: str) -> Optional[CompanyConvKind]:
    """Regex fast-path: canonical utterances only. Never throws."""
    for kind, rx in _FAST_COMPILED:
        if rx.search(text or ""):
            return kind
    return None


# --- Semantic detector --------------------------------------------------------
# ~8 prototypes per kind. Pure data: add/replace utterances freely — embeddings
# are rebuilt once per process at warmup.
PROTOTYPES = {
    CompanyConvKind.GREETING: [
        "hi", "hello", "hey there", "good morning", "namaste",
        "hiya", "hello, how's it going", "hey!",
    ],
    CompanyConvKind.THANKS: [
        "thank you", "thanks a lot", "thanks so much", "appreciate it",
        "that's very helpful, thanks", "great, thank you", "thx",
        "cheers, that helped",
    ],
    CompanyConvKind.FAREWELL: [
        "bye", "goodbye", "see you later", "good night", "that's all, bye",
        "gotta run, bye", "talk to you later", "later!",
    ],
    CompanyConvKind.HOW_ARE_YOU: [
        "how are you?", "how are you doing today?", "how's it going?",
        "are you doing well?", "how do you feel?", "how are things?",
        "you doing okay?", "how have you been?",
    ],
    CompanyConvKind.ACK: [
        "ok", "okay", "got it", "cool", "nice", "great", "sounds good",
        "understood", "alright", "awesome, thanks",
    ],
}


def _unit(vec: List[float]) -> List[float]:
    norm = sum(v * v for v in vec) ** 0.5
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class ConversationalDetector:
    """Zero-shot chitchat classifier: query embedding vs prototype embeddings.

    `embed_fn` is injected (usually `RAGPipeline.get_embedding`) so detection
    shares the pipeline's embedding cache and this class stays testable with a
    fake embedder. All failures degrade to `detect() -> None`, i.e. "not
    chitchat as far as we can tell" — never an exception in the query path.
    """

    def __init__(self, embed_fn: Callable[[str], List[float]], cache_size: int = 256):
        self._embed_fn = embed_fn
        self._lock = threading.Lock()
        self._prototypes: Optional[List[tuple]] = None   # [(kind, unit_vec)]
        self._cache: "OrderedDict[str, Optional[ConvHit]]" = OrderedDict()
        self._cache_size = cache_size

    # -- lifecycle ---------------------------------------------------------

    def warmup_async(self) -> None:
        def _run():
            try:
                self.warmup()
            except Exception as e:
                # Background best-effort only: detect() retries lazily on
                # demand, so a failed warm-up never breaks the query path.
                print(f"⚠️  Conversational prototype warm-up failed: {e}")

        threading.Thread(target=_run, daemon=True,
                         name="conv-prototype-warmup").start()

    def warmup(self) -> None:
        """Embed the prototype utterances once. Idempotent, thread-safe."""
        with self._lock:
            if self._prototypes is None:
                self._prototypes = [
                    (kind, _unit(self._embed_fn(text)))
                    for kind, texts in PROTOTYPES.items()
                    for text in texts
                ]

    # -- inference -----------------------------------------------------------

    def detect(self, query: str) -> Optional[ConvHit]:
        """Return the best chitchat kind for `query`, or None.

        None means "no confident verdict": detector disabled, backend error,
        or the score/margin below threshold. Callers then fall through to the
        knowledge path.
        """
        if not SEMANTIC_ENABLED or not (query or "").strip():
            return None

        key = query.strip().lower()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        hit = self._compute(key)

        with self._lock:
            self._cache[key] = hit
            self._cache.move_to_end(key)
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return hit

    def _compute(self, key: str) -> Optional[ConvHit]:
        try:
            self._ensure_prototypes()
            qvec = _unit(self._embed_fn(key))
        except Exception as e:  # backend down / embed error → not chitchat
            print(f"⚠️  Conversational detector unavailable: {e}")
            return None

        scores: dict = {}
        for kind, pvec in self._prototypes:
            s = _dot(qvec, pvec)
            if s > scores.get(kind, -1.0):
                scores[kind] = s
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_kind, best = ranked[0]
        margin = best - ranked[1][1] if len(ranked) > 1 else best
        if best >= ACCEPT_TAU and margin >= MARGIN:
            return ConvHit(best_kind, best, margin)
        return None

    def _ensure_prototypes(self) -> None:
        if self._prototypes is None:
            self.warmup()
