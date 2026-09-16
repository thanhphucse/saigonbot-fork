#!/usr/bin/env python3
"""
Offline unit tests for the conversational layer (company tenant).

Covers, without any model backend:
  - fast_detect regex behaviour (incl. the anchored HOW_ARE_YOU)
  - ConversationalDetector scoring/thresholds/margin via a fake embedder
  - CompanyIntentClassifier precedence rules 1-5
  - the chitchat prompt builder and canned fallback mapping (company_rag)

Run:  python test_company_conversational.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from company_conversational import (  # noqa: E402
    ACCEPT_TAU, HIGH_TAU, MARGIN,
    CompanyConvKind, ConversationalDetector, PROTOTYPES,
    fast_detect, _unit,
)
from company_intents import (  # noqa: E402
    CompanyCategory, CompanyIntent, CompanyIntentClassifier, CompanyIntentResult,
)
from company_rag import CompanyRAG, CONVERSATIONAL_FALLBACKS  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}")


# ---------------------------------------------------------------- fake embedder
_ALL = [t for texts in PROTOTYPES.values() for t in texts]
_IDX = {t: i for i, t in enumerate(_ALL)}
# One reserved orthogonal dimension: mixing in "*noise*" lowers the cosine to
# every prototype without favouring any kind — that is how tests synthesise a
# "moderate" semantic score.
_NOISE = len(_ALL)
_DIM = len(_ALL) + 1
_ZERO = [0.0] * _DIM


def _mix(*pairs):
    """Vector as a weighted mix of prototype texts plus optional noise:
    _mix(("bye", 0.6), ("*noise*", 0.6))."""
    v = list(_ZERO)
    for text, weight in pairs:
        v[_NOISE if text == "*noise*" else _IDX[text]] = weight
    return v


def make_embedder(mapping):
    """Fake embedder: every prototype text resolves to its own basis vector;
    extra entries override/add query vectors; anything else embeds to zero."""
    table = {t.lower(): _mix((t, 1.0)) for t in _ALL}
    table.update({k.strip().lower(): v for k, v in mapping.items()})

    def embed(text: str):
        return table.get(text.strip().lower(), list(_ZERO))

    return embed


def make_classifier(mapping=None):
    return CompanyIntentClassifier(embed_fn=make_embedder(mapping or {}))


def detector_with(mapping):
    return ConversationalDetector(make_embedder(mapping))


# ------------------------------------------------------------------ fast_detect
print("\nfast_detect (regex fast-path)")
check("'hi' -> GREETING", fast_detect("hi") == CompanyConvKind.GREETING)
check("'good morning!' -> GREETING", fast_detect("good morning!") == CompanyConvKind.GREETING)
check("'thank you bye' -> THANKS (anchored thanks)", fast_detect("thank you bye") == CompanyConvKind.THANKS)
check("'bye!' -> FAREWELL", fast_detect("bye!") == CompanyConvKind.FAREWELL)
check("'ok' -> ACK", fast_detect("ok") == CompanyConvKind.ACK)
check("'how are you?' -> HOW_ARE_YOU", fast_detect("how are you?") == CompanyConvKind.HOW_ARE_YOU)
check("'how are you useful?' -> None (anchored)", fast_detect("how are you useful?") is None)
check("'what is the price?' -> None", fast_detect("what is the price?") is None)

# ---------------------------------------------------------------- semantic layer
print("\nConversationalDetector (fake embedder)")
hit = detector_with({"bye": _mix(("bye", 1.0))}).detect("bye")
check("exact prototype 'bye' -> FAREWELL ~1.0", hit is not None and hit.kind == CompanyConvKind.FAREWELL and hit.score > 0.99)
check("zero-vector query -> None (below threshold)", detector_with({}).detect("completely unmapped text") is None)
hit = detector_with({"laterz my friend": _mix(("bye", 0.6))}).detect("laterz my friend")
check("paraphrase 0.6 >= ACCEPT with margin -> FAREWELL", hit is not None and hit.kind == CompanyConvKind.FAREWELL)
hit = detector_with({"torn message": _mix(("bye", 0.55), ("thank you", 0.52))}).detect("torn message")
check("margin 0.04 < 0.06 -> None (ambiguous)", hit is None)
hit = detector_with({"price laterz": _mix(("bye", 0.6))}).detect("price laterz")
check("0.6 >= ACCEPT even though borderline", hit is not None)

# ------------------------------------------------------------------ classify
print("\nCompanyIntentClassifier precedence")
c = make_classifier({"cool robots": _mix(("cool", 0.9))})
r = c.classify("cool robots")
check("Rule 1: short + semantic 0.9 beats PRODUCT keyword", r.is_conversational() and r.conv_kind == CompanyConvKind.ACK)

c = make_classifier({"price bye": _mix(("bye", 0.55), ("*noise*", 0.6))})
r = c.classify("price bye")
check("Rule 2: knowledge keyword beats moderate semantic", r.intent_type == CompanyIntent.SALES and not r.is_conversational())

c = make_classifier({"how are you useful?": _mix(("how are you?", 0.5), ("*noise*", 0.5))})
r = c.classify("how are you useful?")
check("regression: 'how are you useful?' -> IDENTITY (capabilities)", r.intent_type == CompanyIntent.IDENTITY)

c = make_classifier({"what is your problem?": _mix(("bye", 0.55), ("*noise*", 0.6))})
r = c.classify("what is your problem?")
check("Rule 2: 'what is your problem?' -> SUPPORT", r.intent_type == CompanyIntent.SUPPORT)

c = make_classifier({"laterz my friend": _mix(("bye", 0.6), ("*noise*", 0.6))})
r = c.classify("laterz my friend")
check("Rule 3: no knowledge hit + semantic in accept band -> FAREWELL", r.is_conversational() and r.conv_kind == CompanyConvKind.FAREWELL)

c = make_classifier({"torn message": _mix(("bye", 0.55), ("thank you", 0.52))})
r = c.classify("torn message")
check("Rule 3 rejected by margin -> Rule 5 UNKNOWN -> KB", r.intent_type == CompanyIntent.UNKNOWN and r.category == CompanyCategory.KNOWLEDGE)

c = CompanyIntentClassifier(embed_fn=None)  # detector unavailable
r = c.classify("bye!")
check("Rule 4: regex fast-path without detector", r.is_conversational() and r.conv_kind == CompanyConvKind.FAREWELL)

r = c.classify("how are you?")
check("Rule 4: 'how are you?' -> HOW_ARE_YOU", r.is_conversational() and r.conv_kind == CompanyConvKind.HOW_ARE_YOU)

r = c.classify("quantum flute melodies")
check("Rule 5: open-ended phrasing -> UNKNOWN -> KB", r.intent_type == CompanyIntent.UNKNOWN and r.category == CompanyCategory.KNOWLEDGE)

r = c.classify("")
check("empty query -> CLARIFICATION", r.category == CompanyCategory.CLARIFICATION)

check("to_dict exposes conv_kind",
      "conv_kind" in CompanyIntentResult(
          CompanyIntent.CONVERSATIONAL, 0.6, CompanyCategory.CONVERSATIONAL,
          conv_kind=CompanyConvKind.FAREWELL).to_dict())

# ------------------------------------------------------------------ company_rag
print("\nChitchat prompt builder + fallbacks (company_rag)")
rag = CompanyRAG.__new__(CompanyRAG)  # no pipeline needed for these methods
farewell_intent = CompanyIntentResult(
    CompanyIntent.CONVERSATIONAL, 0.8, CompanyCategory.CONVERSATIONAL,
    conv_kind=CompanyConvKind.FAREWELL)

prompt = rag.build_conversational_prompt(
    "bye!", farewell_intent,
    [{"role": "user", "content": "what robots do you have?"},
     {"role": "assistant", "content": "We offer the Voice AI Kiosk..."}],
    language="vi")
check("prompt names the detected tone", "farewell" in prompt)
check("prompt blocks re-introduction mid-session", "do NOT reintroduce" in prompt)
check("prompt carries recent conversation", "Recent conversation:" in prompt)
check("prompt carries the language directive", "Vietnamese" in prompt or "vi" in prompt.lower())

first_prompt = rag.build_conversational_prompt("hello", farewell_intent, None, None)
check("first-message prompt asks for a self-introduction", "self-introduction" in first_prompt)

fb = rag._fallback_reply(farewell_intent)
check("farewell fallback is a goodbye", "Goodbye" in fb)
check("fallback covers all five kinds", len(CONVERSATIONAL_FALLBACKS) == 5)

# ------------------------------------------------------------------------- wrap
print(f"\n{'=' * 50}")
print(f"Total: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
