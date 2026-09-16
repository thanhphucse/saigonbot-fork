"""
Intent classification for the company-support tenant (Chikku Robotics).

Mirrors the interface of `intent_classifier.IntentResult` (intent_type,
confidence, category, entities, requires_clarification, to_dict) so the RAG
pipeline and API response models work unchanged, but the taxonomy is a B2B
support one rather than a restaurant one.

Each intent also carries the `doc_type` values it should retrieve from, which
maps directly onto the metadata written by `ingest_chikku.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from company_conversational import (
    ACCEPT_TAU,
    ConvHit,
    ConversationalDetector,
    CompanyConvKind,
    HIGH_TAU,
    MARGIN,
    SHORT_QUERY_WORDS,
    fast_detect,
)


class CompanyIntent(Enum):
    IDENTITY = "identity"                # who are you / what is Chikku
    COMPANY_INFO = "company_info"        # mission, vision, values, offices, leadership
    PRODUCT = "product"                  # robots, kiosk, hardware, software platforms
    SERVICE = "service"                  # consulting, training, integration, maintenance
    SOLUTION = "solution"                # industry solutions
    TECHNOLOGY = "technology"            # AI, vision, edge, cloud, safety/certification
    PARTNERSHIP = "partnership"          # NVIDIA, AWS, Qualcomm
    SUPPORT = "support"                  # troubleshooting, errors, how-to
    SALES = "sales"                      # pricing, quote, demo, buy, contact sales
    CONVERSATIONAL = "conversational"    # greetings, thanks, small talk
    UNKNOWN = "unknown"


class CompanyCategory(Enum):
    KNOWLEDGE = "knowledge"          # answer from the vector store
    CONVERSATIONAL = "conversational"
    CLARIFICATION = "clarification"


# doc_type values (from ingest_chikku.py metadata) each intent should search.
# "pdfreport" is included everywhere the PDF reports could hold the answer:
# they restate the JSON content but also carry summary tables the JSON lacks.
# Near-duplicate suppression in company_rag.py stops the restatements from
# crowding out distinct facts.
INTENT_DOC_TYPES: Dict[CompanyIntent, List[str]] = {
    CompanyIntent.IDENTITY: ["identity", "faq", "pdfreport"],
    CompanyIntent.COMPANY_INFO: ["company", "identity", "faq", "pdfreport"],
    CompanyIntent.PRODUCT: ["product", "faq", "solution", "pdfreport"],
    CompanyIntent.SERVICE: ["service", "faq", "pdfreport"],
    CompanyIntent.SOLUTION: ["solution", "product", "faq", "pdfreport"],
    CompanyIntent.TECHNOLOGY: ["technology", "product", "faq", "pdfreport"],
    CompanyIntent.PARTNERSHIP: ["partnership", "faq", "pdfreport"],
    CompanyIntent.SUPPORT: ["service", "faq", "pdfreport"],
    CompanyIntent.SALES: ["faq", "service", "product", "pdfreport"],
    CompanyIntent.CONVERSATIONAL: [],
    CompanyIntent.UNKNOWN: [],
}

@dataclass
class CompanyIntentResult:
    intent_type: CompanyIntent
    confidence: float
    category: CompanyCategory
    entities: Optional[Dict] = None
    requires_clarification: bool = False
    # Set only for CONVERSATIONAL results: which flavour of small talk this is
    # (drives the LLM chitchat prompt and the canned fallbacks).
    conv_kind: Optional[CompanyConvKind] = None

    def __post_init__(self):
        self.entities = self.entities or {}

    @property
    def doc_types(self) -> List[str]:
        return INTENT_DOC_TYPES.get(self.intent_type, [])

    def is_conversational(self) -> bool:
        return self.category == CompanyCategory.CONVERSATIONAL

    def to_dict(self) -> Dict:
        return {
            "intent_type": self.intent_type.value,
            "confidence": self.confidence,
            "category": self.category.value,
            "entities": self.entities,
            "requires_clarification": self.requires_clarification,
            "doc_types": self.doc_types,
            "conv_kind": self.conv_kind.value if self.conv_kind else None,
        }


# Patterns are ordered by specificity: the first list to match wins, so
# narrower intents (sales, support, partnership) are checked before the broad
# knowledge intents.
# Small talk is NOT here: conversational detection lives in
# company_conversational.py (regex fast-path + semantic detector) and runs in
# CompanyIntentClassifier.classify with its own precedence rules.
_PATTERNS: List[Tuple[CompanyIntent, CompanyCategory, List[str]]] = [
    (CompanyIntent.IDENTITY, CompanyCategory.KNOWLEDGE, [
        r'\bwho\s+are\s+you\b',
        r'\bwhat\s+are\s+you\b',
        r'\bwhat\s+is\s+chikku\b',
        r'\bwho\s+is\s+chikku\b',
        r'\bintroduce\s+your\s*self\b',
        r'\bwhy\s+.*\bname\b',
        r'\bname\s+(origin|meaning)\b',
        r'\bwhat\s+can\s+you\s+do\b',
        r'\bhow\s+are\s+you\s+(useful|helpful|different|better)\b',
    ]),
    (CompanyIntent.SALES, CompanyCategory.KNOWLEDGE, [
        r'\b(price|pricing|cost|quote|quotation|how\s+much)\b',
        r'\b(buy|purchase|order|procure)\b',
        r'\b(demo|trial|pilot|poc|proof\s+of\s+concept)\b',
        r'\b(contact|reach|talk\s+to)\s+(sales|you|your\s+team)\b',
        r'\b(rfp|rfq|tender|invoice|payment\s+terms)\b',
        r'\blead\s+time\b',
    ]),
    (CompanyIntent.SUPPORT, CompanyCategory.KNOWLEDGE, [
        r'\b(not\s+working|doesn\'?t\s+work|stopped\s+working|broken|failing|failed)\b',
        r'\b(error|bug|issue|problem|crash|crashed|troubleshoot)\b',
        r'\b(fix|repair|debug|diagnose)\b',
        r'\b(how\s+do\s+i|how\s+to)\s+(configure|install|set\s*up|deploy|update|calibrate)\b',
        r'\b(warranty|rma|spare\s+part|downtime|maintenance\s+schedule)\b',
        r'\b(offline|wake\s*word|latency|logs?)\b',
    ]),
    (CompanyIntent.PARTNERSHIP, CompanyCategory.KNOWLEDGE, [
        r'\b(partner|partnership|partners|collabor\w+|alliance|reseller|distributor)\b',
        r'\b(nvidia|jetson|aws|amazon\s+web\s+services|qualcomm|snapdragon)\b',
    ]),
    (CompanyIntent.PRODUCT, CompanyCategory.KNOWLEDGE, [
        r'\b(product|products|robot|robots|catalog|catalogue|line\s*up|model)\b',
        r'\b(kiosk|s-?robot|watchguard|cobot|drone|amr|humanoid)\b',
        r'\b(hardware|sensor|embedded|autonomous\s+vehicle)\b',
        r'\bwhat\s+do\s+you\s+(make|build|sell|manufacture)\b',
        r'\b(spec|specs|specification|feature|features)\b',
    ]),
    (CompanyIntent.SERVICE, CompanyCategory.KNOWLEDGE, [
        r'\b(service|services|consulting|consultancy|advisory)\b',
        r'\b(training|workshop|course|upskill|certification\s+program)\b',
        r'\b(integration|integrate|deployment|onboarding|commissioning)\b',
        r'\b(maintenance|amc|support\s+plan|sla|lifecycle)\b',
        r'\b(custom|bespoke)\s+(robot|development|build)\b',
    ]),
    (CompanyIntent.SOLUTION, CompanyCategory.KNOWLEDGE, [
        r'\b(solution|solutions|use\s*case|application|deploy\w*\s+in)\b',
        r'\b(manufactur\w+|factory|assembly|production\s+line)\b',
        r'\b(healthcare|hospital|medical|clinic|patient)\b',
        r'\b(warehouse|logistics|fulfilment|fulfillment|inventory|picking)\b',
        r'\b(smart\s+cit\w+|public\s+infrastructure|inspection|surveillance)\b',
        r'\b(agricultur\w+|agritech|farm|crop|harvest)\b',
        r'\b(retail|restaurant|hotel|mall|supermarket)\b',
    ]),
    (CompanyIntent.TECHNOLOGY, CompanyCategory.KNOWLEDGE, [
        r'\b(technology|technologies|tech\s+stack|architecture)\b',
        r'\b(ai|artificial\s+intelligence|machine\s+learning|ml|deep\s+learning)\b',
        r'\b(computer\s+vision|perception|lidar|slam|navigation)\b',
        r'\b(edge|jetson|on-?device|inference)\b',
        r'\b(cloud|fleet\s+management|telemetry|ota)\b',
        r'\b(safety|certif\w+|iso\s*\d+|standard|compliance)\b',
        r'\b(rag|knowledge\s+graph|vector\s+search|llm)\b',
    ]),
    (CompanyIntent.COMPANY_INFO, CompanyCategory.KNOWLEDGE, [
        r'\b(about\s+(the\s+)?company|about\s+chikku|company\s+(profile|overview))\b',
        r'\b(mission|vision|values|culture|philosophy)\b',
        r'\b(founder|ceo|chairman|director|leadership|team|management)\b',
        r'\b(headquarter\w*|hq|office|offices|located|location|where\s+are\s+you)\b',
        r'\b(history|founded|established|when\s+did\s+you\s+start)\b',
        r'\b(career|careers|job|hiring|vacanc\w+)\b',
    ]),
]


class CompanyIntentClassifier:
    """Classifies a query into a company intent.

    Conversational detection is layered ahead of / around the knowledge
    patterns (see company_conversational.py):

      1. Semantic verdict with a high score on a short message wins outright,
         even if it contains a knowledge keyword ("cool robots!").
      2. An explicit knowledge pattern beats a moderate semantic score — this
         is what keeps "how are you useful?" a capabilities question.
      3. Semantic verdict with a decent score AND a clear margin over the
         runner-up kind ("laterz", "gotta run").
      4. Regex fast-path (canonical utterances; also the fallback when the
         embedding backend is unavailable).
      5. Nothing matched → UNKNOWN, and the KB is searched anyway.
    """

    def __init__(self, embed_fn: Optional[Callable[[str], List[float]]] = None):
        self._compiled = [
            (intent, category, [re.compile(p, re.IGNORECASE) for p in patterns])
            for intent, category, patterns in _PATTERNS
        ]
        self.detector = ConversationalDetector(embed_fn) if embed_fn else None

    def classify(self, query: str) -> CompanyIntentResult:
        text = (query or "").strip()
        if not text:
            return CompanyIntentResult(
                CompanyIntent.UNKNOWN, 0.0, CompanyCategory.CLARIFICATION,
                requires_clarification=True,
            )

        hit = self.detector.detect(text) if self.detector else None

        # Rule 1 — unambiguous chitchat in a short message wins outright.
        if hit and hit.score >= HIGH_TAU and len(text.split()) <= SHORT_QUERY_WORDS:
            return self._conversational_result(hit)

        # Rule 2 — explicit knowledge patterns.
        scores: Dict[CompanyIntent, Tuple[int, CompanyCategory]] = {}
        for intent, category, patterns in self._compiled:
            hits = sum(1 for p in patterns if p.search(text))
            if hits:
                scores[intent] = (hits, category)
        if scores:
            # Highest hit count wins; ties break toward the earlier (more
            # specific) pattern group, which is the order _PATTERNS is declared in.
            order = {intent: i for i, (intent, _, _) in enumerate(self._compiled)}
            best = min(scores.items(), key=lambda kv: (-kv[1][0], order[kv[0]]))
            intent, (hits, category) = best
            confidence = min(0.95, 0.55 + 0.15 * hits)
            return CompanyIntentResult(intent, confidence, category,
                                       {"matched_patterns": hits})

        # Rule 3 — semantic chitchat with a clear margin.
        if hit and hit.score >= ACCEPT_TAU and hit.margin >= MARGIN:
            return self._conversational_result(hit)

        # Rule 4 — regex fast-path (also covers the detector being unavailable).
        kind = fast_detect(text)
        if kind:
            return self._conversational_result(ConvHit(kind, 0.0, 0.0))

        # Rule 5 — no pattern matched — still search the KB, the embedding
        # model is better at open-ended phrasing than these rules are.
        return CompanyIntentResult(
            CompanyIntent.UNKNOWN, 0.3, CompanyCategory.KNOWLEDGE,
            requires_clarification=False,
        )

    @staticmethod
    def _conversational_result(hit: ConvHit) -> CompanyIntentResult:
        confidence = min(0.95, max(0.6, hit.score)) if hit.score else 0.6
        return CompanyIntentResult(
            CompanyIntent.CONVERSATIONAL, confidence, CompanyCategory.CONVERSATIONAL,
            conv_kind=hit.kind,
        )


_classifier: Optional[CompanyIntentClassifier] = None


def get_company_intent_classifier(
    embed_fn: Optional[Callable[[str], List[float]]] = None,
) -> CompanyIntentClassifier:
    """Singleton accessor. The FIRST caller decides whether the semantic
    detector exists by passing (or omitting) `embed_fn`."""
    global _classifier
    if _classifier is None:
        _classifier = CompanyIntentClassifier(embed_fn=embed_fn)
    return _classifier
