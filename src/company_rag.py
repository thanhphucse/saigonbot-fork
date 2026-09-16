"""
Company-support RAG path (Chikku Robotics tenant).

`RAGPipeline` delegates here whenever the active tenant's domain is "company".
It reuses the pipeline's embedding cache, retrieval, reranker, provider and
validator, but replaces everything that is restaurant-shaped: intent taxonomy,
metadata-aware context formatting, prompts, and fallbacks.

Retrieval is metadata-driven. The intent picks a set of `doc_type` values, and
every lookup filters audience='customer' — the corpus contains no internal
engineering docs, and this constant filter keeps it that way even if stray
metadata ever reappears in the store.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from company_intents import (
    CompanyCategory,
    CompanyIntent,
    CompanyIntentResult,
    get_company_intent_classifier,
)
from language import answer_language_directive, localize
from response_validator import get_validator

# Minimum number of hits before we stop narrowing by doc_type and retry broad.
MIN_HITS_BEFORE_BROADENING = 3

# The PDF reports restate most of what the JSON files contain, so a query can
# retrieve the same fact twice in different wording. Above this token overlap
# the lower-ranked copy is dropped, keeping the context window for new facts.
DUPLICATE_OVERLAP_THRESHOLD = 0.72

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def dedupe_near_identical(items: List[Dict]) -> List[Dict]:
    """Drop chunks that mostly repeat a higher-ranked chunk already kept.

    Retrieval order is relevance order, so the first copy of a fact wins and
    later restatements (typically the PDF rendering of the same JSON content)
    are discarded.
    """
    kept: List[Dict] = []
    kept_tokens: List[set] = []

    for item in items:
        toks = _tokens(item.get("document", ""))
        if not toks:
            continue

        duplicate = False
        for prev in kept_tokens:
            overlap = len(toks & prev) / min(len(toks), len(prev))
            if overlap >= DUPLICATE_OVERLAP_THRESHOLD:
                duplicate = True
                break

        if not duplicate:
            kept.append(item)
            kept_tokens.append(toks)

    if len(kept) < len(items):
        print(f"   🧹 Dropped {len(items) - len(kept)} near-duplicate chunk(s)")
    return kept


class CompanyRAG:
    def __init__(self, pipeline):
        self.p = pipeline                       # the owning RAGPipeline
        self.tenant = pipeline.tenant
        self.classifier = get_company_intent_classifier()

    # ------------------------------------------------------------ retrieval

    def _build_where(self, intent: CompanyIntentResult, use_doc_types: bool) -> Optional[Dict]:
        # audience='customer' is unconditional: internal docs are not part of
        # this corpus, and the filter guards against any stray ones returning.
        clauses: List[Dict] = [{"audience": "customer"}]

        if use_doc_types and intent.doc_types:
            clauses.append({"doc_type": {"$in": intent.doc_types}})

        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def retrieve(self, query: str, intent: CompanyIntentResult, top_k: int) -> List[Dict]:
        """Narrow by doc_type first, then broaden if the filter was too tight."""
        items = self.p.retrieve(query, top_k=top_k, where=self._build_where(intent, True))

        if len(items) < MIN_HITS_BEFORE_BROADENING:
            print(f"   ↔️  Only {len(items)} hits with doc_type filter — broadening")
            broad = self.p.retrieve(query, top_k=top_k, where=self._build_where(intent, False))
            seen = {i["id"] for i in items}
            items += [i for i in broad if i["id"] not in seen]

        return dedupe_near_identical(items)[:top_k]

    # ------------------------------------------------------------- context

    def format_context(self, items: List[Dict]) -> str:
        """Render chunks with their provenance so the LLM can cite section names."""
        parts = []
        for item in items:
            m = item.get("metadata", {})
            header = m.get("category_title") or m.get("category", "")
            title = m.get("title", "")
            entity = m.get("entity", "")

            label = " › ".join(x for x in (header, title) if x and x != header) or title or header
            if entity and entity not in label:
                label = f"{label} ({entity})"

            parts.append(f"[{m.get('doc_type', 'info')}] {label}\n{item['document'].strip()}\n")

        return "\n---\n".join(parts)

    # -------------------------------------------------------------- prompts

    def welcome_message(self) -> str:
        return (
            "Hello, I am Chikku! 🤖 I'm the virtual assistant for **Chikku Robotics** — "
            "we build intelligent robots for a smarter future.\n\n"
            "I can help you with our **products**, **industry solutions**, **services**, "
            "**technology**, and **partnerships**. What would you like to know?"
        )

    def _intent_guidance(self, intent: CompanyIntentResult) -> str:
        guidance = {
            CompanyIntent.SALES: (
                "The customer is asking about pricing, purchasing, or a demo. Give whatever "
                "commercial detail the sources contain, and invite them to contact the sales "
                "team for a tailored quote. Never invent prices or discounts."
            ),
            CompanyIntent.SUPPORT: (
                "The customer has a technical or operational problem. Give clear, ordered steps "
                "from the sources. If the sources do not cover their exact case, say so and "
                "point them to the support team rather than guessing."
            ),
            CompanyIntent.PRODUCT: (
                "Name the specific products and their concrete capabilities. Do not attribute "
                "features to a product unless the sources state them."
            ),
            CompanyIntent.SOLUTION: (
                "Explain the industry solution and which robots it uses, and mention the case "
                "study outcome if one appears in the sources."
            ),
            CompanyIntent.TECHNOLOGY: (
                "Be precise and technical. Name the actual platforms, standards, and techniques "
                "in the sources rather than speaking in generalities."
            ),
            CompanyIntent.PARTNERSHIP: (
                "Name the partner, what they provide, and the nature of the collaboration."
            ),
            CompanyIntent.COMPANY_INFO: (
                "Answer factually from the company profile — mission, values, leadership, and "
                "locations as stated in the sources."
            ),
            CompanyIntent.IDENTITY: (
                "Introduce yourself as Chikku, the assistant for Chikku Robotics, and briefly "
                "say what you can help with."
            ),
        }
        return guidance.get(intent.intent_type, "")

    def build_user_prompt(
        self,
        query: str,
        context: str,
        intent: CompanyIntentResult,
        is_first_message: bool,
        language: Optional[str] = None,
    ) -> str:
        greeting_line = (
            "1. Open with a short, warm greeting — only on this first message of the session."
            if is_first_message
            else "1. Do NOT greet again — answer directly."
        )
        guidance = self._intent_guidance(intent)
        guidance_block = f"\nCONTEXT FOR THIS QUESTION:\n{guidance}\n" if guidance else ""

        return f"""Customer question: {query}
{guidance_block}
You are Chikku, the customer support assistant for Chikku Robotics.

How to answer:
{greeting_line}
2. Answer using ONLY the sources below. Do not invent products, prices, clients, or specifications.
3. Be specific — name the actual products, technologies, standards, and partners from the sources.
4. Keep it concise and business-appropriate. Use short paragraphs or bullets.
5. Never write a bullet or heading you cannot fill in — if you only know a product's name, mention it in a sentence instead of an empty list item.
6. If the sources do not answer the question, say so plainly and offer to connect them with the team.
7. Never say you are an AI language model, and never mention "sources", "context", "retrieval", or "documents" — just answer.
8. End with a brief, relevant follow-up question.

SOURCES:
{context}

Answer as Chikku, using only the information above.""" + answer_language_directive(language)

    def no_results_response(self, intent: CompanyIntentResult) -> str:
        if intent.intent_type == CompanyIntent.SALES:
            return (
                "I don't have that commercial detail on hand. 🙂 Our sales team can put together "
                "a tailored quote for your requirement — would you like me to note down what "
                "you're looking for?"
            )
        if intent.intent_type == CompanyIntent.SUPPORT:
            return (
                "I couldn't find that in our support material. Our engineering support team can "
                "help directly — could you tell me which robot or system you're working with?"
            )
        return (
            "I don't have information on that yet. 🙂 I can help with our products, industry "
            "solutions, services, technology, and partnerships — which of those would be useful?"
        )

    # -------------------------------------------------------------- generate

    def generate(
        self,
        query: str,
        context: str,
        intent: CompanyIntentResult,
        conversation_history: Optional[List[Dict]] = None,
        language: Optional[str] = None,
    ) -> str:
        system_prompt = self.p._build_system_prompt(context, conversation_history)
        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            for msg in conversation_history[-6:]:
                if msg.get("role") == "user":
                    messages.append({"role": "user", "content": msg.get("content", "")})

        messages.append({
            "role": "user",
            "content": self.build_user_prompt(
                query, context, intent,
                is_first_message=not conversation_history,
                language=language,
            ),
        })

        try:
            result = self.p.provider.chat(messages, {
                "max_tokens": 500,
                "temperature": 0.3,
                "top_p": 0.9,
                "num_ctx": 4096,
            })
        except Exception as e:
            print(f"❌ LLM error: {e}")
            return (
                "I'm having trouble reaching our systems right now. Please try again in a "
                "moment, or contact our team directly."
            )

        if not result or not result.strip():
            return self.no_results_response(intent)

        validator = get_validator()
        validation = validator.validate(
            result, query=query, context_provided=bool(context), language=language
        )
        if not validation["valid"]:
            print(f"⚠️  Response validation failed (score: {validation['score']:.2f})")
            print(f"   Issues: {validation['issues']}")
            if validator.enabled:
                result = validation["cleaned_response"]

        return result

    def stream(
        self,
        query: str,
        context: str,
        intent: CompanyIntentResult,
        conversation_history: Optional[List[Dict]] = None,
        language: Optional[str] = None,
    ):
        system_prompt = self.p._build_system_prompt(context, conversation_history)
        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            for msg in conversation_history[-4:]:
                if msg.get("role") in ("user", "assistant"):
                    messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({
            "role": "user",
            "content": self.build_user_prompt(
                query, context, intent,
                is_first_message=not conversation_history,
                language=language,
            ),
        })

        try:
            for chunk in self.p.provider.stream_chat(messages, {
                "max_tokens": 500,
                "temperature": 0.3,
                "top_p": 0.9,
                "num_ctx": 4096,
            }):
                yield chunk
        except Exception as e:
            print(f"❌ LLM streaming error: {e}")
            # Nothing downstream localizes a streamed chunk, so do it here. The
            # string is fixed, so the translation is a cache hit after the first.
            yield localize(
                self.p.provider,
                "\n\n[Sorry — I hit a problem generating that answer. Please try again.]",
                language,
            )

    # ---------------------------------------------------------------- query

    def query(
        self,
        user_query: str,
        top_k: int,
        rerank_k: int,
        conversation_history: Optional[List[Dict]] = None,
        language: Optional[str] = None,
    ) -> Dict:
        """`user_query` is English (RAGPipeline.query translated it if needed).

        Canned replies are returned in English; RAGPipeline.query translates them
        on the way out. `language` only travels as far as the LLM prompts, so the
        model answers natively rather than being translated afterwards.
        """
        intent = self.classifier.classify(user_query)
        print(f"📋 Intent: {intent.intent_type.value} ({intent.category.value}, "
              f"confidence {intent.confidence:.2f})")

        # Pure small talk needs no retrieval.
        if intent.category == CompanyCategory.CONVERSATIONAL:
            response = (
                self.welcome_message()
                if not conversation_history
                else "Happy to help! What would you like to know about Chikku Robotics?"
            )
            return {
                "query": user_query,
                "response": response,
                "items": [],
                "retrieved_count": 0,
                "intent": intent.to_dict(),
            }

        print(f"📥 Retrieving top {top_k} (doc_types={intent.doc_types or 'any'})")
        retrieved = self.retrieve(user_query, intent, top_k)
        print(f"✅ Retrieved {len(retrieved)} chunks")

        if not retrieved:
            return {
                "query": user_query,
                "response": self.no_results_response(intent),
                "items": [],
                "retrieved_count": 0,
                "intent": intent.to_dict(),
            }

        if self.p.use_reranking:
            retrieved = self.p.rerank(user_query, retrieved, top_k=rerank_k)
            print(f"🔄 Reranked to {len(retrieved)} chunks")
        else:
            retrieved = retrieved[:rerank_k]

        context = self.format_context(retrieved)
        sanitized = self.p._sanitize_conversation_history(conversation_history) if conversation_history else None
        response = self.generate(user_query, context, intent, sanitized, language=language)

        return {
            "query": user_query,
            "response": response,
            "items": [
                {
                    "name": i["metadata"].get("title"),
                    "section": i["metadata"].get("category_title"),
                    "price": None,
                    "currency": None,
                    "tags": [
                        t.strip()
                        for t in (i["metadata"].get("keywords") or "").split("|")
                        if t.strip()
                    ],
                }
                for i in retrieved
            ],
            "retrieved_count": len(retrieved),
            "intent": intent.to_dict(),
        }
