"""
RAG Pipeline: Query processing with retrieval and generation
"""
import chromadb
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys
import os
import re
from collections import OrderedDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Force CPU usage for Ollama if OLLAMA_NUM_GPU is set in .env
# This prevents CUDA out of memory errors on servers
if os.getenv("OLLAMA_NUM_GPU") is not None:
    os.environ['OLLAMA_NUM_GPU'] = os.getenv("OLLAMA_NUM_GPU")
elif os.getenv("FORCE_CPU", "false").lower() == "true":
    # Alternative: Use FORCE_CPU=true in .env to disable GPU
    os.environ['OLLAMA_NUM_GPU'] = '0'

# Import intent classifier
from intent_classifier import get_intent_classifier, IntentCategory, IntentType
# Import response templates
from response_templates import get_template, get_welcome_message, get_error_response
# Import error handling
from errors import (
    RetrievalError, LLMError, IntentError, DatabaseError,
    ValidationError, SystemError, handle_error
)
# Import response validator
from response_validator import get_validator
# Import knowledge base
from knowledge_base import get_knowledge_base
# Import LLM provider abstraction
from llm_provider import get_provider
# Import tenant configuration (selects vector store, prompt and behaviour)
from tenant_config import get_tenant
# Multilingual edge: the index and intent patterns are English, so a non-English
# question is translated in and the answer is generated back out (see language.py)
from language import (
    answer_language_directive,
    localize,
    resolve_language,
    translate_to_english,
)


class RAGPipeline:
    def __init__(
        self,
        chroma_db_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        use_reranking: Optional[bool] = None,
        top_k: Optional[int] = None,
        rerank_k: Optional[int] = None
    ):
        """Initialize RAG Pipeline"""
        # Active tenant decides the vector store, system prompt and behaviour.
        # Explicit arguments still win, so callers/tests can override.
        self.tenant = get_tenant()
        chroma_db_path = chroma_db_path or str(self.tenant.resolve_chroma_dir())
        collection_name = collection_name or self.tenant.collection_name

        # Initialize LLM provider (Ollama or OpenAI based on LLM_PROVIDER env var)
        self.provider = get_provider()
        # Model names are owned by the provider; keep as attributes for logging/errors
        self.embedding_model = self.provider.embedding_model
        self.llm_model = self.provider.llm_model
        self.use_reranking = use_reranking if use_reranking is not None else os.getenv("USE_RERANKING", "false").lower() == "true"
        self.query_normalization = os.getenv("QUERY_NORMALIZATION", "true").lower() == "true"
        
        # Ensure top_k and rerank_k are valid integers (minimum 1)
        # Reduced defaults for faster processing
        env_top_k = os.getenv("TOP_K", "7")
        env_rerank_k = os.getenv("RERANK_K", "4")
        try:
            self.top_k = top_k if top_k is not None else max(1, int(env_top_k))
        except (ValueError, TypeError):
            self.top_k = 10
        
        try:
            self.rerank_k = rerank_k if rerank_k is not None else max(1, int(env_rerank_k))
        except (ValueError, TypeError):
            self.rerank_k = 5
        
        # Setup ChromaDB
        client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = client.get_collection(name=collection_name)
        
        # Reranking model (optional)
        if self.use_reranking:
            try:
                from FlagEmbedding import FlagReranker
                self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
                print("✅ Reranking enabled")
            except ImportError as e:
                error_msg = str(e)
                if "numpy" in error_msg.lower() or "_multiarray" in error_msg.lower():
                    print(f"⚠️  Reranking not available: NumPy version incompatibility")
                    print(f"   FlagEmbedding requires NumPy < 2.0, but NumPy 2.x is installed")
                    print(f"   Fix with: pip install \"numpy<2.0\"")
                else:
                    print(f"⚠️  Reranking not available: FlagEmbedding not installed")
                    print(f"   Install with: pip install FlagEmbedding")
                print(f"   Error: {e}")
                self.use_reranking = False
                self.reranker = None
            except Exception as e:
                error_msg = str(e)
                if "numpy" in error_msg.lower():
                    print(f"⚠️  Reranking initialization failed: NumPy version incompatibility")
                    print(f"   Fix with: pip install \"numpy<2.0\"")
                else:
                    print(f"⚠️  Reranking initialization failed: {e}")
                self.use_reranking = False
                self.reranker = None
        else:
            self.reranker = None
            print("ℹ️  Reranking disabled (set USE_RERANKING=true in .env to enable)")
        
        print(f"✅ RAG Pipeline initialized")
        print(f"   Tenant: {self.tenant.name} ({self.tenant.display_name}, domain={self.tenant.domain})")
        print(f"   Collection: {collection_name} @ {chroma_db_path}")
        print(f"   Embedding model: {self.embedding_model}")
        print(f"   LLM model: {self.llm_model}")
        print(f"   Top K: {self.top_k}")
        print(f"   Rerank K: {self.rerank_k}")
        print(f"   Reranking: {'enabled' if self.use_reranking else 'disabled'}")

        # Load system prompt from file (configurable via SYSTEM_PROMPT_FILE in .env)
        self.base_system_prompt: str = ""
        try:
            # Tenant supplies the default; SYSTEM_PROMPT_FILE (applied inside
            # get_tenant) can still override it.
            prompt_path = self.tenant.resolve_prompt_path()
            if prompt_path.exists():
                self.base_system_prompt = prompt_path.read_text(encoding="utf-8").strip()
                print(f"✅ Loaded system prompt from {prompt_path.name}")
            else:
                print(f"⚠️  System prompt file not found: {prompt_path}")
        except Exception as e:
            print(f"⚠️  Failed to load system prompt file: {e}")

        # Load static restaurant information from about.txt for use in system prompt.
        # Restaurant-only: the company tenant carries this knowledge in its vector store.
        self.restaurant_info: str = ""
        if self.tenant.is_restaurant:
            try:
                about_path = Path(__file__).parent.parent / "data" / "processed" / "about.txt"
                if about_path.exists():
                    text = about_path.read_text(encoding="utf-8").strip()
                    if text:
                        self.restaurant_info = text
                        print("✅ Loaded restaurant information from about.txt")
                    else:
                        print("⚠️  about.txt is empty, restaurant info section will be omitted")
                else:
                    print("⚠️  about.txt not found, restaurant info section will be omitted")
            except Exception as e:
                print(f"⚠️  Failed to load about.txt: {e}")
                self.restaurant_info = ""

        # Restaurant knowledge base (hours, location, policies) — restaurant tenant only.
        self.knowledge_base = None
        if self.tenant.is_restaurant:
            self.knowledge_base = get_knowledge_base()
            print("✅ Knowledge base initialized")

        # Company-support handler — used when the tenant's domain is "company".
        self.company = None
        if self.tenant.is_company:
            from company_rag import CompanyRAG

            self.company = CompanyRAG(self)
            print("✅ Company support handler initialized")

        # Simple in-memory retrieval cache (for menu queries)
        self.enable_cache = os.getenv("RAG_CACHE_ENABLED", "true").lower() == "true"
        self.cache_max_size = max(10, int(os.getenv("RAG_CACHE_MAX_SIZE", "100")))
        # key: (query, top_k, doc_type, where_repr) -> list[Dict]
        self._retrieval_cache: "OrderedDict[Tuple[str, int, Optional[str], str], List[Dict]]" = OrderedDict()
        
        # Embedding cache (for faster repeated queries)
        self.enable_embedding_cache = os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
        self.embedding_cache_max_size = max(50, int(os.getenv("EMBEDDING_CACHE_MAX_SIZE", "200")))
        # key: query_text -> List[float]
        self._embedding_cache: "OrderedDict[str, List[float]]" = OrderedDict()

        # Pipeline is fully built here — safe to start background warm-up of
        # the conversational detector's prototype embeddings (company tenant).
        if self.company and self.company.classifier.detector:
            self.company.classifier.detector.warmup_async()

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text (with caching)"""
        # Normalize text for cache key
        cache_key = text.strip().lower()
        
        # Check embedding cache first
        if self.enable_embedding_cache and cache_key in self._embedding_cache:
            # Move to end (LRU)
            embedding = self._embedding_cache.pop(cache_key)
            self._embedding_cache[cache_key] = embedding
            return embedding
        
        # Generate embedding
        try:
            embedding = self.provider.embed(text)
            
            # Cache embedding
            if self.enable_embedding_cache:
                if len(self._embedding_cache) >= self.embedding_cache_max_size:
                    # Remove oldest (first) item
                    self._embedding_cache.popitem(last=False)
                self._embedding_cache[cache_key] = embedding
            
            return embedding
        except Exception as e:
            raise RetrievalError(f"Failed to generate embedding: {str(e)}", query=text)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        doc_type: Optional[str] = None,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """Retrieve relevant items from ChromaDB.

        `doc_type` is the restaurant tenant's post-filter (menu vs about).
        `where` is a ChromaDB metadata filter applied server-side, used by the
        company tenant to scope by doc_type/audience.
        """
        cache_key = (query, top_k, doc_type, repr(where))

        # Check cache first
        if self.enable_cache and cache_key in self._retrieval_cache:
            print(f"🧠 Using cached retrieval results for query: '{query}'")
            # Move to end to mark as recently used
            items = self._retrieval_cache.pop(cache_key)
            self._retrieval_cache[cache_key] = items
            return items[:top_k]

        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Always query without filter first (for backward compatibility)
        # Then filter manually if doc_type is specified
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k * 2 if doc_type else top_k  # Get more if we need to filter
        }
        if where:
            query_params["where"] = where

        # Search in ChromaDB (without doc_type filter for backward compatibility)
        results = self.collection.query(**query_params)
        
        # Format results
        retrieved_items = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                item = {
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i]
                }
                retrieved_items.append(item)
        
        # Filter by doc_type if requested
        # Menu items are identified by item_name presence (ingestion.py never sets doc_type on them).
        # About docs have doc_type="about" set explicitly by ingest_about.py.
        # Never use has_doc_type branch — it drops all menu items when any about doc appears in results.
        if doc_type and retrieved_items:
            if doc_type == "menu":
                retrieved_items = [item for item in retrieved_items
                                   if item.get('metadata', {}).get('item_name')]
            elif doc_type == "about":
                retrieved_items = [item for item in retrieved_items
                                   if item.get('metadata', {}).get('doc_type') == 'about']

            # Trim to requested top_k after filtering
            retrieved_items = retrieved_items[:top_k]
        
        # Cache the results
        if self.enable_cache:
            self._retrieval_cache[cache_key] = retrieved_items
            # Evict least recently used if cache is too big
            if len(self._retrieval_cache) > self.cache_max_size:
                self._retrieval_cache.popitem(last=False)
        
        return retrieved_items
    
    def rerank(self, query: str, items: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank retrieved items using cross-encoder"""
        # Ensure top_k is at least 1
        top_k = max(1, int(top_k))
        
        if not self.use_reranking or not self.reranker:
            return items[:top_k]
        
        # Prepare pairs for reranking
        pairs = []
        for item in items:
            pairs.append([query, item['document']])
        
        # Get reranking scores
        scores = self.reranker.compute_score(pairs)
        
        # Handle single score vs list
        if not isinstance(scores, list):
            scores = [scores]
        
        # Add scores and sort
        for i, item in enumerate(items):
            item['rerank_score'] = scores[i] if i < len(scores) else 0.0
        
        # Sort by rerank score (descending)
        items.sort(key=lambda x: x.get('rerank_score', 0.0), reverse=True)
        
        return items[:top_k]
    
    def format_context(self, items: List[Dict]) -> str:
        """Format retrieved items as context for LLM"""
        context_parts = []
        
        for item in items:
            metadata = item['metadata']
            item_name = metadata.get('item_name', 'Unknown')
            section = metadata.get('section', 'Unknown')
            price = metadata.get('price')
            currency = metadata.get('currency', 'VND')
            tags = metadata.get('tags', [])
            
            # Format price correctly (X,XXX VND)
            price_str = "N/A"
            if price is not None:
                try:
                    # Convert to int/float and format with commas
                    price_num = float(price) if isinstance(price, str) else price
                    price_str = f"{price_num:,.0f} {currency}".replace(',', ',')
                except (ValueError, TypeError):
                    price_str = f"{price} {currency}" if price else "N/A"
            
            # Format tags naturally
            tags_str = ", ".join(tags) if tags else "N/A"
            
            context_parts.append(
                f"• {item_name}\n"
                f"  Section: {section}\n"
                f"  Price: {price_str}\n"
                f"  Features: {tags_str}\n"
            )
        
        return "\n".join(context_parts)
    
    def _restaurant_info_section(self) -> str:
        """
        Build a restaurant information section for the system prompt from about.txt.
        This is static knowledge (address, hours, phone, description), separate from menu RAG.
        """
        if not self.restaurant_info:
            return ""
        return (
            "\n\nRESTAURANT INFORMATION (for internal reference):\n"
            f"{self.restaurant_info}\n"
        )
    
    def _is_new_session(self, conversation_history: Optional[List[Dict]]) -> bool:
        """
        Check if this is a new conversation session (no previous messages).
        Returns True if conversation_history is None or empty.
        """
        return not conversation_history or len(conversation_history) == 0
    
    def _get_welcome_message(self) -> str:
        """
        Returns the merged welcome message that should be shown only once per session.
        """
        if self.company:
            return self.company.welcome_message()
        return get_welcome_message()
    
    def _enhance_query_with_history(self, query: str, conversation_history: Optional[List[Dict]]) -> str:
        """
        Enhance query using conversation history for contextual understanding.
        Example: "tell me more about that dish" → "tell me more about biryani" (if biryani was mentioned)
        """
        if not conversation_history:
            return query
        
        query_lower = query.lower()
        
        # Check if query references previous conversation
        contextual_keywords = ['that', 'this', 'it', 'the', 'previous', 'above', 'mentioned', 'you said']
        is_contextual = any(keyword in query_lower for keyword in contextual_keywords)
        
        if not is_contextual:
            return query
        
        # Extract dish names from previous user queries
        dish_keywords = []
        for msg in conversation_history[-4:]:  # Last 4 messages
            if msg.get('role') == 'user':
                content = msg.get('content', '').lower()
                # Look for common dish-related words
                dish_words = ['biryani', 'curry', 'tikka', 'naan', 'dosa', 'chicken', 'mutton', 'fish',
                             'prawn', 'paneer', 'vegetable', 'spicy', 'starter', 'main',
                             'idly', 'idli', 'podi', 'vada', 'uttapam', 'uthappam', 'pongal',
                             'upma', 'bath', 'bonda', 'parotta', 'poori', 'rasam', 'sambar',
                             'korma', 'masala', 'dal', 'kebab', 'kulcha', 'roti']
                for word in dish_words:
                    if word in content and word not in dish_keywords:
                        dish_keywords.append(word)
        
        # If we found dish keywords, enhance the query
        if dish_keywords:
            enhanced = f"{query} {' '.join(dish_keywords)}"
            print(f"   🔍 Enhanced contextual query: '{query}' → '{enhanced}'")
            return enhanced
        
        return query
    
    def _sanitize_conversation_history(self, history: List[Dict]) -> List[Dict]:
        """
        Sanitize conversation history to prevent hallucination.
        Only includes user queries, not full assistant responses.
        This allows contextual understanding without copying wrong responses.
        """
        if not history:
            return []
        
        sanitized = []
        for msg in history:
            if msg.get('role') == 'user':
                # Include user queries - they provide context for understanding references
                sanitized.append({
                    "role": "user",
                    "content": msg.get('content', '')
                })
            # Skip assistant messages - they can cause contamination
            # The retrieved items in context are sufficient for generating responses
        
        return sanitized
    
    def _validate_response_intent(self, response: str, query: str, context: str) -> str:
        """
        Validate that response matches query intent and only mentions items from retrieved context.
        Logs warnings and detects hallucinations.
        """
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Extract item names from context (these are the ONLY valid items)
        import re
        # Extract item names from context format: "• ItemName\n"
        valid_items = re.findall(r'•\s+([^\n]+)', context)
        valid_items_lower = [item.lower().strip() for item in valid_items]
        
        # Check for vegetarian query (must not be a non-veg query)
        is_non_veg_query_check = any(word in query_lower for word in ['non-veg', 'non veg'])
        is_vegetarian_query = not is_non_veg_query_check and any(word in query_lower for word in ['vegetarian', 'veg', 'vegan'])
        
        if is_vegetarian_query:
            # Check if response mentions non-vegetarian items
            non_veg_keywords = ['chicken', 'mutton', 'fish', 'prawn', 'seafood', 'non-veg', 'non vegetarian', 'meat', 'egg']
            mentioned_non_veg = any(keyword in response_lower for keyword in non_veg_keywords)
            
            if mentioned_non_veg:
                print(f"⚠️  VALIDATION FAILED: Response mentions non-vegetarian items for vegetarian query!")
                print(f"   Query: {query}")
                print(f"   Response mentions: {[kw for kw in non_veg_keywords if kw in response_lower]}")
        
        # Check for common hallucinated dishes that are often mentioned but not in retrieved items
        hallucinated_dishes = {
            'paneer butter masala': 'Paneer Butter Masala',
            'paneer tikka': 'Paneer Tikka',
            'chicken pepper dry': 'Chicken Pepper Dry',
            'fish chilly': 'Fish Chilly',
            'vegetable jalfrezi': 'Vegetable Jalfrezi'
        }
        
        found_hallucinations = []
        for dish_key, dish_name in hallucinated_dishes.items():
            if dish_key in response_lower:
                # Check if this dish is actually in the valid items
                if dish_name.lower() not in valid_items_lower:
                    found_hallucinations.append(dish_name)
        
        if found_hallucinations:
            print(f"⚠️  VALIDATION FAILED: Response mentions dishes NOT in retrieved items!")
            print(f"   Query: {query}")
            print(f"   Valid items: {valid_items}")
            print(f"   Hallucinated dishes: {found_hallucinations}")
            print(f"   ⚠️  LLM is hallucinating - these dishes are not in the retrieved list!")
        
        return response
    
    def _build_system_prompt(self, context: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Build the complete system prompt with welcome message (only for new sessions) and restaurant info.
        
        NOTE: We intentionally ignore SYSTEM_PROMPT.md and always use this strong, inline prompt,
        so that menu queries cannot fall back to a weaker external prompt that allows AI disclaimers.
        """
        # Check if this is a new session
        is_new_session = self._is_new_session(conversation_history)
        
        # Load base prompt from file, fall back to minimal inline prompt
        if self.base_system_prompt:
            base_prompt = self.base_system_prompt.replace('{context}', context)
        elif self.tenant.is_company:
            base_prompt = (
                f"You are {self.tenant.assistant_name}, the customer support assistant of "
                f"{self.tenant.display_name}.\n\nKnowledge:\n{context}"
            )
        else:
            base_prompt = f"You are Chikku, the hospitality assistant of Saigon Indian Restaurant.\n\nCurrent Menu Context:\n{context}"
        
        # Add welcome message section if this is a new session
        welcome_section = ""
        if is_new_session:
            welcome_section = f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━
WELCOME MESSAGE TO USE (ONLY FOR FIRST MESSAGE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━

When the user sends their first message (greeting, "who are you", or any query), respond with this exact welcome message:

{self._get_welcome_message()}

CRITICAL: Only show this welcome message ONCE at the very beginning of a new conversation. After showing it, continue normally and NEVER repeat it again in the same session.

━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # Inject static restaurant information (about.txt) into the system prompt
        restaurant_info_section = self._restaurant_info_section()
        
        # Combine all parts
        final_prompt = base_prompt + welcome_section + restaurant_info_section
        
        return final_prompt
    
    def resolve_turn(self, query: str, language: Optional[str] = None) -> Tuple[str, str]:
        """Settle the language of a turn and produce the query to retrieve with.

        Returns (language, english_query). Retrieval, reranking and every intent
        classifier in this project operate on English, so a Tamil question is
        translated once here and the English form is what the rest of the
        pipeline sees. The original wording is never needed downstream — only
        the API echoes it back to the caller.

        Callers that stream must invoke this first and pass the English query to
        `prepare_stream_context` / `generate_response_stream`; `query()` does it
        for them.
        """
        lang = resolve_language(query, language)
        if lang == "en":
            return lang, query

        english = translate_to_english(self.provider, query, lang)
        return lang, english

    def prepare_stream_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank_k: Optional[int] = None,
    ) -> Tuple[str, List[Dict]]:
        """Retrieve + format the context for a streaming turn.

        Returns (context, items) where `items` is the display payload the API
        emits before the tokens. Keeps the tenant-specific retrieval logic in
        one place instead of duplicating it in the API layer.
        """
        top_k = max(1, top_k) if top_k is not None else self.top_k
        rerank_k = max(1, rerank_k) if rerank_k is not None else self.rerank_k

        if self.company:
            intent = self.company.classifier.classify(query)
            if intent.category.value == "conversational":
                return "", []

            retrieved = self.company.retrieve(query, intent, top_k)
            if not retrieved:
                return "", []

            if self.use_reranking:
                retrieved = self.rerank(query, retrieved, top_k=rerank_k)
            else:
                retrieved = retrieved[:rerank_k]

            items = [
                {
                    'name': i['metadata'].get('title'),
                    'section': i['metadata'].get('category_title'),
                    'price': None,
                    'currency': None,
                }
                for i in retrieved
            ]
            return self.company.format_context(retrieved), items

        # Restaurant path: only menu queries carry retrieved context.
        if not self.is_menu_query(query):
            return "", []

        retrieved = self.retrieve(query, top_k=top_k, doc_type="menu")
        if not retrieved:
            return "", []

        if self.use_reranking:
            retrieved = self.rerank(query, retrieved, top_k=rerank_k)
        else:
            retrieved = retrieved[:rerank_k]

        items = []
        for item in retrieved:
            meta = item.get('metadata', {}) if isinstance(item, dict) else {}
            raw_price = meta.get('price')
            items.append({
                'name': meta.get('item_name') or None,
                'section': meta.get('section') or None,
                'price': None if raw_price in (None, "") else str(raw_price),
                'currency': meta.get('currency') or None,
            })
        return self.format_context(retrieved), items

    def generate_response_stream(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None,
        language: Optional[str] = None
    ):
        """Generate streaming response using LLM.

        `query` must already be English (see `resolve_turn`); `language` is the
        language the reply has to come back in.
        """
        if self.company:
            intent = self.company.classifier.classify(query)
            if intent.category.value == "conversational" and not context:
                if not conversation_history:
                    # Canned brand intro, first turn only. It bypasses the LLM,
                    # so the answer directive cannot do the work — translate it
                    # (cached) before it is streamed.
                    yield localize(self.provider, self.company.welcome_message(), language)
                    return
                # Later chitchat turns are LLM-generated and answer natively
                # via the language directive; fallbacks handled inside.
                yield from self.company.stream_conversational(query, intent, conversation_history, language)
                return
            yield from self.company.stream(query, context, intent, conversation_history, language)
            return

        # Build system prompt with welcome message (only for new sessions) and restaurant info
        system_prompt = self._build_system_prompt(context, conversation_history)
        
        # Build messages with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last few messages for context)
        # Reduced from 6 to 4 for faster processing
        if conversation_history:
            for msg in conversation_history[-4:]:  # Last 4 messages for context
                if msg.get('role') in ['user', 'assistant']:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
        
        # Add current query with explicit instructions
        is_first_message = not conversation_history
        greeting_line = '1. Start with "Namaste! 🙏" — only on the FIRST message of a session.' if is_first_message else '1. Do NOT start with Namaste or any greeting — get straight to the answer.'
        user_prompt = f"""Customer Question: {query}

CRITICAL INSTRUCTIONS:
{greeting_line}
2. Use ONLY the menu items provided below - mention them naturally
3. Format prices as: X,XXX VND (e.g., "164,000 VND")
4. Describe items conversationally - mention key features
5. End with a warm follow-up question
6. NEVER say "I don't have access", "I'm an AI", or any AI disclaimers
7. You ARE Chikku, a real hospitality assistant - speak as a human would

Menu Items Available:
{context}

Respond naturally as Chikku."""
        user_prompt += answer_language_directive(language)
        messages.append({"role": "user", "content": user_prompt})

        try:
            print(f"🔄 Streaming LLM response ({self.llm_model})...")
            
            llm_options = {
                "max_tokens": 500,
                "temperature": 0.4,
                "top_p": 0.9,
                "num_ctx": 2048,
            }

            full_response = ""
            for content in self.provider.stream_chat(messages, llm_options):
                full_response += content
                yield content
            
            # Validate full response after streaming (if enabled)
            # Note: For streaming, we can't modify the stream, but we log validation issues
            if full_response:
                validator = get_validator()
                validation = validator.validate(
                    full_response, query=query, context_provided=bool(context), language=language
                )

                if not validation['valid']:
                    print(f"⚠️  Streamed response validation failed (score: {validation['score']:.2f})")
                    print(f"   Issues: {validation['issues']}")
                    # Note: Streamed responses are already sent, so we can't modify them
                    # This is mainly for monitoring/logging purposes
                        
        except Exception as e:
            print(f"❌ LLM streaming error: {str(e)}")
            yield f"\n\n[Error: {str(e)}]"
    
    def generate_response(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None,
        language: Optional[str] = None
    ) -> str:
        """Generate response using LLM with context.

        `query` must already be English (see `resolve_turn`); `language` is the
        language the reply has to come back in.
        """

        # Build system prompt with welcome message (only for new sessions) and restaurant info
        system_prompt = self._build_system_prompt(context, conversation_history)
        
        # Build messages with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add sanitized conversation history (only user queries, not full assistant responses)
        # This allows contextual understanding ("tell me more about that dish") without copying wrong responses
        if conversation_history:
            # Only include last 3 turns (6 messages) to keep context focused
            for msg in conversation_history[-6:]:
                if msg.get('role') == 'user':
                    # Include user queries - they provide context
                    messages.append({
                        "role": "user",
                        "content": msg.get('content', '')
                    })
                elif msg.get('role') == 'assistant':
                    # Sanitize assistant responses - only include key info, not full response
                    # This prevents LLM from copying entire responses
                    assistant_content = msg.get('content', '')
                    # Extract only dish names mentioned (if any) for context
                    # But don't include the full response text
                    # For now, we'll skip assistant messages to prevent contamination
                    # The retrieved items in context are sufficient
                    pass
        
        # Extract query intent for better instructions
        query_lower = query.lower()
        is_non_veg = any(word in query_lower for word in ['non-veg', 'non veg', 'chicken', 'mutton', 'fish', 'seafood', 'prawn', 'egg'])
        is_vegetarian = not is_non_veg and any(word in query_lower for word in ['vegetarian', 'veg', 'vegan'])
        is_spicy = any(word in query_lower for word in ['spicy', 'spice', 'hot', 'fiery'])
        is_starter = any(word in query_lower for word in ['starter', 'appetizer'])
        is_recommendation = any(word in query_lower for word in ['recommend', 'suggest', 'best', 'popular', 'good'])
        
        # Build query-specific instructions - SIMPLIFIED to avoid confusing LLM
        query_specific = ""
        
        is_order_query = any(phrase in query_lower for phrase in [
            'how to order', 'want to order', 'i want to order', 'would like to order',
            'place an order', 'order now', "i'd like to order"
        ])

        if is_vegetarian:
            query_specific += "\nIMPORTANT: User asked for VEGETARIAN options. Only mention vegetarian items from the list below.\n"
        elif is_non_veg:
            query_specific += "\nIMPORTANT: User asked for NON-VEGETARIAN options. Only mention non-vegetarian items from the list below.\n"

        if is_spicy:
            query_specific += "\nUser wants SPICY options. Prioritize spicy items.\n"
        if is_starter:
            query_specific += "\nUser wants STARTERS. Focus on starter/appetizer items.\n"
        if is_order_query:
            query_specific += (
                "\nIMPORTANT: User wants to ORDER this dish. Do the following in your response:\n"
                "1. Confirm the dish name and price.\n"
                "2. End with this EXACT block (do not skip it):\n"
                "   '🧑‍🍳 **Ready to order?** Please speak to our waiter — they will be happy to take your order right away!\n"
                "   Or you can also reach us at:\n"
                "   📞 +84 (028) 6291 3672\n"
                "   📍 26 Lê Anh Xuân Street, District 1 | Open 7:30 AM – 10:30 PM daily'\n"
            )
        
        # STRENGTHENED prompt - explicit bans on apologies
        is_first_message = not conversation_history
        greeting_line = '1. Start with "Namaste! 🙏" — only on the FIRST message of a session.' if is_first_message else '1. Do NOT start with Namaste or any greeting — get straight to the answer.'
        user_prompt = f"""Customer Question: {query}
{query_specific}

You are Chikku, a confident restaurant host at Saigon Indian Restaurant.

Respond directly and confidently:
{greeting_line}
2. Use ONLY the menu items listed below - mention them naturally
3. Format prices as: X,XXX VND (e.g., "154,000 VND")
4. Describe items naturally (spicy, creamy, grilled, etc.)
5. End with a warm follow-up question

FORBIDDEN PHRASES - NEVER USE THESE:
- "I apologize" or "I'm sorry"
- "I'm sorry for any confusion"
- "your request is not clear"
- "I would be more than happy to assist"
- "based on my capabilities"
- "within this context"
- "based on the information provided"
- Any dishes NOT in the list below
- Technical terms like "context", "retrieved items", "validation"

You are a real restaurant host. Be confident and direct. Answer the question immediately. No apologies, no explanations.

Menu Items:
{context}

Respond naturally as Chikku would, using ONLY the items listed above."""
        user_prompt += answer_language_directive(language)
        messages.append({"role": "user", "content": user_prompt})

        try:
            print(f"🔄 Calling LLM ({self.llm_model})...")
            print(f"   System prompt length: {len(system_prompt)} chars")
            print(f"   Conversation history: {len(conversation_history) if conversation_history else 0} messages")
            
            llm_options = {
                "max_tokens": 400,
                "temperature": 0.2,
                "top_p": 0.85,
                "num_ctx": 2048,
            }

            result = self.provider.chat(messages, llm_options)
            print(f"✅ LLM response received")
            if not result or len(result.strip()) == 0:
                return "I apologize, but I couldn't generate a response. Please try rephrasing your query."
            
            # Validate response matches query intent (e.g., vegetarian queries don't mention non-veg items)
            result = self._validate_response_intent(result, query, context)
            
            # Validate response quality (if enabled)
            validator = get_validator()
            validation = validator.validate(
                result, query=query, context_provided=bool(context), language=language
            )

            if not validation['valid']:
                print(f"⚠️  Response validation failed (score: {validation['score']:.2f})")
                print(f"   Issues: {validation['issues']}")
                
                # Use cleaned response if validation enabled
                if validator.enabled:
                    result = validation['cleaned_response']
                    print(f"✅ Using cleaned response")
            
            return result
        except Exception as e:
            print(f"❌ LLM error: {str(e)}")
            import traceback
            traceback.print_exc()
            # Use error handler for user-friendly message
            error_msg = handle_error(LLMError(str(e), model=self.llm_model), context={
                "query": query,
                "model": self.llm_model
            })
            return error_msg
    
    def filter_by_intent(self, query: str, items: List[Dict], intent_result=None) -> List[Dict]:
        """Filter items based on query intent (dietary preferences, protein type, etc.)"""
        if not items:
            return items
        
        query_lower = query.lower()
        filtered_items = []
        
        # Extract dietary preferences from query
        is_non_veg_query = any(word in query_lower for word in ['non-veg', 'non veg', 'nonvegetarian', 'meat', 'chicken', 'mutton', 'fish', 'seafood', 'prawn', 'egg'])
        is_vegetarian_query = not is_non_veg_query and any(word in query_lower for word in ['vegetarian', 'veg', 'vegan', 'jain'])
        is_spicy_query = any(word in query_lower for word in ['spicy', 'spice', 'hot', 'fiery', 'pepper'])
        is_starter_query = any(word in query_lower for word in ['starter', 'appetizer', 'appetiser', 'snack'])
        is_breakfast_query = any(word in query_lower for word in ['breakfast', 'morning', 'idly', 'idli', 'dosa', 'vada', 'pongal', 'upma', 'uttapam', 'uthappam'])
        is_dessert_query = any(word in query_lower for word in ['dessert', 'sweet', 'payasam', 'kheer', 'ice cream', 'halwa'])
        is_drink_query = any(word in query_lower for word in ['drink', 'beverage', 'juice', 'beer', 'wine', 'cocktail', 'mocktail', 'lassi', 'tea', 'coffee', 'water'])
        is_rice_query = 'rice' in query_lower and not any(w in query_lower for w in ['starter', 'breakfast', 'dessert'])
        is_bread_query = any(word in query_lower for word in ['naan', 'roti', 'bread', 'kulcha', 'paratha', 'poori'])
        is_thali_query = any(word in query_lower for word in ['thali', 'set lunch', 'combo', 'meal'])

        # Section maps (matches actual ChromaDB section values)
        STARTER_SECTIONS = {'veg starters', 'non veg starters', 'non - vegeterian srarters – from tandoor', 'salads/papad/appetizer'}
        BREAKFAST_SECTIONS = {'breakfast'}
        DESSERT_SECTIONS = {'desserts & sweets'}
        DRINK_SECTIONS = {'beer', 'wine', 'cocktails', 'mocktails', 'liquor', 'hot & cold beverages', 'native hot beverages', 'native cold beverages'}
        RICE_SECTIONS = {'rice - veg', 'rice non-veg'}
        BREAD_SECTIONS = {'from the clay pot – tandoor - breads', 'from the clay pot – tandoor - breadsg'}
        THALI_SECTIONS = {'thalis/set lunch'}
        
        # Extract protein type
        protein_type = None
        if 'chicken' in query_lower:
            protein_type = 'chicken'
        elif 'mutton' in query_lower or 'lamb' in query_lower:
            protein_type = 'mutton'
        elif 'fish' in query_lower:
            protein_type = 'fish'
        elif 'prawn' in query_lower or 'shrimp' in query_lower:
            protein_type = 'prawn'
        elif 'egg' in query_lower:
            protein_type = 'egg'
        elif 'seafood' in query_lower:
            protein_type = 'seafood'
        
        for item in items:
            metadata = item.get('metadata', {})
            tags_str = metadata.get('tags', '')
            tags = tags_str.lower() if isinstance(tags_str, str) else str(tags_str).lower()
            item_name = metadata.get('item_name', '').lower()
            section = metadata.get('section', '').lower()
            
            # IMPROVED: Stricter filtering by dietary preference
            if is_vegetarian_query:
                # STRICT: For vegetarian queries, exclude ANY non-vegetarian items
                if 'non-vegetarian' in tags or 'non vegetarian' in tags:
                    continue
                # Check item name for non-veg indicators (more comprehensive)
                non_veg_indicators = ['chicken', 'mutton', 'fish', 'prawn', 'shrimp', 'meat', 'egg', 'seafood']
                if any(indicator in item_name for indicator in non_veg_indicators):
                    continue
                # Must have vegetarian tag OR be clearly vegetarian by name
                if 'vegetarian' not in tags and 'vegan' not in tags:
                    # Allow if it's clearly vegetarian (paneer, vegetable, etc.)
                    veg_indicators = ['paneer', 'vegetable', 'dal', 'dhal', 'naan', 'roti', 'raita', 'salad']
                    if not any(indicator in item_name for indicator in veg_indicators):
                        continue
            
            if is_non_veg_query:
                # STRICT: For non-veg queries, exclude vegetarian-only items
                if 'vegetarian' in tags and 'non-vegetarian' not in tags:
                    # Check if it's actually non-veg by name
                    non_veg_indicators = ['chicken', 'mutton', 'fish', 'prawn', 'shrimp', 'meat', 'egg', 'seafood']
                    if not any(indicator in item_name for indicator in non_veg_indicators):
                        continue
            
            # Filter by protein type
            if protein_type:
                if protein_type == 'chicken' and 'chicken' not in item_name:
                    continue
                elif protein_type == 'mutton' and 'mutton' not in item_name:
                    continue
                elif protein_type == 'fish' and 'fish' not in item_name:
                    continue
                elif protein_type == 'prawn' and 'prawn' not in item_name and 'shrimp' not in item_name:
                    continue
                elif protein_type == 'egg' and 'egg' not in item_name:
                    continue
                elif protein_type == 'seafood':
                    if 'fish' not in item_name and 'prawn' not in item_name and 'shrimp' not in item_name:
                        continue
            
            # Section-based course filtering
            if is_starter_query and STARTER_SECTIONS:
                if section not in STARTER_SECTIONS:
                    continue
            elif is_breakfast_query and not any(w in query_lower for w in ['chicken', 'mutton', 'fish', 'prawn']):
                if section not in BREAKFAST_SECTIONS:
                    continue
            elif is_dessert_query:
                if section not in DESSERT_SECTIONS:
                    continue
            elif is_drink_query:
                if section not in DRINK_SECTIONS:
                    continue
            elif is_thali_query:
                if section not in THALI_SECTIONS:
                    continue
            
            filtered_items.append(item)
        
        # IMPROVED: If filtering removed everything, return original but log warning
        if not filtered_items and items:
            print(f"⚠️  Intent filtering removed all items, using original {len(items)} items")
            return items
        
        return filtered_items
    
    def filter_by_relevance(self, query: str, items: List[Dict]) -> List[Dict]:
        """Filter items by keyword relevance to query"""
        if not items:
            return items
        
        # Extract keywords from query (lowercase for matching)
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Common query patterns
        sweet_keywords = {'sweet', 'dessert', 'sugar', 'candy', 'honey'}
        savory_keywords = {'spicy', 'savory', 'salty', 'main', 'course', 'curry', 'masala'}
        
        # Check if query is about sweet items
        is_sweet_query = any(word in query_words for word in sweet_keywords)
        is_savory_query = any(word in query_words for word in savory_keywords)
        
        filtered_items = []
        for item in items:
            metadata = item.get('metadata', {})
            item_name = metadata.get('item_name', '').lower()
            tags = metadata.get('tags', '').lower()
            section = metadata.get('section', '').lower()
            
            # Check if item matches query keywords
            item_text = f"{item_name} {tags} {section}"
            
            # For sweet queries, filter out savory items
            if is_sweet_query:
                # Check if item is actually sweet
                item_is_sweet = any(word in item_text for word in ['sweet', 'dessert', 'lassi', 'juice', 'shake', 'ice cream', 'cake'])
                item_is_savory = any(word in item_text for word in ['paneer', 'chicken', 'mutton', 'curry', 'masala', 'chilly', 'pepper', 'dry'])
                
                if item_is_savory and not item_is_sweet:
                    print(f"   ⚠️  Filtering out non-sweet item: {metadata.get('item_name')}")
                    continue
            
            # For savory queries, filter out sweet items
            if is_savory_query and not is_sweet_query:
                item_is_sweet = any(word in item_text for word in ['sweet', 'dessert', 'lassi', 'juice', 'shake'])
                if item_is_sweet:
                    print(f"   ⚠️  Filtering out sweet item: {metadata.get('item_name')}")
                    continue
            
            # If we've already filtered by sweet/savory, add the item
            # Otherwise check if item matches query keywords
            if is_sweet_query or is_savory_query:
                # Already handled above, add item if it passed sweet/savory filter
                filtered_items.append(item)
            else:
                # For general queries, check if item matches any query word
                matches_query = any(word in item_text for word in query_words if len(word) > 2)
                if matches_query or len(query_words) == 0:
                    filtered_items.append(item)
        
        # If filtering removed all items, return original (better than nothing)
        if not filtered_items and items:
            print(f"   ⚠️  Relevance filtering removed all items, using original results")
            return items
        
        return filtered_items
    
    def _classify_intent(self, query: str):
        """
        Classify query intent using intent classifier.
        Returns IntentResult object.
        """
        try:
            classifier = get_intent_classifier()
            return classifier.classify_intent(query)
        except Exception as e:
            # Log error but continue with default behavior
            print(f"⚠️  Intent classification error: {str(e)}")
            # Return default menu intent as fallback
            from intent_classifier import IntentResult, IntentType, IntentCategory
            return IntentResult(
                intent_type=IntentType.MENU_SEARCH,
                confidence=0.5,
                category=IntentCategory.MENU
            )
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize speech-to-text errors using LLM.
        Fixes homophones, mishearing, wrong words in context of an Indian restaurant.
        Enabled via QUERY_NORMALIZATION=true in .env.
        """
        if not self.query_normalization:
            return query

        # Skip normalization for very short queries or greetings — not worth the latency
        query_stripped = query.strip()
        if len(query_stripped) <= 6:
            return query

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a speech-to-text correction assistant for an Indian restaurant chatbot. "
                        "Your ONLY job is to fix obvious pronunciation or mishearing errors "
                        "(e.g., 'biriany' → 'biryani', 'dossa' → 'dosa', 'nann' → 'naan'). "
                        "STRICT RULES: "
                        "1. NEVER replace one valid word with a different word "
                        "(do NOT change 'beer' to 'biryani', 'egg' to 'chicken', etc.). "
                        "2. NEVER convert a non-food or conversational query into a food query. "
                        "3. If the query has no obvious pronunciation error, return it EXACTLY as given. "
                        "4. Only correct a word if the corrected version sounds nearly identical to the original. "
                        "Context: Indian food (biryani, idly, dosa, podi, paneer, korma, tikka, naan, "
                        "beer, wine, cocktails, whisky). "
                        "Return ONLY the corrected query — no explanation, no extra text."
                    )
                },
                {"role": "user", "content": query_stripped}
            ]
            corrected = self.provider.chat(messages, {"max_tokens": 80, "temperature": 0.0}).strip()
            # Safety checks
            if corrected and len(corrected) < len(query_stripped) * 3:
                # Semantic drift guard: reject if more than 1 new significant word introduced
                orig_words = set(re.findall(r'\b\w+\b', query_stripped.lower()))
                corr_words = set(re.findall(r'\b\w+\b', corrected.lower()))
                new_significant = [w for w in (corr_words - orig_words) if len(w) > 3]
                if len(new_significant) > 1:
                    print(f"⚠️  Normalization rejected (semantic drift): '{query_stripped}' → '{corrected}'")
                    return query
                if corrected != query_stripped:
                    print(f"🔧 Query normalized: '{query_stripped}' → '{corrected}'")
                return corrected
        except Exception as e:
            print(f"⚠️  Query normalization failed: {e}")

        return query

    def _extract_dish_from_order_query(self, query: str) -> str:
        """
        Strip ordering preamble from a query to get the dish name for retrieval.
        e.g. 'I want to order Chicken Tikka Masala' → 'Chicken Tikka Masala'
        """
        order_prefixes = [
            r'^how\s+to\s+order\s+',
            r'^i\s+want\s+to\s+order\s+',
            r'^i\s+would\s+like\s+to\s+order\s+',
            r"^i'?d\s+like\s+to\s+order\s+",
            r'^can\s+i\s+order\s+',
            r'^want\s+to\s+order\s+',
            r'^let\s+me\s+order\s+',
            r'^place\s+an?\s+order\s+for\s+',
            r'^order\s+',
        ]
        query_stripped = query.strip().rstrip('?.!')
        for prefix in order_prefixes:
            cleaned = re.sub(prefix, '', query_stripped, flags=re.IGNORECASE).strip()
            if cleaned and cleaned.lower() != query_stripped.lower():
                return cleaned.rstrip('?.!')
        return query_stripped

    def is_menu_query(self, query: str) -> bool:
        """
        Detect if query is about menu items (food, dishes, prices) vs identity/about/restaurant info.
        Returns True if menu query, False if identity/about query.
        Uses new intent classifier for accurate detection.
        """
        intent_result = self._classify_intent(query)
        is_menu = intent_result.is_menu_intent()
        
        print(f"🔍 Intent: {intent_result.intent_type.value} (confidence: {intent_result.confidence:.2f})")
        
        return is_menu
    
    def answer_about_or_identity(self, query: str, conversation_history: Optional[List[Dict]] = None, intent_result=None, language: Optional[str] = None) -> str:
        """
        Generate response for identity/about queries without menu retrieval.
        Uses deterministic welcome message for identity queries to prevent LLM hallucinations.

        Most branches here return a fixed English template; `query()` translates
        those at the exit. `language` is only needed by the one branch that calls
        the LLM, so that it answers natively instead of being translated twice.
        """
        query_lower = query.lower().strip()
        is_new_session = self._is_new_session(conversation_history)
        
        # Check if it's a greeting/identity query
        is_greeting = any(phrase in query_lower for phrase in [
            'who are you', 'what are you', 'who is chikku', 'what is chikku',
            'hello', 'hi', 'hey', 'namaste', 'greetings'
        ])
        
        # Check if it's an "about" query
        is_about_query = any(phrase in query_lower for phrase in [
            'tell me about', 'about saigon', 'about restaurant', 'about the restaurant'
        ])
        
        # DETERMINISTIC: Use templates for identity queries
        # This prevents LLM from hallucinating ChatGPT responses
        if is_greeting:
            if intent_result and intent_result.intent_type:
                template = get_template(intent_result.intent_type)
                if template:
                    if is_new_session and intent_result.intent_type == IntentType.IDENTITY_WHO_ARE_YOU:
                        print("✅ Returning welcome message (deterministic, new session)")
                        return template
                    elif not is_new_session:
                        print("✅ Returning identity response (deterministic)")
                        return template
            
            # Fallback
            if is_new_session:
                print("✅ Returning welcome message (deterministic, new session)")
                return self._get_welcome_message()
            else:
                print("✅ Returning identity response (deterministic)")
                return "I'm Chikku, your personal food companion, flavor guide, and menu expert at Saigon Indian Restaurant. I'm here to help you discover the perfect dish! What are you craving today? 😊"
        
        # Check if template exists for restaurant info queries
        # Use knowledge base data to populate templates
        if intent_result and intent_result.intent_type:
            # Use knowledge base for restaurant info queries
            if intent_result.intent_type == IntentType.RESTAURANT_LOCATION:
                kb_location = self.knowledge_base.format_location()
                if kb_location:
                    print(f"✅ Using knowledge base for location")
                    return kb_location
            
            elif intent_result.intent_type == IntentType.RESTAURANT_HOURS:
                kb_hours = self.knowledge_base.format_hours()
                if kb_hours:
                    print(f"✅ Using knowledge base for hours")
                    return kb_hours
            
            elif intent_result.intent_type == IntentType.RESTAURANT_CONTACT:
                kb_contact = self.knowledge_base.format_contact()
                if kb_contact:
                    print(f"✅ Using knowledge base for contact")
                    return kb_contact
            
            elif intent_result.intent_type == IntentType.RESTAURANT_PARKING:
                accessibility = self.knowledge_base.get_accessibility()
                if accessibility and accessibility.get('parking'):
                    parking_info = accessibility['parking']
                    return f"""🅿️ **Parking Information**

{parking_info.get('alternatives', 'Street parking available')}

For the best parking options, I'd recommend calling us at +84 (028) 6291 3672, and our team can guide you to the nearest parking! 😊"""
            
            elif intent_result.intent_type == IntentType.RESTAURANT_ACCESSIBILITY:
                accessibility = self.knowledge_base.get_accessibility()
                if accessibility:
                    wheelchair = accessibility.get('wheelchair_access', {})
                    return f"""♿ **Accessibility**

We're committed to making our restaurant accessible to all guests. Our restaurant is located on the ground floor with easy access.

{wheelchair.get('details', 'Restaurant is located on ground floor with easy access')}

For specific accessibility needs or questions, please call us at +84 (028) 6291 3672, and we'll be happy to assist you and ensure your visit is comfortable! 😊"""
            
            # Check template as fallback
            template = get_template(intent_result.intent_type)
            if template:
                print(f"✅ Using template for {intent_result.intent_type.value}")
                return template
        
        # For "about" queries without template, use knowledge base or LLM
        if is_about_query:
            # Try knowledge base first
            kb_info = self.knowledge_base.get_restaurant_info_text()
            if kb_info:
                print("📋 Using knowledge base for about query")
                return f"""Welcome to Saigon Indian Restaurant! 🇮🇳✨

{kb_info}

I'm Chikku, your personal food companion here. How can I help you discover our menu today?"""
            
            # Fallback to LLM with restaurant info
            if self.restaurant_info:
                print("📋 Generating about response using restaurant info...")
                context = f"RESTAURANT INFORMATION:\n{self.restaurant_info}"
                system_prompt = self._build_system_prompt(context, conversation_history)
            
            # Build messages with VERY STRICT instructions
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-4:]:
                    if msg.get('role') in ['user', 'assistant']:
                        messages.append({
                            "role": msg['role'],
                            "content": msg['content']
                        })
            
            # Add user query with STRICT instructions
            user_prompt = f"""Query: {query}

CRITICAL INSTRUCTIONS:
- You are Chikku, a real hospitality assistant at Saigon Indian Restaurant.
- You are NOT an AI model, NOT ChatGPT, NOT a language model.
- Answer using ONLY the restaurant information provided below.
- Sound warm, human, and professional.
- NEVER mention AI, ChatGPT, models, or technology.

Restaurant Information:
{self.restaurant_info}

Now answer the query naturally as Chikku."""

            user_prompt += answer_language_directive(language)
            messages.append({"role": "user", "content": user_prompt})
            
            try:
                print(f"🔄 Generating about response ({self.llm_model})...")
                
                llm_options = {
                    "max_tokens": 300,
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "num_ctx": 2048,
                }

                result = self.provider.chat(messages, llm_options).strip()
                
                # Safety check: if response contains AI disclaimers, use fallback
                ai_keywords = ['chatgpt', 'ai model', 'language model', 'artificial intelligence', 'openai', 'google']
                if any(keyword in result.lower() for keyword in ai_keywords):
                    print("⚠️  LLM generated AI disclaimer, using fallback")
                    # Return restaurant info directly formatted nicely
                    return f"""Welcome to Saigon Indian Restaurant! 🇮🇳✨

{self.restaurant_info}

I'm Chikku, your personal food companion here. How can I help you discover our menu today?"""
                
                if result:
                    return result
                else:
                    # Fallback
                    return f"""Welcome to Saigon Indian Restaurant! 🇮🇳✨

{self.restaurant_info}

I'm Chikku, your personal food companion here. How can I help you discover our menu today?"""
            except Exception as e:
                print(f"❌ LLM error in answer_about_or_identity: {str(e)}")
                # Fallback
                return f"""Welcome to Saigon Indian Restaurant! 🇮🇳✨

{self.restaurant_info}

I'm Chikku, your personal food companion here. How can I help you discover our menu today?"""
        
        # Default fallback
        return "I'm Chikku, your personal food companion at Saigon Indian Restaurant. How can I help you today?"
    
    def answer_service_query(self, query: str, conversation_history: Optional[List[Dict]] = None, intent_result=None) -> str:
        """
        Handle service queries (reservations, events, etc.)
        Uses knowledge base data + templates to prevent LLM hallucinations.
        NOTE: Reservation flow is single-turn (no complex dialog) by design.
        """
        if not intent_result or not intent_result.intent_type:
            # Fallback
            return """I can help you with reservations, events, catering, and more!

For service inquiries, please call us at:
📞 +84 (028) 6291 3672 or +84 (028) 3824 5671

Or tell me what you need, and I'll guide you! 😊"""
        
        intent_type = intent_result.intent_type
        
        # Use knowledge base for service queries
        if intent_type == IntentType.SERVICE_RESERVATION:
            contact = self.knowledge_base.get_contact()
            phone = contact.get('phone', {}).get('formatted', '+84 (028) 6291 3672 / +84 (028) 3824 5671') if contact else '+84 (028) 6291 3672 / +84 (028) 3824 5671'
            return (
                f"I'd be happy to help you with a reservation. ✨\n\n"
                f"For the fastest and most accurate booking, please call us directly at 📞 {phone}.\n\n"
                "If you share your preferred date, time, and number of guests here, I can also help you double‑check availability style (but final confirmation is always through our team). 😊"
            )
        
        elif intent_type == IntentType.SERVICE_DELIVERY:
            delivery_info = self.knowledge_base.get_delivery_info()
            if delivery_info:
                contact = self.knowledge_base.get_contact()
                phone = contact.get('phone', {}).get('formatted', '+84 (028) 6291 3672 / +84 (028) 3824 5671') if contact else '+84 (028) 6291 3672 / +84 (028) 3824 5671'
                
                return f"""We offer delivery services! 🚚

For delivery orders, please call us at:
📞 {phone}

Our team will be happy to help you place your order and arrange delivery to your location.

What would you like to order? I can help you explore our menu! 🍽️"""
        
        elif intent_type == IntentType.SERVICE_EVENT_BOOKING:
            events_info = self.knowledge_base.get_events_info()
            if events_info:
                contact = self.knowledge_base.get_contact()
                phone = contact.get('phone', {}).get('formatted', '+84 (028) 6291 3672 / +84 (028) 3824 5671') if contact else '+84 (028) 6291 3672 / +84 (028) 3824 5671'
                
                event_types = events_info.get('event_types', {})
                types_list = [k.replace('_', ' ').title() for k in event_types.keys() if event_types[k].get('available')]
                
                return f"""We'd love to host your event! 🎉

**We offer:**
{chr(10).join(f'• {t}' for t in types_list)}

For event bookings, please call us at:
📞 {phone}

Our team can customize menus and arrangements to make your event memorable! Tell me what kind of event you're planning, and I can help! 😊"""
        
        elif intent_type == IntentType.SERVICE_CATERING:
            catering_info = self.knowledge_base.get_catering_info()
            if catering_info:
                contact = self.knowledge_base.get_contact()
                phone = contact.get('phone', {}).get('formatted', '+84 (028) 6291 3672 / +84 (028) 3824 5671') if contact else '+84 (028) 6291 3672 / +84 (028) 3824 5671'
                
                return f"""We offer catering services! 🍽️

For catering inquiries and custom menus, please call us at:
📞 {phone}

We can tailor our menu to your event needs. Tell me about your event, and I can help guide you! 😊"""
        
        # Fallback to template
        template = get_template(intent_type)
        if template:
            print(f"✅ Using template for {intent_type.value}")
            return template
        
        # Final fallback
        return """I can help you with reservations, events, catering, and more!

For service inquiries, please call us at:
📞 +84 (028) 6291 3672 or +84 (028) 3824 5671

Or tell me what you need, and I'll guide you! 😊"""
    
    def ask_clarification(self, query: str, intent_result=None) -> str:
        """
        Ask user for clarification when intent is unclear.
        Uses deterministic template.
        """
        return get_error_response('clarification_needed')
    
    def query(
        self,
        user_query: str,
        top_k: Optional[int] = None,
        rerank_k: Optional[int] = None,
        conversation_history: Optional[List[Dict]] = None,
        language: Optional[str] = None
    ) -> Dict:
        """Complete RAG query pipeline, in the caller's language.

        The English pipeline is left exactly as it was; this wraps it with the
        two translation boundaries. Doing it here rather than inside the pipeline
        is what keeps the change small: the restaurant path returns hardcoded
        English templates from around thirty different places, and every one of
        them passes through this single exit.
        """
        lang, english_query = self.resolve_turn(user_query, language)

        result = self._query_english(
            english_query, top_k, rerank_k, conversation_history, language=lang
        )

        # LLM answers already come back in `lang` (answer_language_directive);
        # localize() detects that and only translates the template responses.
        result['response'] = localize(self.provider, result.get('response', ''), lang)
        # Echo what the customer actually asked, not the translation.
        result['query'] = user_query
        result['language'] = lang
        if english_query != user_query:
            result['query_english'] = english_query

        return result

    def _query_english(
        self,
        user_query: str,
        top_k: Optional[int] = None,
        rerank_k: Optional[int] = None,
        conversation_history: Optional[List[Dict]] = None,
        language: Optional[str] = None
    ) -> Dict:
        """The English RAG pipeline. `user_query` must already be English."""
        # Use instance defaults or provided values, ensure minimum of 1
        top_k = max(1, top_k) if top_k is not None else self.top_k
        rerank_k = max(1, rerank_k) if rerank_k is not None else self.rerank_k

        # Company tenants use the generic company-support path; everything below
        # this point is the restaurant menu pipeline.
        if self.company:
            return self.company.query(
                user_query, top_k, rerank_k, conversation_history, language=language
            )

        # Step 0a: Classify intent first on the raw query
        intent_result = self._classify_intent(user_query)
        print(f"📋 Intent classified: {intent_result.intent_type.value} ({intent_result.category.value})")

        # Step 0b: Normalize only for menu queries — skip for conversational/identity to prevent corruption
        if intent_result.category == IntentCategory.MENU:
            normalized = self.normalize_query(user_query)
            if normalized != user_query:
                user_query = normalized
                intent_result = self._classify_intent(user_query)
                print(f"📋 Re-classified after normalization: {intent_result.intent_type.value} ({intent_result.category.value})")
        
        # Route based on intent category
        if intent_result.category == IntentCategory.MENU:
            # Continue with menu RAG pipeline
            pass
        elif intent_result.category in [IntentCategory.IDENTITY, IntentCategory.RESTAURANT_INFO, IntentCategory.CONVERSATIONAL]:
            # Use identity/about handler
            print(f"📋 Non-menu query detected, using identity/about handler...")
            response = self.answer_about_or_identity(user_query, conversation_history, intent_result, language=language)
            return {
                'query': user_query,
                'response': response,
                'items': [],
                'retrieved_count': 0,
                'intent': intent_result.to_dict()
            }
        elif intent_result.category == IntentCategory.SERVICE:
            # Service queries (reservations, etc.) - handle separately
            print(f"📋 Service query detected: {intent_result.intent_type.value}")
            response = self.answer_service_query(user_query, conversation_history, intent_result)
            return {
                'query': user_query,
                'response': response,
                'items': [],
                'retrieved_count': 0,
                'intent': intent_result.to_dict()
            }
        elif intent_result.requires_clarification:
            # Low confidence — send directly to LLM instead of generic template
            print(f"📋 Low confidence ({intent_result.confidence:.2f}), falling back to LLM...")
            try:
                system_prompt = self._build_system_prompt("", conversation_history)
                messages = [{"role": "system", "content": system_prompt}]
                if conversation_history:
                    for msg in conversation_history[-4:]:
                        if msg.get('role') in ['user', 'assistant']:
                            messages.append({"role": msg['role'], "content": msg['content']})
                messages.append({
                    "role": "user",
                    "content": user_query + answer_language_directive(language),
                })
                response = self.provider.chat(messages, {
                    "max_tokens": 300,
                    "temperature": 0.5,
                    "top_p": 0.9,
                }).strip()
            except Exception as e:
                print(f"⚠️  LLM fallback failed: {e}")
                response = self.ask_clarification(user_query, intent_result)
            return {
                'query': user_query,
                'response': response,
                'items': [],
                'retrieved_count': 0,
                'intent': intent_result.to_dict()
            }
        else:
            # Default: treat as menu query
            print(f"📋 Unclear intent, defaulting to menu query...")
        
        # Step 1: Enhance query with conversation history for contextual understanding
        enhanced_query = self._enhance_query_with_history(user_query, conversation_history)

        # For ORDER intent, extract the dish name so the vector search targets the dish accurately
        if intent_result.intent_type == IntentType.MENU_ORDER:
            dish_name = self._extract_dish_from_order_query(user_query)
            if dish_name and dish_name.lower() != user_query.lower().strip():
                print(f"🍽️  Order query: using extracted dish name for retrieval: '{dish_name}'")
                enhanced_query = dish_name

        # Step 2: Retrieve (only menu items)
        print(f"📥 Retrieving top {top_k} menu items...")
        print(f"   Collection has {self.collection.count()} total items")
        retrieved = self.retrieve(enhanced_query, top_k=top_k, doc_type="menu")
        print(f"✅ Retrieved {len(retrieved)} items")
        
        if not retrieved:
            print(f"⚠️  No menu items found for query: '{user_query}'")
            # Try without doc_type filter to see if ANY items exist
            print(f"   Testing retrieval without filter...")
            test_retrieved = self.retrieve(user_query, top_k=top_k, doc_type=None)
            print(f"   Without filter: {len(test_retrieved)} items")
            if test_retrieved:
                print(f"   Sample item metadata keys: {list(test_retrieved[0].get('metadata', {}).keys())}")
            
            if intent_result.intent_type == IntentType.MENU_ORDER:
                dish_name = self._extract_dish_from_order_query(user_query)
                response = (
                    f"I'm sorry, I couldn't find **{dish_name}** in our current menu. 😊\n\n"
                    f"🧑‍🍳 **Please speak to our waiter** — they can check availability and take your order directly!\n\n"
                    f"Or reach us at:\n"
                    f"📞 +84 (028) 6291 3672\n"
                    f"📍 26 Lê Anh Xuân Street, District 1 | Open 7:30 AM – 10:30 PM daily"
                )
            else:
                response = get_error_response('no_menu_results')
            return {
                'query': user_query,
                'response': response,
                'items': [],
                'retrieved_count': 0,
                'intent': intent_result.to_dict()
            }
        
        # Step 3: Filter by intent (dietary preferences, protein type, etc.)
        print(f"🔍 Filtering items by query intent...")
        retrieved_before = len(retrieved)
        retrieved = self.filter_by_intent(user_query, retrieved, intent_result)
        print(f"✅ {len(retrieved)} items after intent filtering (was {retrieved_before})")
        
        # Step 4: Filter by keyword relevance
        print(f"🔍 Filtering items by keyword relevance...")
        retrieved_before_relevance = len(retrieved)
        retrieved = self.filter_by_relevance(user_query, retrieved)
        print(f"✅ {len(retrieved)} items after relevance filtering (was {retrieved_before_relevance})")
        
        # If filtering removed everything, use original results
        if not retrieved and retrieved_before > 0:
            print(f"⚠️  Filtering removed all items, using original {retrieved_before} items")
            retrieved = self.retrieve(user_query, top_k=top_k, doc_type="menu")
            retrieved = [item for item in retrieved if item.get('metadata', {}).get('doc_type') != 'about']
        
        # Step 5: Rerank (optional) - IMPROVED with smart limits
        # Smart limit: Even if user requests top_k=40, limit final context to 10-15 items for better LLM focus
        # This prevents apologies and improves quality
        MAX_CONTEXT_ITEMS = 15  # Optimal context size for LLM
        
        if self.use_reranking:
            # Use reranking to prioritize best items
            # Rerank to more items than needed, then limit to MAX_CONTEXT_ITEMS
            rerank_to = min(len(retrieved), max(rerank_k, MAX_CONTEXT_ITEMS))
            print(f"🔄 Reranking to top {rerank_to} items...")
            retrieved = self.rerank(user_query, retrieved, top_k=rerank_to)
            # Limit to optimal context size
            retrieved = retrieved[:MAX_CONTEXT_ITEMS]
            print(f"✅ Using top {len(retrieved)} items after reranking (optimal context size)")
        else:
            # Without reranking, still limit to optimal context size for better quality
            # This prevents too much context causing apologies
            max_items = min(len(retrieved), MAX_CONTEXT_ITEMS)
            retrieved = retrieved[:max_items]
            print(f"📊 Using top {len(retrieved)} items (limited to optimal context size for better quality)")
        
        # Step 6: Format context (use all retrieved items)
        print(f"📝 Formatting context for {len(retrieved)} items...")
        context = self.format_context(retrieved)
        
        # Step 7: Generate response with improved prompt
        # Use conversation history BUT sanitize it to prevent hallucination
        # History is needed for contextual queries like "tell me more about that dish"
        print(f"🤖 Generating response...")
        # Sanitize conversation history: only include user queries, not full assistant responses
        sanitized_history = self._sanitize_conversation_history(conversation_history) if conversation_history else None
        response = self.generate_response(user_query, context, sanitized_history, language=language)
        print(f"✅ Response generated ({len(response)} chars)")
        
        # Prepare result (include all retrieved items)
        intent_result = self._classify_intent(user_query)
        result = {
            'query': user_query,
            'response': response,
            'items': [
                {
                    'name': item['metadata'].get('item_name'),
                    'section': item['metadata'].get('section'),
                    'price': item['metadata'].get('price'),
                    'currency': item['metadata'].get('currency'),
                    'tags': item['metadata'].get('tags', '').split(', ') if item['metadata'].get('tags') else []
                }
                for item in retrieved
            ],
            'retrieved_count': len(retrieved),
            'intent': intent_result.to_dict()
        }
        
        return result


def main():
    """Test RAG pipeline"""
    project_root = Path(__file__).parent.parent
    chroma_db_path = project_root / "chroma_db"
    
    # Initialize pipeline
    print("Initializing RAG pipeline...")
    rag = RAGPipeline(
        chroma_db_path=str(chroma_db_path),
        use_reranking=False  # Set to True if reranking installed
    )
    
    # Test with single query first
    print("\n" + "="*60)
    print("Testing RAG Pipeline")
    print("="*60 + "\n")
    
    query = "What vegetarian dosas do you have?"
    print(f"Query: {query}")
    print("-" * 60)
    
    try:
        result = rag.query(query, top_k=10, rerank_k=5)
        
        print(f"\nResponse:\n{result['response']}")
        print(f"\nRetrieved {result['retrieved_count']} items")
        if result['items']:
            print("\nTop items:")
            for item in result['items']:
                price_str = f"{item['price']} {item['currency']}" if item.get('price') else "N/A"
                print(f"  - {item['name']} ({item['section']}) - {price_str}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
