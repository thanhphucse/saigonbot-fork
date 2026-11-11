"""
LLM Provider abstraction layer.

Embeddings:  always Ollama (mxbai-embed-large) — keeps ChromaDB index stable.
Generation:  switch via LLM_PROVIDER in .env
               LLM_PROVIDER=ollama  → local Ollama model
               LLM_PROVIDER=openai  → OpenAI API
"""
import os
from typing import List, Dict, Iterator
from dotenv import load_dotenv

load_dotenv()


class LLMProvider:
    """Base interface for LLM providers."""

    llm_model: str
    embedding_model: str

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError

    def chat(self, messages: List[Dict], options: Dict) -> str:
        """Non-streaming generation. options keys: max_tokens, temperature, top_p, num_ctx"""
        raise NotImplementedError

    def stream_chat(self, messages: List[Dict], options: Dict) -> Iterator[str]:
        """Streaming generation — yields text chunks."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Ollama — used for BOTH embeddings and generation
# ---------------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    def __init__(self, llm_model: str, embedding_model: str):
        import ollama as _ollama
        self._ollama = _ollama
        self.llm_model = llm_model
        self.embedding_model = embedding_model

    def _ollama_options(self, options: Dict) -> Dict:
        opts = {
            "num_predict": options.get("max_tokens", 512),
            "temperature": options.get("temperature", 0.7),
            "top_p":       options.get("top_p", 0.9),
            "num_ctx":     options.get("num_ctx", 2048),
        }
        num_gpu_env = os.getenv("OLLAMA_NUM_GPU")
        if num_gpu_env is not None:
            try:
                num_gpu = int(num_gpu_env)
                opts["num_gpu"] = num_gpu
                if num_gpu == 0:
                    print("   ⚙️  Forcing CPU usage (OLLAMA_NUM_GPU=0)")
            except (ValueError, TypeError):
                pass
        return opts

    def embed(self, text: str) -> List[float]:
        response = self._ollama.embeddings(model=self.embedding_model, prompt=text)
        return response["embedding"]

    def chat(self, messages: List[Dict], options: Dict) -> str:
        response = self._ollama.chat(
            model=self.llm_model,
            messages=messages,
            options=self._ollama_options(options),
            stream=False,
        )
        return response["message"]["content"]

    def stream_chat(self, messages: List[Dict], options: Dict) -> Iterator[str]:
        stream = self._ollama.chat(
            model=self.llm_model,
            messages=messages,
            options=self._ollama_options(options),
            stream=True,
        )
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                content = chunk["message"]["content"]
                if content:
                    yield content


# ---------------------------------------------------------------------------
# OpenAI — generation only; embeddings are delegated to Ollama
# ---------------------------------------------------------------------------

class OpenAIGenerationProvider(LLMProvider):
    """
    Uses OpenAI for chat/stream_chat.
    Embeddings always go through Ollama so ChromaDB index stays consistent.
    """

    def __init__(self, llm_model: str, embedding_model: str, api_key: str):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self.llm_model = llm_model
        self.embedding_model = embedding_model  # Ollama model, used by embed()

        # Internal Ollama instance for embeddings only
        import ollama as _ollama
        self._ollama = _ollama

    def embed(self, text: str) -> List[float]:
        """Always uses Ollama — keeps embeddings consistent with ChromaDB index."""
        response = self._ollama.embeddings(model=self.embedding_model, prompt=text)
        return response["embedding"]

    def chat(self, messages: List[Dict], options: Dict) -> str:
        response = self._client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            max_tokens=options.get("max_tokens", 512),
            temperature=options.get("temperature", 0.7),
            top_p=options.get("top_p", 0.9),
            stream=False,
        )
        return response.choices[0].message.content

    def stream_chat(self, messages: List[Dict], options: Dict) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            max_tokens=options.get("max_tokens", 512),
            temperature=options.get("temperature", 0.7),
            top_p=options.get("top_p", 0.9),
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider() -> LLMProvider:
    """
    Returns the configured provider.

    Embeddings always use Ollama (EMBEDDING_MODEL).
    Generation uses LLM_PROVIDER:
      - openai  → OpenAIGenerationProvider (OpenAI chat + Ollama embeddings, default)
      - ollama  → OllamaProvider           (local Ollama chat + embeddings)
    """
    provider_name   = (os.getenv("LLM_PROVIDER") or "openai").lower().strip()
    embedding_model = (os.getenv("EMBEDDING_MODEL") or "mxbai-embed-large").strip()

    if provider_name == "ollama":
        llm_model = (os.getenv("LLM_MODEL") or "llama3:8b-instruct-q4_0").strip()
        print(f"🤖 Generation: Ollama ({llm_model})  |  Embeddings: Ollama ({embedding_model})")
        return OllamaProvider(llm_model=llm_model, embedding_model=embedding_model)

    elif provider_name == "openai":
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai. "
                "Add it to your .env file (or set LLM_PROVIDER=ollama to run locally)."
            )
        llm_model = (os.getenv("OPENAI_LLM_MODEL") or "gpt-4o-mini").strip()
        print(f"🤖 Generation: OpenAI ({llm_model})  |  Embeddings: Ollama ({embedding_model})")
        return OpenAIGenerationProvider(
            llm_model=llm_model,
            embedding_model=embedding_model,
            api_key=api_key,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider_name!r}. Valid values: 'openai', 'ollama'."
        )
