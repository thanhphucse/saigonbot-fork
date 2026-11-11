"""
Tenant configuration.

The same codebase serves two assistants:

    TENANT=saigon  -> Saigon Indian Restaurant host  (chroma_db / hotel_saigon_menu)
    TENANT=chikku  -> Chikku Robotics support agent  (chikku_db / chikku_kb)

Everything that differs between them — vector store, system prompt, persona
name, retrieval filters, validator rule set — lives here rather than being
hardcoded across the pipeline.

`domain` is the behavioural switch:
    "restaurant" -> menu RAG path (dietary filters, dish extraction, ordering)
    "company"    -> generic company-support RAG path (src/company_rag.py)

Individual fields can still be overridden by env vars (CHROMA_DIR,
COLLECTION_NAME, SYSTEM_PROMPT_FILE) so a tenant can be pointed at a different
store without editing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class TenantConfig:
    name: str
    display_name: str
    domain: str                      # "restaurant" | "company"
    chroma_dir: str
    collection_name: str
    system_prompt_file: str
    assistant_name: str = "Chikku"
    api_title: str = "Chatbot API"
    # Terms the response validator must NOT strip for this tenant.
    validator_allow: tuple = field(default_factory=tuple)

    @property
    def is_restaurant(self) -> bool:
        return self.domain == "restaurant"

    @property
    def is_company(self) -> bool:
        return self.domain == "company"

    def resolve_prompt_path(self) -> Path:
        p = Path(self.system_prompt_file)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def resolve_chroma_dir(self) -> Path:
        p = Path(self.chroma_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p


TENANTS: Dict[str, TenantConfig] = {
    "saigon": TenantConfig(
        name="saigon",
        display_name="Saigon Indian Restaurant",
        domain="restaurant",
        chroma_dir="chroma_db",
        collection_name="hotel_saigon_menu",
        system_prompt_file="data/system_prompt.txt",
        assistant_name="Chikku",
        api_title="Hotel Saigon Chatbot API",
    ),
    "chikku": TenantConfig(
        name="chikku",
        display_name="Chikku Robotics",
        domain="company",
        chroma_dir="chikku_db",
        collection_name="chikku_kb",
        system_prompt_file="data/chikku_system_prompt.txt",
        assistant_name="Chikku",
        api_title="Chikku Robotics Support API",
        # Chikku Robotics is an AI/robotics company: these terms are legitimate
        # product vocabulary and must survive response cleaning.
        validator_allow=(
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "neural network",
            "ai model",
            "language model",
            "training data",
            "computer vision",
            "generative ai",
        ),
    ),
}

_tenant: Optional[TenantConfig] = None


def get_tenant() -> TenantConfig:
    """Return the active tenant, applying any env-var overrides."""
    global _tenant
    if _tenant is not None:
        return _tenant

    name = (os.getenv("TENANT") or "chikku").strip().lower()
    if name not in TENANTS:
        raise SystemExit(
            f"Unknown TENANT={name!r}. Valid values: {', '.join(sorted(TENANTS))}"
        )

    cfg = TENANTS[name]

    # Per-field env overrides win over the tenant defaults.
    cfg.chroma_dir = os.getenv("CHROMA_DIR", cfg.chroma_dir)
    cfg.collection_name = os.getenv("COLLECTION_NAME", cfg.collection_name)
    cfg.system_prompt_file = os.getenv("SYSTEM_PROMPT_FILE", cfg.system_prompt_file)

    _tenant = cfg
    return cfg


def reset_tenant() -> None:
    """Clear the cached tenant — used by tests that switch TENANT at runtime."""
    global _tenant
    _tenant = None
