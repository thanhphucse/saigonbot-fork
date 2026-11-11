"""
Ingestion Pipeline: Load menu data and index into ChromaDB with embeddings
"""
import json
import sys
import os
import chromadb
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv

# Ensure project src is on path when run as standalone script
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from llm_provider import get_provider


# Module-level provider (lazy-initialized)
_provider = None

def _get_provider():
    global _provider
    if _provider is None:
        _provider = get_provider()
    return _provider


def load_menu_items(json_path: str) -> List[Dict]:
    """Load menu items from JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    print(f"📦 Loaded {len(items)} menu items from {json_path}")
    return items


def create_text_chunk(item: Dict) -> str:
    """Create searchable text chunk from menu item"""
    parts = []

    # Item name (most important)
    parts.append(item['item_name'])

    # Section
    if item.get('section') and item['section'] != 'UNKNOWN':
        parts.append(f"Section: {item['section']}")

    # Price
    if item.get('price'):
        parts.append(f"Price: {item['price']:,.0f} {item.get('currency', 'VND')}")

    # Tags
    if item.get('tags') and item['tags'] != ['unknown']:
        tags_str = ', '.join(item['tags'])
        parts.append(f"Tags: {tags_str}")

    return '. '.join(parts)


def get_embedding(text: str, model: str = None) -> List[float]:
    """Get embedding via configured provider (Ollama or OpenAI)."""
    try:
        return _get_provider().embed(text)
    except Exception as e:
        print(f"⚠️  Error getting embedding: {e}")
        raise


def setup_chromadb(persist_directory: str = "chroma_db"):
    """Setup ChromaDB collection"""
    client = chromadb.PersistentClient(path=persist_directory)

    collection_name = "hotel_saigon_menu"
    try:
        collection = client.get_collection(name=collection_name)
        print(f"📂 Using existing collection: {collection_name}")
        print(f"   Current items: {collection.count()}")
    except Exception:
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Hotel Saigon menu items with embeddings"}
        )
        print(f"✅ Created new collection: {collection_name}")

    return collection


def batch_embeddings(texts: List[str], model: str = None, batch_size: int = 10):
    """Generate embeddings in batches via configured provider."""
    provider = _get_provider()
    provider_name = os.getenv("LLM_PROVIDER", "openai").lower()
    print(f"🔄 Generating embeddings via Ollama ({provider.embedding_model}, active LLM_PROVIDER={provider_name})...")

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i + batch_size]
        batch_embeds = []

        for text in batch:
            try:
                embedding = provider.embed(text)
                batch_embeds.append(embedding)
            except Exception as e:
                print(f"⚠️  Error embedding text: {text[:50]}... - {e}")
                # Fallback zero vector — dimension from first successful embed or default
                dim = len(embeddings[0]) if embeddings else (len(batch_embeds[0]) if batch_embeds else 1024)
                batch_embeds.append([0.0] * dim)

        embeddings.extend(batch_embeds)

    return embeddings


def index_menu_items(collection, items: List[Dict], model: str = None):
    """Index menu items into ChromaDB"""
    print(f"\n📝 Indexing {len(items)} menu items...")

    ids = []
    texts = []
    metadatas = []

    for item in items:
        ids.append(item['id'])

        text_chunk = create_text_chunk(item)
        texts.append(text_chunk)

        metadata = {
            'item_number': str(item['item_number']),
            'item_name':   item['item_name'],
            'section':     item.get('section', 'UNKNOWN'),
            'price':       str(item.get('price', '')) if item.get('price') else '',
            'currency':    item.get('currency', '') or '',
            'tags':        ', '.join(item.get('tags', [])),
            'source':      item.get('source', 'saigon.pdf'),
        }
        metadatas.append(metadata)

    # Generate embeddings
    embeddings = batch_embeddings(texts)

    # Add to ChromaDB
    print(f"\n💾 Storing items in ChromaDB...")
    try:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print(f"✅ Successfully indexed {len(items)} items")
    except Exception as e:
        print(f"❌ Error indexing: {e}")
        # Try updating existing items
        try:
            collection.update(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            print(f"✅ Updated {len(items)} items")
        except Exception as e2:
            print(f"❌ Update also failed: {e2}")
            raise


def index_about_knowledge(collection, about_path: str, model: str = None):
    """
    Index about.txt content into ChromaDB.
    Documents get ids like 'about:0', 'about:1', ... and doc_type='about'.
    Safe to re-run — existing about docs are replaced via upsert.
    """
    provider = _get_provider()
    text = Path(about_path).read_text(encoding="utf-8").strip()

    # Split into paragraphs (blank-line separated), skip empty ones
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        print("⚠️  about.txt is empty — nothing to index")
        return

    print(f"\n📝 Indexing {len(paragraphs)} about paragraphs from {about_path}...")

    ids        = [f"about:{i}" for i in range(len(paragraphs))]
    embeddings = []
    metadatas  = [{"doc_type": "about", "source": "about.txt"} for _ in paragraphs]

    for para in tqdm(paragraphs, desc="Embedding about chunks"):
        try:
            embeddings.append(provider.embed(para))
        except Exception as e:
            print(f"⚠️  Embed error: {e}")
            dim = len(embeddings[0]) if embeddings else 1024
            embeddings.append([0.0] * dim)

    # Delete old about docs then re-add (upsert not available in all chroma versions)
    try:
        existing = collection.get(ids=ids)
        existing_ids = [i for i in existing["ids"] if i]
        if existing_ids:
            collection.delete(ids=existing_ids)
            print(f"   🗑️  Removed {len(existing_ids)} old about docs")
    except Exception:
        pass

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=paragraphs,
        metadatas=metadatas
    )
    print(f"✅ Indexed {len(paragraphs)} about paragraphs")


def validate_indexing(collection, test_queries: List[str] = None):
    """Validate indexing with test queries"""
    if test_queries is None:
        test_queries = [
            "vegetarian dosa",
            "chicken curry",
            "beverages under 50000",
            "breakfast items",
            "mutton dishes"
        ]

    print(f"\n🔍 Validating indexing with test queries...")
    provider = _get_provider()

    for query in test_queries:
        print(f"\n   Query: '{query}'")
        try:
            query_embedding = provider.embed(query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )

            if results['ids'] and len(results['ids'][0]) > 0:
                print(f"   Top results:")
                for i, (item_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                    metadata = results['metadatas'][0][i]
                    print(f"     {i+1}. {metadata.get('item_name', item_id)} ({metadata.get('section', '')}) - Distance: {distance:.4f}")
            else:
                print(f"   ⚠️  No results found")
        except Exception as e:
            print(f"   ❌ Error: {e}")


def main():
    """Main ingestion function"""
    project_root = Path(__file__).parent.parent
    menu_json  = project_root / "data" / "processed" / "menu_items.json"
    chroma_dir = project_root / "chroma_db"

    # Show active provider
    provider = _get_provider()
    print(f"🤖 Using provider: {os.getenv('LLM_PROVIDER', 'openai')} | embedding model: {provider.embedding_model}")

    # Load menu items
    items = load_menu_items(str(menu_json))

    # Setup ChromaDB
    collection = setup_chromadb(str(chroma_dir))

    # Check if already indexed
    if collection.count() > 0:
        print(f"\n⚠️  Collection already has {collection.count()} items")
        response = input("   Re-index? This will replace existing data (y/n): ").strip().lower()
        if response == 'y':
            client = chromadb.PersistentClient(path=str(chroma_dir))
            client.delete_collection("hotel_saigon_menu")
            collection = client.create_collection(
                name="hotel_saigon_menu",
                metadata={"description": "Hotel Saigon menu items with embeddings"}
            )
            print("   ✅ Cleared existing collection")
        else:
            print("   ℹ️  Skipping indexing")
            validate_indexing(collection)
            return

    # Index items
    index_menu_items(collection, items)

    # Validate
    validate_indexing(collection)

    print(f"\n✅ Ingestion complete!")
    print(f"   Collection: hotel_saigon_menu")
    print(f"   Items indexed: {collection.count()}")
    print(f"   Storage: {chroma_dir}")


if __name__ == "__main__":
    main()
