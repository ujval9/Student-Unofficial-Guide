"""
embed.py — Stage 3 + 4 of the RAG pipeline: Embedding -> Vector Store.

Loads the chunks produced by ingest.py (chunks.json), embeds each one with the
all-MiniLM-L6-v2 sentence-transformer (runs locally, no API key, no rate limits),
and stores the vectors in a persistent ChromaDB collection together with the
metadata we need for attribution at answer time.

Metadata stored per chunk:
    source       — human-readable source document name (for citing answers)
    slug         — the document's file stem
    url          — original URL of the source
    chunk_index  — the chunk's position within its document

Run once after ingestion (re-run to rebuild after re-chunking):
    .venv/bin/python embed.py
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("chunks.json")
CHROMA_PATH = "chroma_db"               # persistent on-disk store (gitignored)
COLLECTION_NAME = "ub_housing"
EMBED_MODEL = "all-MiniLM-L6-v2"        # 384-dim, local; see planning.md


def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        raise SystemExit("chunks.json not found — run `python ingest.py` first.")
    return json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))


def build_index() -> None:
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    print(f"Loading embedding model: {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in chunks]
    # normalize_embeddings=True pairs with the collection's cosine space below,
    # so distances come back as cosine distance (0 = identical, ~2 = opposite).
    print("Embedding chunks ...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Rebuild from scratch so re-runs never mix stale vectors with new ones.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{
            "source": c["source"],
            "slug": c["slug"],
            "url": c["url"],
            "chunk_index": c["chunk_index"],
        } for c in chunks],
    )

    print(f"\nStored {collection.count()} chunks in ChromaDB collection "
          f"'{COLLECTION_NAME}' at ./{CHROMA_PATH}")


if __name__ == "__main__":
    build_index()
