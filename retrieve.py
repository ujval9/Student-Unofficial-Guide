"""
retrieve.py — Stage 5 of the RAG pipeline: Retrieval.

Embeds a query with the same all-MiniLM-L6-v2 model used to build the index and
returns the top-k most similar chunks from ChromaDB, each with its source
metadata and cosine distance (0 = identical, higher = less related).

As a library:
    from retrieve import retrieve
    hits = retrieve("housing without a car", k=5)

As a script (runs the planning.md evaluation queries and prints results):
    .venv/bin/python retrieve.py
"""

from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from embed import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL

DEFAULT_K = 5  # start small; tune after seeing real results (see planning.md)

# The five evaluation queries from planning.md -> Evaluation Plan.
EVAL_QUERIES = [
    "As a student without a car, which neighborhoods near UB are best for off-campus housing?",
    "What do students think about popular housing complexes near North Campus?",
    "Is a budget of $750 per month enough for off-campus housing near UB?",
    "What should I check before signing an off-campus lease?",
    "How do students commute to North Campus without a car?",
]


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        raise SystemExit(f"Collection '{COLLECTION_NAME}' not found — run `python embed.py` first.")


def retrieve(query: str, k: int = DEFAULT_K) -> list[dict]:
    """Return the top-k chunks most relevant to `query`, ranked by cosine distance."""
    q_emb = _model().encode([query], normalize_embeddings=True).tolist()
    res = _collection().query(query_embeddings=q_emb, n_results=k)
    hits = []
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({
            "id": cid,
            "text": doc,
            "source": meta["source"],
            "url": meta["url"],
            "chunk_index": meta["chunk_index"],
            "distance": dist,
        })
    return hits


def _print_results(query: str, k: int = DEFAULT_K) -> None:
    print("\n" + "=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)
    for rank, hit in enumerate(retrieve(query, k), 1):
        snippet = hit["text"].replace("\n", " ").strip()
        if len(snippet) > 320:
            snippet = snippet[:320] + "..."
        print(f"\n[{rank}] distance={hit['distance']:.3f}  source={hit['source']}  "
              f"(chunk #{hit['chunk_index']})")
        print(f"    {snippet}")


if __name__ == "__main__":
    for q in EVAL_QUERIES:
        _print_results(q)
