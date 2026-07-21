"""Inspect what's actually stored in Pinecone.

Prints:
  - namespace stats (total vector count),
  - one sample record with the vector length + first 8 values + metadata.

Usage:
    python verify_pinecone.py
"""
from __future__ import annotations

from pinecone import Pinecone

from src.config import get_settings


def main() -> int:
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index)

    stats = index.describe_index_stats()
    ns = stats.get("namespaces", {}).get(settings.pinecone_namespace, {})
    total = ns.get("vector_count", 0)
    print(f"Index: {settings.pinecone_index}")
    print(f"Namespace: {settings.pinecone_namespace}")
    print(f"Total vectors stored: {total}")
    print(f"Index dimension: {stats.get('dimension')}")
    print()

    if total == 0:
        print("No records in this namespace. Run `python ingest.py` first.")
        return 1

    # Pull one ID via a random query, then fetch that record with vector values.
    dummy_query = [0.0] * stats["dimension"]
    hits = index.query(
        vector=dummy_query,
        top_k=1,
        namespace=settings.pinecone_namespace,
        include_values=True,
        include_metadata=True,
    )
    if not hits.matches:
        print("No matches returned — namespace may be empty.")
        return 1

    match = hits.matches[0]
    values = list(match.values or [])
    print(f"Sample record id: {match.id}")
    print(f"  vector length: {len(values)}  (should equal index dimension)")
    print(f"  first 8 values: {values[:8]}")
    print(f"  metadata keys: {list((match.metadata or {}).keys())}")
    snippet = (match.metadata or {}).get("text", "")
    print(f"  text snippet: {snippet[:180]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
