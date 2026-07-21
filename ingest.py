"""Ingest documents under DATA_DIR into Pinecone.

Usage:
    python ingest.py
"""
from __future__ import annotations

import sys

from src.config import get_settings
from src.document_loader import load_documents, split_documents
from src.vector_store import build_vector_store


def main() -> int:
    settings = get_settings()
    print(f"[ingest] Loading documents from {settings.data_dir} ...")
    documents = load_documents(settings)
    print(f"[ingest] Loaded {len(documents)} raw documents.")

    print(
        f"[ingest] Splitting into chunks "
        f"(size={settings.chunk_size}, overlap={settings.chunk_overlap}) ..."
    )
    chunks = split_documents(documents, settings)
    print(f"[ingest] Produced {len(chunks)} chunks.")

    print(
        f"[ingest] Embedding with {settings.embedding_model} and "
        f"upserting to Pinecone index {settings.pinecone_index!r} "
        f"namespace {settings.pinecone_namespace!r} ..."
    )
    build_vector_store(chunks, settings)
    print("[ingest] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
