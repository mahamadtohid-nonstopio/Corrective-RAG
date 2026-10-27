"""Pinecone-backed vector store using NVIDIA-hosted asymmetric embeddings.

The embedding model (``nvidia/nv-embedqa-e5-v5``) is asymmetric, so we use
``langchain-nvidia-ai-endpoints``' ``NVIDIAEmbeddings``, which handles the
``input_type=passage`` (indexing) vs ``input_type=query`` (retrieval) split
automatically.

If the configured Pinecone index does not yet exist, we create it as a
serverless index in the configured cloud/region, matching the embedding
model's dimension (``Settings.embedding_dimension``) and cosine similarity.
Swapping ``EMBEDDING_MODEL`` almost always means the vector width changes
too (e.g. nv-embedqa-e5-v5 → 1024, nemotron-3.5-embed-1b → 2048), so the
index name and ``EMBEDDING_DIMENSION`` must be updated together — an
existing index can't be resized in place, and vectors from one embedding
model are meaningless to another.
"""
from __future__ import annotations

import logging
import os

from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from .config import Settings

logger = logging.getLogger(__name__)


def build_embeddings(settings: Settings) -> NVIDIAEmbeddings:
    return NVIDIAEmbeddings(
        model=settings.embedding_model,
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
        truncate="END",
    )


def _get_pinecone(settings: Settings) -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def _ensure_index(settings: Settings) -> None:
    pc = _get_pinecone(settings)
    existing = {idx["name"] for idx in pc.list_indexes()}
    if settings.pinecone_index in existing:
        return
    logger.info(
        "Creating Pinecone index %r (dim=%d, cloud=%s, region=%s) …",
        settings.pinecone_index,
        settings.embedding_dimension,
        settings.pinecone_cloud,
        settings.pinecone_region,
    )
    pc.create_index(
        name=settings.pinecone_index,
        dimension=settings.embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        ),
    )


def build_vector_store(chunks: list[Document], settings: Settings) -> PineconeVectorStore:
    """Upsert ``chunks`` into Pinecone under the configured namespace.

    ``langchain-pinecone`` reads ``PINECONE_API_KEY`` from the process
    environment (already populated by ``load_dotenv()`` in :mod:`config`).
    """
    _ensure_index(settings)
    os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key
    embeddings = build_embeddings(settings)
    return PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=settings.pinecone_index,
        namespace=settings.pinecone_namespace,
    )


def load_vector_store(settings: Settings) -> PineconeVectorStore:
    """Attach to the existing Pinecone index for retrieval."""
    pc = _get_pinecone(settings)
    existing = {idx["name"] for idx in pc.list_indexes()}
    if settings.pinecone_index not in existing:
        raise RuntimeError(
            f"Pinecone index {settings.pinecone_index!r} does not exist. "
            "Run `python ingest.py` first (it will create the index and upsert data)."
        )
    os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key
    embeddings = build_embeddings(settings)
    return PineconeVectorStore.from_existing_index(
        index_name=settings.pinecone_index,
        embedding=embeddings,
        namespace=settings.pinecone_namespace,
    )
