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
    """Pinecone client, pointed at a custom CA bundle when one is configured.

    The SDK hands its own ``ssl_ca_cert`` to urllib3 and never consults
    ``SSL_CERT_FILE``/``REQUESTS_CA_BUNDLE``, so behind a TLS-inspecting proxy
    it keeps failing on the default certifi bundle even once the rest of the
    stack is happy. Passing the path explicitly is what makes it agree.
    """
    if settings.ca_bundle:
        return Pinecone(api_key=settings.pinecone_api_key, ssl_ca_certs=settings.ca_bundle)
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


def _build_store(settings: Settings, pc: Pinecone) -> PineconeVectorStore:
    """Wrap an index handle taken from ``pc`` in a LangChain vector store.

    The store is constructed around an existing index handle rather than via
    ``from_documents``/``from_existing_index``, because those build a Pinecone
    client of their own from the environment — which drops the custom CA
    bundle :func:`_get_pinecone` applies and fails behind a TLS-inspecting
    proxy. Reusing the handle keeps one client, and one TLS configuration.
    """
    os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key
    return PineconeVectorStore(
        index=pc.Index(settings.pinecone_index),
        embedding=build_embeddings(settings),
        namespace=settings.pinecone_namespace,
    )


def build_vector_store(chunks: list[Document], settings: Settings) -> PineconeVectorStore:
    """Upsert ``chunks`` into Pinecone under the configured namespace."""
    _ensure_index(settings)
    store = _build_store(settings, _get_pinecone(settings))
    store.add_documents(chunks)
    return store


def load_vector_store(settings: Settings) -> PineconeVectorStore:
    """Attach to the existing Pinecone index for retrieval."""
    pc = _get_pinecone(settings)
    existing = {idx["name"] for idx in pc.list_indexes()}
    if settings.pinecone_index not in existing:
        raise RuntimeError(
            f"Pinecone index {settings.pinecone_index!r} does not exist. "
            "Run `python ingest.py` first (it will create the index and upsert data)."
        )
    return _build_store(settings, pc)
