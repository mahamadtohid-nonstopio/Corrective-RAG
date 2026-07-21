"""Load and chunk medical knowledge documents.

Supports .md, .txt, and .pdf files placed under the configured DATA_DIR.
Each chunk carries source metadata used later for citations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def _load_one(path: Path) -> list[Document]:
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault("source", str(path))
        d.metadata["source_name"] = path.name
    return docs


def load_documents(settings: Settings) -> list[Document]:
    if not settings.data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {settings.data_dir}. "
            "Add clinical guideline / research documents there."
        )
    documents: list[Document] = []
    for file_path in _iter_files(settings.data_dir):
        documents.extend(_load_one(file_path))
    if not documents:
        raise RuntimeError(
            f"No supported documents found under {settings.data_dir}. "
            "Add .md, .txt, or .pdf files."
        )
    return documents


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
    return chunks
