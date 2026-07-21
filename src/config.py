"""Central configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    nvidia_api_key: str
    nvidia_base_url: str
    tavily_api_key: str | None
    chat_model: str
    embedding_model: str
    pinecone_api_key: str
    pinecone_index: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_namespace: str
    data_dir: Path
    top_k: int
    chunk_size: int
    chunk_overlap: int


def _require(name: str, hint: str) -> str:
    value = os.getenv(name, "").strip().strip('"').strip("'")
    if not value or value.startswith("your_") or "your_" in value.lower():
        raise RuntimeError(
            f"Missing required environment variable {name}. {hint}"
        )
    return value


def get_settings() -> Settings:
    return Settings(
        nvidia_api_key=_require(
            "NVIDIA_API_KEY",
            "Get one at https://build.nvidia.com/ and set it in .env.",
        ),
        nvidia_base_url=(os.getenv("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1").strip().strip('"').rstrip("/"),
        tavily_api_key=(os.getenv("TAVILY_API_KEY") or "").strip().strip('"') or None,
        chat_model=os.getenv("CHAT_MODEL", "deepseek-ai/deepseek-v4-pro").strip().strip('"'),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5").strip().strip('"'),
        pinecone_api_key=_require(
            "PINECONE_API_KEY",
            "Get one at https://app.pinecone.io/ and set it in .env.",
        ),
        pinecone_index=os.getenv("PINECONE_INDEX", "lightning-talk").strip().strip('"'),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws").strip().strip('"'),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1").strip().strip('"'),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "healthcare-nav").strip().strip('"'),
        data_dir=(ROOT_DIR / os.getenv("DATA_DIR", "./data")).resolve(),
        top_k=int(os.getenv("TOP_K", "5")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
    )
