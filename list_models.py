"""Lists chat + embedding models available on your NVIDIA endpoint.

Usage:
    python list_models.py
"""
from __future__ import annotations

from openai import OpenAI

from src.config import get_settings


def main() -> int:
    settings = get_settings()
    client = OpenAI(base_url=settings.nvidia_base_url, api_key=settings.nvidia_api_key)
    print(f"Models available at {settings.nvidia_base_url}:\n")
    names: list[str] = []
    for m in client.models.list():
        names.append(m.id)
        print(f"  {m.id}")

    embeds = [n for n in names if "embed" in n.lower()]
    print("\nEmbedding-looking models:")
    for n in embeds:
        print(f"  - {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
