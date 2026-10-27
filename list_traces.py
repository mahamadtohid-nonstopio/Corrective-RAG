"""Lists recent LangSmith traces for the configured project.

Confirms tracing is actually reaching LangSmith (an invalid key fails silently
in the app — the SDK only logs a warning per call) and prints the UI link to
open the full waterfall.

Usage:
    python list_traces.py          # 20 most recent runs
    python list_traces.py 50       # 50 most recent runs
    python list_traces.py --llm    # only the LLM calls, with token counts
"""
from __future__ import annotations

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

# The Windows console is cp1252 and this script prints em dashes and run output
# that may contain clinical/unicode text.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LANGSMITH_UI_BASE = "https://smith.langchain.com"
DEFAULT_LIMIT = 20
# LangSmith's /runs/query rejects anything larger with a 400.
MAX_LIMIT = 100
REQUEST_TIMEOUT_SECONDS = 30


def _require_env(name: str, hint: str) -> str:
    value = (os.getenv(name) or "").strip().strip('"')
    if not value:
        raise SystemExit(f"{name} is not set in .env. {hint}")
    return value


def _api_base() -> str:
    return (
        os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com"
    ).strip().strip('"').rstrip("/")


def _get(path: str, api_key: str, **params) -> object:
    response = requests.get(
        f"{_api_base()}{path}",
        headers={"x-api-key": api_key},
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code == 403:
        raise SystemExit(
            "LangSmith rejected the API key (403). Generate a new one at "
            f"{LANGSMITH_UI_BASE}/settings and update LANGSMITH_API_KEY in .env."
        )
    response.raise_for_status()
    return response.json()


def _find_project(api_key: str, project_name: str) -> dict:
    sessions = _get("/sessions", api_key, name=project_name)
    if not sessions:
        raise SystemExit(
            f"No LangSmith project named {project_name!r} yet. Ask the app a "
            "question first — the project is created on the first traced run."
        )
    return sessions[0]


def _fetch_runs(
    api_key: str, project_id: str, limit: int, run_type: str | None = None
) -> list[dict]:
    """Recent runs, newest first — filtered server-side so ``limit`` means
    ``limit`` matching runs, not ``limit`` of every span including wrappers."""
    query = {"session": [project_id], "limit": limit, "order": "desc"}
    if run_type:
        query["run_type"] = run_type
    response = requests.post(
        f"{_api_base()}/runs/query",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json=query,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("runs", payload if isinstance(payload, list) else [])


def _token_summary(run: dict) -> str:
    prompt_tokens = run.get("prompt_tokens")
    completion_tokens = run.get("completion_tokens")
    if prompt_tokens is None and completion_tokens is None:
        return ""
    return f"  tokens {prompt_tokens or 0}→{completion_tokens or 0}"


def _print_run(run: dict) -> None:
    started = str(run.get("start_time") or "")[:19].replace("T", " ")
    status = "ERROR" if run.get("error") else "ok"
    name = str(run.get("name") or "?")[:38]
    print(
        f"  {started}  {str(run.get('run_type') or '?'):8s} "
        f"{name:38s} {status:5s}{_token_summary(run)}"
    )


def main() -> int:
    args = [a for a in sys.argv[1:]]
    llm_only = "--llm" in args
    limits = [a for a in args if a.isdigit()]
    limit = min(int(limits[0]) if limits else DEFAULT_LIMIT, MAX_LIMIT)

    api_key = _require_env(
        "LANGSMITH_API_KEY",
        f"Create one at {LANGSMITH_UI_BASE}/settings → API Keys.",
    )
    project_name = (os.getenv("LANGSMITH_PROJECT") or "default").strip().strip('"')

    if (os.getenv("LANGSMITH_TRACING") or "").strip().lower() != "true":
        print("Warning: LANGSMITH_TRACING is not 'true' — new runs are not being traced.\n")

    project = _find_project(api_key, project_name)
    project_id = project["id"]
    runs = _fetch_runs(api_key, project_id, limit, run_type="llm" if llm_only else None)

    print(f"Project {project_name!r} — {len(runs)} most recent runs:\n")
    for run in runs:
        _print_run(run)
    if not runs:
        print("  (none yet — ask the app a question, then re-run this)")

    tenant_id = project.get("tenant_id")
    org_path = f"/o/{tenant_id}" if tenant_id else ""
    print(f"\nFull waterfall: {LANGSMITH_UI_BASE}{org_path}/projects/p/{project_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
