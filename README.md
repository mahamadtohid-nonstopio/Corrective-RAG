# Healthcare Knowledge Navigator

An evidence-based medical assistant that retrieves and synthesises information
from clinical guidelines, research papers, and treatment protocols. Ships **two**
LangGraph pipelines side-by-side:

1. **RAG** — classical retrieve → generate → confidence-score.
2. **CRAG (Corrective RAG)** — retrieve → grade each passage → optionally
   rewrite the query and pull the web via Tavily → generate → confidence-score.

Backend: **FastAPI + Uvicorn**.
Frontend: plain **HTML / CSS / JavaScript** (no build step).
LLM + embeddings: **NVIDIA Free Endpoints** via the OpenAI-compatible API
(`deepseek-ai/deepseek-v4-pro` chat + `nvidia/nv-embedqa-e5-v5` embeddings by
default). Any model on `integrate.api.nvidia.com` can be dropped in.
Retrieval: **Pinecone** (serverless). Orchestration: **LangGraph**.

> Educational reference only. Not a substitute for professional clinical
> judgement. Verify against primary sources before making patient-care decisions.

---

## 1. Setup

```powershell
# 1. Create + activate a Python 3.10+ venv
python -m venv .venv
.\.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
copy .env.example .env
#   then open .env and set:
#     NVIDIA_API_KEY=nvapi-…       (required — https://build.nvidia.com/)
#     PINECONE_API_KEY=pcsk-…      (required — https://app.pinecone.io/)
#     PINECONE_INDEX=lightning-talk    (must match your Pinecone index name)
#     PINECONE_CLOUD=aws           PINECONE_REGION=us-east-1
#     PINECONE_NAMESPACE=healthcare-nav
#     TAVILY_API_KEY=…             (optional — enables CRAG web fallback)
#     LANGSMITH_API_KEY=lsv2_pt_…  (optional — hosted traces at https://smith.langchain.com/)
#     LANGSMITH_PROJECT=healthcare-nav
```

## 2. Ingest documents

Drop your `.md`, `.txt`, or `.pdf` clinical documents into `./data/` (a starter
corpus of ADA T2DM, GINA asthma, hypertension, sepsis, and AF anticoagulation is
already included). Then build the FAISS index:

```powershell
python ingest.py
```

## 3. Run the web app

```powershell
uvicorn server:app --reload
```

Open <http://127.0.0.1:8000> in your browser. Pick **Compare RAG vs CRAG**,
**RAG only**, or **CRAG only**, type a clinical question, and hit **Ask**.

Each answer shows:

- The synthesised answer with inline `[1] [2]` citations.
- A colour-coded confidence label + score (LLM-audited grounding).
- Expandable citation snippets.
- (CRAG) the graph trace of routing decisions.

---

## Architecture

```
data/*.md,*.pdf ──► ingest.py ──► FAISS (Gemini embeddings)
                                       │
                                       ▼
        ┌──────────────────────────────────────────────┐
        │              LangGraph agents                │
        │                                              │
        │  RAG:   retrieve ─► generate ─► score        │
        │                                              │
        │  CRAG:  retrieve ─► grade  ─► decide         │
        │           ├─ generate                        │
        │           └─ rewrite ─► web_search ─► generate│
        │                            ─► score          │
        └──────────────────────────────────────────────┘
                             │
                             ▼
                 FastAPI (/api/ask)  ◄──►  HTML/CSS/JS frontend
```

### API

| Method | Path          | Description                                      |
|--------|---------------|--------------------------------------------------|
| GET    | `/`           | Serves the HTML UI                               |
| GET    | `/api/health` | Status + active model / top-k / web-fallback flag |
| POST   | `/api/ask`    | `{ "question": "...", "mode": "rag"\|"crag" }` |

`POST /api/ask` returns:

```jsonc
{
  "mode": "crag",
  "question": "...",
  "answer": "text with inline [1] citations …",
  "citations": [{ "id": 1, "source": "...", "page": null, "snippet": "..." }],
  "confidence": 0.83,
  "confidence_label": "HIGH",
  "trace": ["retrieved 5 passages", "graded relevance: 4/5 relevant", "decision: generate"],
  "disclaimer": "Educational reference only. …"
}
```

Interactive API docs (Swagger) are served at `/docs`.

### Project layout

| File / dir                | Purpose |
|---------------------------|---------|
| `server.py`               | FastAPI app: routes, pipeline lifecycle |
| `static/index.html`       | Single-page UI |
| `static/styles.css`       | Styling |
| `static/app.js`           | Fetches `/api/ask`, renders answers + trace |
| `ingest.py`               | Builds the FAISS vector store |
| `src/config.py`           | Env-driven `Settings` (fails fast if key missing) |
| `src/document_loader.py`  | Recursive load + chunk of md/txt/pdf |
| `src/vector_store.py`     | FAISS index build / load (Gemini embeddings) |
| `src/rag_agent.py`        | Classical RAG LangGraph |
| `src/crag_agent.py`       | Corrective RAG LangGraph (grading + web fallback) |
| `src/utils.py`            | LLM factory, citation + confidence helpers |
| `data/`                   | Source documents (starter corpus included) |

### Confidence scoring

After each answer is generated, a second Gemini call audits how well the answer
is grounded in the retrieved context and returns a score in `[0, 1]`.

- HIGH ≥ 0.80
- MODERATE ≥ 0.55
- LOW ≥ 0.30
- VERY LOW < 0.30

CRAG additionally caps confidence at 0.6 when the answer was produced entirely
from web-search fallback, since those sources aren't part of the vetted corpus.

### CRAG routing

A per-passage LLM grader labels each retrieved chunk `yes` / `no`. The router
picks one of three actions based on the fraction of relevant passages:

| Ratio of relevant docs | Action |
|------------------------|--------|
| = 0                    | rewrite query and answer purely from web results |
| < 0.5                  | supplement local docs with web results |
| ≥ 0.5                  | answer directly from local docs |

## Extending the corpus

- Drop new files under `./data/` (any nested directory is fine).
- Re-run `python ingest.py`. The old index is overwritten.
- PDFs are page-aware — page numbers appear in citations.
