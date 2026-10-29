# Pipeline flow and LLM call accounting

Exactly what runs, in what order, and where each LLM call is spent — for both
pipelines. Line references point at the real implementations:
[`src/rag_agent.py`](src/rag_agent.py), [`src/crag_agent.py`](src/crag_agent.py),
[`src/utils.py`](src/utils.py).

> These are **not** LangGraph `StateGraph`s. Both pipelines are hand-written
> async generators composed from LangChain runnables (`chain = PROMPT | llm`).
> They stream cleanly; they have no checkpointing, replay or resumability.

---

## 1. The short answer

Four things can cost an LLM call. Only `generate` always runs.

| Step | Model call | When it runs |
|---|---|---|
| **retrieve** | embedding (not a chat call) | Always — every query embeds the question |
| **classify_domain** | chat, structured | Only when top cosine < `TOPIC_SCORE_THRESHOLD` |
| **grade_passages** | chat, structured | CRAG only, and only when retrieval was *good* |
| **rewrite_query** | chat, structured | CRAG only, and only when correcting |
| **web_search** | *not an LLM* — Tavily HTTP | CRAG only, after a rewrite |
| **generate** | chat, **streamed** | Always, unless the query was refused |

Two things that commonly surprise people:

- **Grading is one call, not one per passage.** All `TOP_K` passages go into a
  single prompt and come back as a `GradeBatch` ([`_grade_docs`](src/crag_agent.py#L296)).
- **Grading and classification are mutually exclusive.** The gate decides which
  branch you are on, so CRAG never pays for both.

---

## 2. RAG

```
question
   │
   ▼
┌──────────────┐
│ 1. retrieve  │  embedding call → Pinecone top-k          rag_agent.py:105
└──────┬───────┘
       │
       ├── 0 passages ──────────────► "no context" canned reply ──► END   (0 chat calls)
       │
       ▼
  top cosine ≥ TOPIC_SCORE_THRESHOLD ?
       │
       ├── YES ───────────────────────────────────────┐
       │                                              │
       └── NO                                         │
            │                                         │
            ▼                                         │
     ┌──────────────────┐                             │
     │ classify_domain  │  CHAT CALL #1               │   rag_agent.py:141
     └────────┬─────────┘                             │
              │                                       │
              ├── not medical ──► refuse ──► END      │
              │                                       │
              └── medical ───────────────────────────┤
                   (answer from weak context —        │
                    RAG has no correction step)       │
                                                      ▼
                                             ┌──────────────┐
                                             │ 3. generate  │  CHAT CALL (streamed)
                                             └──────┬───────┘     rag_agent.py:157
                                                    ▼
                                              answer + meta
```

### RAG cost

| Path | Chat calls | Example |
|---|---|---|
| Retrieval empty | **0** | index empty |
| On topic | **1** | T2DM + CKD question |
| Weak retrieval, not medical | **1** | sourdough recipe |
| Weak retrieval, medical | **2** | "what is hereditary cancer" |

The on-topic path staying at **one call** is deliberate — it is the honest
baseline CRAG is measured against, so the gate must not tax it.

---

## 3. CRAG

```
question
   │
   ▼
┌──────────────┐
│ 1. retrieve  │  embedding call → Pinecone top-k        crag_agent.py:376
└──────┬───────┘
       │
       ├── 0 passages ──────────────► "no context" ──► END          (0 chat calls)
       │
       ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 2. TOPIC GATE   top cosine < TOPIC_SCORE_THRESHOLD ?         │  crag_agent.py:403
  └───────────────┬──────────────────────────┬──────────────────┘
                  │ BELOW                    │ AT OR ABOVE
                  ▼                          ▼
        ┌──────────────────┐        ┌──────────────────┐
        │ classify_domain  │ CALL   │ 3. grade_passages│ CALL      crag_agent.py:439
        └────────┬─────────┘        └────────┬─────────┘
                 │                           │  one batched call,
                 │                           │  all top-k passages
     ┌───────────┼───────────┐               ▼
     │           │           │     ┌──────────────────┐
  not medical  medical    medical  │ 4. decide        │  no LLM — pure ratio
     │        + no Tavily  + Tavily└────────┬─────────┘           crag_agent.py:450
     ▼           ▼           │              │
  refuse      refuse         │   relevant ratio → action
   END         END           │     ├─ 0      → web_only
                             │     ├─ < 0.5  → supplement
                             │     └─ ≥ 0.5  → generate
                             │              │
                             └──────────────┤
                                            ▼
                               action == "generate" ?
                                            │
                        ┌── NO ─────────────┴──── YES ──┐
                        ▼                               │
          ┌───────────────────────────┐                 │
          │ 5a. rewrite_query  CALL   │                 │  crag_agent.py:454
          │     (prose guard: falls   │                 │
          │      back to original)    │                 │
          └─────────────┬─────────────┘                 │
                        ▼                               │
          ┌───────────────────────────┐                 │
          │ 5b. web_search            │                 │
          │     Tavily — NOT an LLM   │                 │
          └─────────────┬─────────────┘                 │
                        │                               │
                        └───────────────┬───────────────┘
                                        ▼
                         context = kept local + web docs
                                        │
                        ├── empty ──► "no context" ──► END
                                        ▼
                             ┌──────────────┐
                             │ 6. generate  │  CHAT CALL (streamed)
                             └──────┬───────┘        crag_agent.py:474
                                    ▼
                             7. meta + confidence     crag_agent.py:495
```

### CRAG cost

| Path | Chat calls | Tavily | Example |
|---|---|---|---|
| Retrieval empty | **0** | – | index empty |
| Below gate, not medical | **1** (classify) | – | sourdough recipe |
| Below gate, medical, no Tavily | **1** (classify) | – | degraded config |
| Below gate, medical, Tavily | **3** (classify + rewrite + generate) | 1 | hereditary cancer |
| Above gate, `generate` | **2** (grade + generate) | – | well-covered question |
| Above gate, `supplement` / `web_only` | **3** (grade + rewrite + generate) | 1 | apixaban GI bleed |

---

## 4. Why the gate is two-stage

A single cosine threshold cannot separate *"not medicine"* from *"medicine my
corpus doesn't cover"*, and those need opposite handling. Measured against this
repo's five-document corpus:

| Question | Top cosine | Correct handling |
|---|---|---|
| sourdough bread recipe | **0.130** | refuse |
| "what is hereditary cancer" | **0.033** | **web-correct** |

Cosine ranks the bread recipe as four times more on-topic than the cancer
question, because the corpus contains no oncology at all while the recipe shares
procedural vocabulary with the sepsis protocol. So a weak score triggers one
cheap domain-classification call ([`is_medical_question`](src/utils.py#L144))
rather than an automatic refusal.

The classifier **fails closed**: if that call errors, the question is refused
rather than answered on a guess. Passage grading, by contrast, **fails open** —
an infrastructure failure there keeps all passages instead of silently
discarding evidence.

---

## 5. Where confidence comes from

Not an LLM self-audit — an earlier version asked the model to grade its own
grounding and it was not a reliable judge.

- **Corpus-grounded answers:** `min(1.0, top_cosine × 1.2)`
- **Web-only answers:** scored on their own evidence — a base of `0.55` rising
  with each corroborating source, capped at `WEB_ONLY_CONF_CAP = 0.6`

The split matters. Scoring a web-corrected answer by the cosine of passages CRAG
deliberately discarded would penalise correction exactly when it worked.

---

## 6. Reading it live

Every step emits a `trace` event on `/api/ask/stream`, so the path is visible in
the UI as it happens. Each chat call is also a named LangSmith run —
`gate.classify_domain`, `crag.grade_passages`, `crag.rewrite_query`,
`crag.generate`, `rag.generate`, plus the `crag.web_search` tool span. List them
with `python list_traces.py --llm`.

In **Compare** mode both pipelines run concurrently, so totals double and
`classify_domain` appears twice — once per pipeline. That is expected, and it is
also the easiest proof that the two really do run in parallel rather than queue.
