const $ = (sel) => document.querySelector(sel);
const questionEl = $("#question");
const askBtn = $("#ask");
const resultsEl = $("#results");
const healthEl = $("#health");
const disclaimerEl = $("#disclaimer");
const template = $("#result-template");
const totalsEl = $("#traces-totals");
const historyEl = $("#traces-history");
const refreshTracesBtn = $("#refresh-traces");
const resetTracesBtn = $("#reset-traces");

const CONF_COLORS = {
    HIGH: "#2a5a3e",
    MODERATE: "#8c6b22",
    LOW: "#a04b1a",
    "VERY LOW": "#762f2f",
};

const CONF_TONE = {
    HIGH: "high",
    MODERATE: "moderate",
    LOW: "low",
    "VERY LOW": "low",
};

const prefersReducedMotion =
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// TokenPacer smooths incoming LLM tokens into a character-by-character reveal.
// It absorbs bursts (tokens arrive 5–30 chars at a time) and drips them out at
// a natural reading pace, catching up when the buffer grows large.
class TokenPacer {
    constructor(el, onDrained) {
        this.el = el;
        this.pending = "";
        this.rendered = "";
        this.finished = false;
        this.ticking = false;
        this.onDrained = onDrained;
    }

    push(text) {
        if (!text) return;
        if (this.rendered === "" && this.pending === "") {
            this.el.textContent = "";
            this.el.classList.remove("thinking");
        }
        this.pending += text;
        this._ensureTicking();
    }

    finish() {
        this.finished = true;
        this._ensureTicking();
    }

    _ensureTicking() {
        if (this.ticking) return;
        this.ticking = true;
        this.el.classList.add("streaming");
        if (prefersReducedMotion) {
            // Flush immediately, no per-char reveal.
            this._drainAll();
            return;
        }
        this._tick();
    }

    _drainAll() {
        this.rendered += this.pending;
        this.pending = "";
        this.el.textContent = this.rendered;
        if (this.finished) this._done();
        else {
            // stay armed for more
            this.ticking = false;
        }
    }

    _tick() {
        if (this.pending.length === 0) {
            if (this.finished) {
                this._done();
                return;
            }
            // idle — poll next frame for new tokens
            requestAnimationFrame(() => this._tick());
            return;
        }
        // Chunk size grows with backlog to catch up without ever "dumping".
        const n = this.pending.length;
        const chunk = n > 400 ? 6 : n > 150 ? 3 : n > 40 ? 2 : 1;
        const step = this.pending.slice(0, chunk);
        this.rendered += step;
        this.pending = this.pending.slice(chunk);
        this.el.textContent = this.rendered;

        const last = this.rendered.slice(-1);
        const prev = this.rendered.slice(-2, -1);
        let delay = 6;
        if (/[.!?]/.test(last) && !/\d/.test(prev)) delay = 55;
        else if (/[,;:]/.test(last)) delay = 22;
        else if (last === "\n") delay = 30;

        setTimeout(() => this._tick(), delay);
    }

    _done() {
        this.el.classList.remove("streaming");
        this.ticking = false;
        this.onDrained?.();
    }
}

async function checkHealth() {
    try {
        const resp = await fetch("/api/health");
        const data = await resp.json();
        if (data.status === "ok") {
            healthEl.className = "health ok";
            const web = data.web_fallback_enabled ? "web fallback: on" : "web fallback: off";
            const provider = data.provider || "llm";
            const store = data.vector_store
                ? `${data.vector_store}:${data.pinecone_index || "?"}/${data.pinecone_namespace || "default"}`
                : "vector store";
            const ls = data.langsmith_enabled
                ? ` · langsmith:${data.langsmith_project || "on"}`
                : "";
            healthEl.textContent = `${provider} · ${data.chat_model} · ${store} · top-k ${data.top_k} · ${web}${ls}`;
        } else {
            healthEl.className = "health error";
            healthEl.textContent = "startup error";
            healthEl.title = data.error || "";
        }
    } catch (err) {
        healthEl.className = "health error";
        healthEl.textContent = "server unreachable";
    }
}

function makeResultCard(title) {
    const node = template.content.firstElementChild.cloneNode(true);
    node.querySelector(".title").textContent = title;
    return node;
}

function renderPending(container, title) {
    const node = makeResultCard(title);
    const answerEl = node.querySelector(".answer");
    answerEl.classList.add("thinking");
    answerEl.innerHTML =
        'Consulting the corpus<span class="dots"><span></span><span></span><span></span></span>';
    container.appendChild(node);
    return node;
}

function applyMeta(node, meta) {
    const conf = meta.confidence ?? 0;
    const label = meta.confidence_label || "VERY LOW";
    const labelEl = node.querySelector(".conf-label");
    labelEl.textContent = label;
    labelEl.setAttribute("data-label", label);
    labelEl.className = "conf-label " + label;
    node.querySelector(".conf-value").textContent = conf.toFixed(2);
    const fill = node.querySelector(".fill");
    fill.style.width = Math.round(Math.min(Math.max(conf, 0), 1) * 100) + "%";
    fill.style.background = CONF_COLORS[label] || "var(--ink)";

    node.classList.add("done");
    const tone = CONF_TONE[label];
    if (tone) node.classList.add(tone);

    const citationList = node.querySelector(".citation-list");
    citationList.innerHTML = "";
    if (meta.citations && meta.citations.length) {
        for (const c of meta.citations) {
            const li = document.createElement("li");
            const page = c.page != null ? `, p.${c.page}` : "";
            li.innerHTML =
                `<span class="source">${escapeHtml(c.source)}${page}</span>` +
                `<span class="snippet">${escapeHtml(c.snippet)}</span>`;
            citationList.appendChild(li);
        }
        node.querySelector("details.citations").style.display = "";
    } else {
        node.querySelector("details.citations").style.display = "none";
    }

    if (meta.disclaimer) {
        disclaimerEl.textContent = meta.disclaimer;
    }
}

function applyTrace(node, messages) {
    const traceList = node.querySelector(".trace-list");
    const traceDetails = node.querySelector("details.trace");
    traceList.innerHTML = "";
    if (messages && messages.length) {
        traceDetails.style.display = "";
        for (const step of messages) {
            const li = document.createElement("li");
            li.textContent = step;
            traceList.appendChild(li);
        }
    } else {
        traceDetails.style.display = "none";
    }
}

function fillError(node, message) {
    const answerEl = node.querySelector(".answer");
    answerEl.classList.remove("thinking", "streaming");
    answerEl.innerHTML =
        `<span style="color:var(--marker);font-style:italic">Could not complete the query — ${escapeHtml(message)}</span>`;
    const labelEl = node.querySelector(".conf-label");
    labelEl.textContent = "N/A";
    labelEl.className = "conf-label";
    node.querySelector(".conf-value").textContent = "–";
    node.querySelector(".fill").style.width = "0%";
    node.querySelector("details.citations").style.display = "none";
    node.querySelector("details.trace").style.display = "none";
    node.classList.add("done", "low");
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

async function streamAsk(mode, question, handlers) {
    const resp = await fetch("/api/ask/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, question }),
    });
    if (!resp.ok) {
        let msg = `HTTP ${resp.status}`;
        try {
            const data = await resp.json();
            msg = data.detail || msg;
        } catch {}
        throw new Error(msg);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const dispatch = (line) => {
        if (!line) return;
        let event;
        try { event = JSON.parse(line); } catch { return; }
        switch (event.type) {
            case "trace": handlers.onTrace?.(event.message); break;
            case "token": handlers.onToken?.(event.text); break;
            case "meta":  handlers.onMeta?.(event); break;
            case "error": throw new Error(event.message || "stream error");
            case "done":  handlers.onDone?.(); break;
        }
    };
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let i;
        while ((i = buffer.indexOf("\n")) !== -1) {
            const line = buffer.slice(0, i).trim();
            buffer = buffer.slice(i + 1);
            dispatch(line);
        }
    }
    if (buffer.trim()) dispatch(buffer.trim());
}

async function runAsk(mode, question, container, title) {
    const node = renderPending(container, title);
    const answerEl = node.querySelector(".answer");
    const traceMessages = [];
    let metaData = null;

    // Pacer drains real tokens into the character-by-character reveal.
    // When the pacer finishes, we apply confidence, citations, and trace.
    const finalize = () => {
        if (metaData) applyMeta(node, metaData);
        applyTrace(node, traceMessages);
    };
    const pacer = new TokenPacer(answerEl, finalize);

    try {
        await streamAsk(mode, question, {
            onTrace: (msg) => traceMessages.push(msg),
            onToken: (text) => pacer.push(text),
            onMeta: (event) => { metaData = event; },
        });
        pacer.finish();
    } catch (err) {
        pacer.finish();
        fillError(node, err.message || String(err));
    }
}

let liveTimer = null;
function startLivePoll() {
    stopLivePoll();
    liveTimer = setInterval(loadTraces, 600);
}
function stopLivePoll() {
    if (liveTimer) {
        clearInterval(liveTimer);
        liveTimer = null;
    }
}

askBtn.addEventListener("click", async () => {
    const question = questionEl.value.trim();
    if (question.length < 3) {
        questionEl.focus();
        return;
    }
    const mode = document.querySelector('input[name="mode"]:checked').value;
    resultsEl.innerHTML = "";
    resultsEl.className = "results" + (mode === "compare" ? " compare" : "");

    askBtn.disabled = true;
    askBtn.textContent = "Retrieving";
    startLivePoll();

    try {
        if (mode === "compare") {
            await Promise.all([
                runAsk("rag", question, resultsEl, "RAG"),
                runAsk("crag", question, resultsEl, "CRAG"),
            ]);
        } else if (mode === "rag") {
            await runAsk("rag", question, resultsEl, "RAG");
        } else {
            await runAsk("crag", question, resultsEl, "CRAG");
        }
    } finally {
        stopLivePoll();
        askBtn.disabled = false;
        askBtn.textContent = "Ask";
        loadTraces();
    }
});

questionEl.addEventListener("keydown", (evt) => {
    if ((evt.ctrlKey || evt.metaKey) && evt.key === "Enter") {
        askBtn.click();
    }
});

function formatWhen(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    if (isNaN(d)) return iso;
    const date = d.toLocaleDateString("en-GB", {
        day: "numeric",
        month: "long",
        year: "numeric",
    });
    const time = d.toLocaleTimeString("en-US", {
        hour: "numeric",
        minute: "2-digit",
        hour12: true,
    });
    return `${date} at ${time}`;
}

function renderTotals(totals) {
    if (!totals) {
        totalsEl.innerHTML = "";
        return;
    }
    const cells = [
        { label: "Total requests", value: totals.requests, sub: `${totals.rag_requests} RAG · ${totals.crag_requests} CRAG` },
        { label: "Total LLM calls", value: totals.llm_calls, sub: "across both pipelines", cls: "" },
        { label: "RAG LLM calls", value: totals.rag_llm_calls, sub: `avg ${totals.avg_rag_llm_calls_per_request}/req`, cls: "rag" },
        { label: "CRAG LLM calls", value: totals.crag_llm_calls, sub: `avg ${totals.avg_crag_llm_calls_per_request}/req`, cls: "crag" },
    ];
    totalsEl.innerHTML = cells
        .map(
            (c) =>
                `<div class="tot ${c.cls || ""}">
                    <div class="tot-label">${c.label}</div>
                    <div class="tot-value">${c.value}</div>
                    <div class="tot-sub">${c.sub}</div>
                </div>`
        )
        .join("");
}

function renderHistory(history) {
    if (!history || !history.length) {
        historyEl.innerHTML = '<li class="trace-empty">No requests yet — ask a question above.</li>';
        return;
    }
    historyEl.innerHTML = history
        .map((t) => {
            const errCls = t.ok ? "" : "err";
            const q = escapeHtml(t.question || "");
            return `<li class="${errCls}">
                <span class="mode-chip ${t.mode}">${t.mode.toUpperCase()}</span>
                <span class="trace-q" title="${q}">${q}</span>
                <span class="trace-when">
                    <strong>${t.llm_calls}</strong> LLM call${t.llm_calls === 1 ? "" : "s"} · ${t.duration_ms} ms<br>
                    ${t.ok ? "done" : "failed"} on ${formatWhen(t.finished_at || t.started_at)}
                </span>
            </li>`;
        })
        .join("");
}

async function loadTraces() {
    try {
        const resp = await fetch("/api/traces");
        if (!resp.ok) return;
        const data = await resp.json();
        renderTotals(data.totals);
        renderHistory(data.history);
    } catch (err) {
        console.error("failed to load traces", err);
    }
}

refreshTracesBtn?.addEventListener("click", loadTraces);
resetTracesBtn?.addEventListener("click", async () => {
    if (!confirm("Reset all trace history?")) return;
    await fetch("/api/traces/reset", { method: "POST" });
    loadTraces();
});

checkHealth();
loadTraces();
