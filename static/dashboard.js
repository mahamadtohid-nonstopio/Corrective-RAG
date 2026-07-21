const totalsEl = document.querySelector("#totals");
const lastUpdatedEl = document.querySelector("#last-updated");
const tbodyEl = document.querySelector("#requests tbody");

const chartColors = {
    rag: "#0b1220",           // prussian ink
    ragLight: "rgba(11, 18, 32, 0.10)",
    crag: "#762f2f",          // oxblood
    cragLight: "rgba(118, 47, 47, 0.12)",
    all: "#2a5a3e",           // forest
    grid: "rgba(11, 18, 32, 0.06)",
    tick: "#5b6259",
};

const chartFont = {
    family: "'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, sans-serif",
    size: 11,
};
if (typeof Chart !== "undefined") {
    Chart.defaults.font.family = chartFont.family;
    Chart.defaults.font.size = chartFont.size;
    Chart.defaults.color = chartColors.tick;
    Chart.defaults.borderColor = chartColors.grid;
}

let timelineChart, shareChart, avgChart;

function initCharts() {
    const timelineCtx = document.getElementById("chart-timeline");
    timelineChart = new Chart(timelineCtx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                { label: "RAG", data: [], borderColor: chartColors.rag, backgroundColor: chartColors.ragLight, tension: 0.25 },
                { label: "CRAG", data: [], borderColor: chartColors.crag, backgroundColor: chartColors.cragLight, tension: 0.25 },
            ],
        },
        options: {
            responsive: true,
            interaction: { mode: "index", intersect: false },
            scales: { y: { beginAtZero: true, title: { display: true, text: "LLM calls" } } },
        },
    });

    const shareCtx = document.getElementById("chart-share");
    shareChart = new Chart(shareCtx, {
        type: "doughnut",
        data: {
            labels: ["RAG", "CRAG"],
            datasets: [{ data: [0, 0], backgroundColor: [chartColors.rag, chartColors.crag] }],
        },
        options: { responsive: true, plugins: { legend: { position: "bottom" } } },
    });

    const avgCtx = document.getElementById("chart-avg");
    avgChart = new Chart(avgCtx, {
        type: "bar",
        data: {
            labels: ["Total LLM calls", "Avg per RAG req", "Avg per CRAG req"],
            datasets: [
                {
                    label: "count",
                    data: [0, 0, 0],
                    backgroundColor: [chartColors.all, chartColors.rag, chartColors.crag],
                },
            ],
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true } },
        },
    });
}

function tile(label, value, sub, cls = "") {
    return `<div class="tot ${cls}">
        <div class="tot-label">${label}</div>
        <div class="tot-value">${value}</div>
        <div class="tot-sub">${sub}</div>
    </div>`;
}

function renderTotals(t) {
    if (!t) { totalsEl.innerHTML = ""; return; }
    totalsEl.innerHTML =
        tile("Total requests", t.requests, `${t.rag_requests} RAG · ${t.crag_requests} CRAG`) +
        tile("Total LLM calls", t.llm_calls, "across both pipelines") +
        tile("RAG LLM calls", t.rag_llm_calls, `avg ${t.avg_rag_llm_calls_per_request}/req`, "rag") +
        tile("CRAG LLM calls", t.crag_llm_calls, `avg ${t.avg_crag_llm_calls_per_request}/req`, "crag");
}

function updateCharts(totals, history) {
    // History is newest-first — reverse for chronological chart order.
    const chrono = [...history].reverse();
    const labels = chrono.map((_, i) => `#${i + 1}`);
    const ragData = chrono.map((t) => (t.mode === "rag" ? t.llm_calls : null));
    const cragData = chrono.map((t) => (t.mode === "crag" ? t.llm_calls : null));

    timelineChart.data.labels = labels;
    timelineChart.data.datasets[0].data = ragData;
    timelineChart.data.datasets[1].data = cragData;
    timelineChart.update();

    shareChart.data.datasets[0].data = [totals.rag_requests, totals.crag_requests];
    shareChart.update();

    avgChart.data.datasets[0].data = [
        totals.llm_calls,
        totals.avg_rag_llm_calls_per_request,
        totals.avg_crag_llm_calls_per_request,
    ];
    avgChart.update();
}

function formatWhen(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    if (isNaN(d)) return iso;
    const date = d.toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
    const time = d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", hour12: true });
    return `${date} at ${time}`;
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function renderTable(history) {
    if (!history.length) {
        tbodyEl.innerHTML = `<tr><td colspan="8" style="text-align:center;color:var(--muted);padding:24px">
            No requests yet — ask a question in the app.</td></tr>`;
        return;
    }
    tbodyEl.innerHTML = history
        .map((t, idx) => {
            const nodes = (t.nodes_visited || []).length
                ? t.nodes_visited.join(" → ")
                : "—";
            const status = t.ok
                ? '<span class="status-ok">ok</span>'
                : `<span class="status-err">err</span>`;
            const dur = t.duration_ms ? `${(t.duration_ms / 1000).toFixed(2)}s` : "—";
            return `<tr>
                <td>${history.length - idx}</td>
                <td><span class="mode-chip ${t.mode}">${t.mode.toUpperCase()}</span></td>
                <td class="q" title="${escapeHtml(t.question)}">${escapeHtml(t.question)}</td>
                <td class="nodes">${escapeHtml(nodes)}</td>
                <td><strong>${t.llm_calls}</strong></td>
                <td>${dur}</td>
                <td>${formatWhen(t.finished_at || t.started_at)}</td>
                <td>${status}</td>
            </tr>`;
        })
        .join("");
}

async function refresh() {
    try {
        const resp = await fetch("/api/traces");
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        const data = await resp.json();
        renderTotals(data.totals);
        updateCharts(data.totals, data.history || []);
        renderTable(data.history || []);
        lastUpdatedEl.textContent = "updated " + new Date().toLocaleTimeString();
    } catch (err) {
        lastUpdatedEl.textContent = "refresh failed: " + err.message;
    }
}

initCharts();
refresh();
setInterval(refresh, 2000);
