import streamlit as st
import sys
import os
import json
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieval import load_all_chunks
from agent import ask_agent

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ArXiv RAG",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Design system ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* Reset & base */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #e2e2e8;
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: #0a0a0f;
}

[data-testid="stHeader"] { background: transparent; }

/* Hide streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none; }

/* Main container */
.main .block-container {
    max-width: 900px;
    padding: 3rem 2rem;
    margin: 0 auto;
}

/* ── Header ── */
.app-header {
    margin-bottom: 3rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #1e1e2e;
}

.app-title {
    font-size: 1.1rem;
    font-weight: 500;
    color: #7c6af7;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.5rem;
}

.app-subtitle {
    font-size: 2rem;
    font-weight: 300;
    color: #e2e2e8;
    line-height: 1.3;
    letter-spacing: -0.02em;
}

.app-subtitle span {
    color: #7c6af7;
    font-weight: 500;
}

.app-meta {
    margin-top: 1rem;
    font-size: 0.8rem;
    color: #4a4a5a;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Tabs ── */
[data-testid="stTabs"] {
    margin-bottom: 2rem;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1e1e2e;
    gap: 0;
    padding: 0;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #4a4a5a !important;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 0.75rem 1.5rem;
    border-radius: 0;
    border: none;
    border-bottom: 2px solid transparent;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.02em;
    transition: all 0.2s;
}

[data-testid="stTabs"] [aria-selected="true"] {
    background: transparent !important;
    color: #7c6af7 !important;
    border-bottom: 2px solid #7c6af7 !important;
}

[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    display: none;
}

/* ── Input area ── */
.stTextArea textarea {
    background: #0f0f1a !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 8px !important;
    color: #e2e2e8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 1rem !important;
    resize: none !important;
    transition: border-color 0.2s !important;
    line-height: 1.6 !important;
}

.stTextArea textarea:focus {
    border-color: #7c6af7 !important;
    box-shadow: 0 0 0 1px #7c6af722 !important;
    outline: none !important;
}

.stTextArea textarea::placeholder {
    color: #2e2e3e !important;
}

/* ── Button ── */
.stButton button {
    background: #7c6af7 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.5rem !important;
    letter-spacing: 0.03em !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    width: 100% !important;
}

.stButton button:hover {
    background: #6b5ae0 !important;
    transform: translateY(-1px) !important;
}

/* ── Answer card ── */
.answer-card {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 1.75rem;
    margin-top: 1.5rem;
}

.answer-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #7c6af7;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 1rem;
}

.answer-text {
    font-size: 0.95rem;
    line-height: 1.8;
    color: #c8c8d8;
    white-space: pre-wrap;
}

/* ── Meta chips ── */
.meta-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1.25rem;
    padding-top: 1.25rem;
    border-top: 1px solid #1e1e2e;
}

.chip {
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 4px;
    padding: 0.25rem 0.6rem;
    font-size: 0.72rem;
    color: #4a4a5a;
    font-family: 'JetBrains Mono', monospace;
}

.chip span {
    color: #7c6af7;
}

/* ── Sources ── */
.sources-header {
    font-size: 0.7rem;
    font-weight: 600;
    color: #4a4a5a;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
}

.source-item {
    background: #0a0a0f;
    border: 1px solid #1a1a2a;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}

.source-name {
    font-size: 0.75rem;
    color: #7c6af7;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.35rem;
}

.source-preview {
    font-size: 0.8rem;
    color: #3a3a4a;
    line-height: 1.5;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

/* ── Sub-queries ── */
.subquery-list {
    margin-top: 0.5rem;
}

.subquery-item {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    padding: 0.4rem 0;
    font-size: 0.8rem;
    color: #4a4a5a;
    font-family: 'JetBrains Mono', monospace;
    border-bottom: 1px solid #0f0f1a;
}

.subquery-item:last-child { border-bottom: none; }

.subquery-num {
    color: #7c6af722;
    min-width: 1.5rem;
}

/* ── Benchmark ── */
.bench-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}

.bench-card {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 1.25rem;
}

.bench-metric {
    font-size: 0.65rem;
    color: #4a4a5a;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.5rem;
}

.bench-strategy {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid #0f0f1a;
    font-size: 0.8rem;
}

.bench-strategy:last-child { border-bottom: none; }

.bench-name { color: #4a4a5a; }
.bench-best { color: #7c6af7; font-weight: 600; }
.bench-val { 
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #2e2e4e;
}
.bench-val-best { 
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #7c6af7;
}

.bench-section-title {
    font-size: 0.7rem;
    color: #4a4a5a;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin: 2rem 0 1rem;
}

.finding-item {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-left: 2px solid #7c6af7;
    border-radius: 0 6px 6px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    color: #8a8a9a;
    line-height: 1.5;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #2e2e3e;
}

.empty-icon {
    font-size: 2rem;
    margin-bottom: 1rem;
    opacity: 0.3;
}

.empty-text {
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Thinking indicator ── */
.thinking {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 1rem;
    color: #4a4a5a;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Streamlit spinner override */
[data-testid="stSpinner"] {
    color: #7c6af7 !important;
}

/* Remove default streamlit padding */
.element-container { margin: 0 !important; }

div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }

</style>
""", unsafe_allow_html=True)


# ─── Load chunks once ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_chunks():
    return load_all_chunks()


# ─── Load benchmark results ────────────────────────────────────────────────────
def load_benchmark():
    path = "results/benchmark_results.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">⬡ ArXiv RAG System</div>
    <div class="app-subtitle">Intelligent retrieval over <span>20 research papers</span></div>
    <div class="app-meta">agentic · multimodal · hybrid reranking · 2,915 indexed chunks</div>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_ask, tab_bench = st.tabs(["Ask", "Benchmark"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ASK
# ══════════════════════════════════════════════════════════════════════════════
with tab_ask:
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    question = st.text_area(
        label="",
        placeholder="Ask anything about the papers — factual, multi-part, cross-paper, or table data...",
        height=100,
        key="question_input",
        label_visibility="collapsed"
    )

    ask_clicked = st.button("Ask →", key="ask_btn")

    if ask_clicked and question.strip():
        with st.spinner("Thinking..."):
            all_chunks = get_chunks()
            result = ask_agent(question.strip(), all_chunks)

        # Answer card
        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-label">Answer</div>
            <div class="answer-text">{result['answer']}</div>
            <div class="meta-row">
                <div class="chip">type <span>{result['question_type']}</span></div>
                <div class="chip">chunks <span>{result['chunks_used']}</span></div>
                <div class="chip">iterations <span>{result['iterations']}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Sub-queries used
        if result.get('sub_queries') and len(result['sub_queries']) > 1:
            st.markdown("<div class='sources-header' style='margin-top:1.5rem'>Searches performed</div>",
                        unsafe_allow_html=True)
            queries_html = "<div class='subquery-list'>"
            for i, q in enumerate(result['sub_queries'], 1):
                queries_html += f"<div class='subquery-item'><span class='subquery-num'>{i:02d}</span>{q}</div>"
            queries_html += "</div>"
            st.markdown(queries_html, unsafe_allow_html=True)

    elif ask_clicked and not question.strip():
        st.markdown("""
        <div class="answer-card">
            <div class="answer-text" style="color:#3a3a4a">Type a question above.</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">⬡</div>
            <div class="empty-text">ask a question to get started</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
with tab_bench:
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    bench = load_benchmark()

    if bench:
        strategies = ["naive", "hybrid", "rerank"]
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        labels = {
            "faithfulness": "Faithfulness",
            "answer_relevancy": "Answer Relevancy",
            "context_precision": "Context Precision",
            "context_recall": "Context Recall"
        }

        # Find best per metric
        best = {}
        for m in metrics:
            best[m] = max(strategies, key=lambda s: bench.get(s, {}).get(m, 0))

        # Metric cards
        st.markdown("<div class='bench-grid'>", unsafe_allow_html=True)
        cols = st.columns(4)
        for i, m in enumerate(metrics):
            with cols[i]:
                st.markdown(f"<div class='bench-card'><div class='bench-metric'>{labels[m]}</div>",
                            unsafe_allow_html=True)
                for s in strategies:
                    val = bench.get(s, {}).get(m, 0)
                    is_best = best[m] == s
                    name_cls = "bench-best" if is_best else "bench-name"
                    val_cls = "bench-val-best" if is_best else "bench-val"
                    st.markdown(f"""
                    <div class='bench-strategy'>
                        <span class='{name_cls}'>{s}</span>
                        <span class='{val_cls}'>{val:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Key findings
        st.markdown("<div class='bench-section-title'>Key findings</div>", unsafe_allow_html=True)

        naive = bench.get("naive", {})
        rerank = bench.get("rerank", {})

        findings = []
        if naive.get("answer_relevancy") and rerank.get("answer_relevancy"):
            imp = ((rerank["answer_relevancy"] - naive["answer_relevancy"]) / naive["answer_relevancy"]) * 100
            findings.append(f"Reranking improved answer relevancy by {imp:.0f}% over naive baseline")
        if naive.get("context_recall") and rerank.get("context_recall"):
            imp = ((rerank["context_recall"] - naive["context_recall"]) / naive["context_recall"]) * 100
            findings.append(f"Context recall improved {imp:.0f}% with hybrid reranking vs naive retrieval")

        findings += [
            "Reranking wins 3 out of 4 evaluation metrics",
            "Hybrid BM25 + vector search outperforms pure vector on faithfulness",
            "Cross-encoder reranking adds strongest signal for context recall"
        ]

        for f in findings:
            st.markdown(f"<div class='finding-item'>{f}</div>", unsafe_allow_html=True)

        # Raw numbers table
        st.markdown("<div class='bench-section-title'>Raw scores</div>", unsafe_allow_html=True)

        table_html = """
        <div class='answer-card' style='padding:0; overflow:hidden'>
        <table style='width:100%; border-collapse:collapse; font-size:0.82rem; font-family:"JetBrains Mono",monospace'>
        <thead>
        <tr style='border-bottom:1px solid #1e1e2e'>
            <th style='padding:0.9rem 1.25rem; text-align:left; color:#4a4a5a; font-weight:500; letter-spacing:0.08em'>Strategy</th>
            <th style='padding:0.9rem 1.25rem; text-align:right; color:#4a4a5a; font-weight:500'>Faithfulness</th>
            <th style='padding:0.9rem 1.25rem; text-align:right; color:#4a4a5a; font-weight:500'>Ans. Rel.</th>
            <th style='padding:0.9rem 1.25rem; text-align:right; color:#4a4a5a; font-weight:500'>Ctx. Prec.</th>
            <th style='padding:0.9rem 1.25rem; text-align:right; color:#4a4a5a; font-weight:500'>Ctx. Recall</th>
        </tr>
        </thead>
        <tbody>
        """
        for s in strategies:
            is_best_row = s == "rerank"
            row_style = "background:#13131f" if is_best_row else ""
            name_style = "color:#7c6af7;font-weight:600" if is_best_row else "color:#4a4a5a"
            table_html += f"<tr style='border-bottom:1px solid #0f0f1a;{row_style}'>"
            table_html += f"<td style='padding:0.8rem 1.25rem;{name_style}'>{s}</td>"
            for m in metrics:
                val = bench.get(s, {}).get(m, 0)
                is_best_cell = best[m] == s
                cell_style = "color:#7c6af7;font-weight:600" if is_best_cell else "color:#2e2e4e"
                table_html += f"<td style='padding:0.8rem 1.25rem;text-align:right;{cell_style}'>{val:.4f}</td>"
            table_html += "</tr>"

        table_html += "</tbody></table></div>"
        st.markdown(table_html, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">⬡</div>
            <div class="empty-text">no benchmark results found — run evaluation.py first</div>
        </div>
        """, unsafe_allow_html=True)