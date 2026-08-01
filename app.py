import os
import json
import numpy as np
import streamlit as st
from ingest import ingest_sources, INDEX_FILE
from ask import query_agent
from verify import CitationVerifier

st.set_page_config(
    page_title="Research Agent with Citations",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-contrast CSS for maximum legibility in both light & dark browser modes
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0d1117 !important;
        color: #f0f6fc !important;
    }
    
    /* Global Typography & Headers */
    h1, h2, h3, h4, h5, h6, p, span, label, div, .stMarkdown {
        color: #f0f6fc !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }
    section[data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }

    /* Input Fields & Text Areas */
    .stTextInput input {
        background-color: #21262d !important;
        color: #ffffff !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
    }
    .stTextInput input:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 3px rgba(56, 139, 253, 0.3) !important;
    }

    /* Expanders & Accordions */
    div[data-aria-expanded] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #f0f6fc !important;
    }
    .data-baseweb {
        color: #f0f6fc !important;
    }
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
    }
    div[data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-weight: bold !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #8b949e !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #238636 !important;
        color: #ffffff !important;
        border: 1px solid rgba(240, 246, 252, 0.1) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #2ea043 !important;
        border-color: #8b949e !important;
    }

    /* Alerts & Notifications */
    .stAlert, div[data-testid="stAlert"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #f0f6fc !important;
    }
    
    /* Code blocks */
    code, pre, .stCodeBlock {
        color: #79c0ff !important;
        background-color: #161b22 !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #c9d1d9 !important;
    }
    button[aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom-color: #58a6ff !important;
    }

    /* Selectbox & Dropdown Popover Options */
    div[data-baseweb="select"] > div {
        background-color: #21262d !important;
        border-color: #30363d !important;
        color: #ffffff !important;
    }
    div[data-baseweb="select"] * {
        color: #ffffff !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
    }
    li[role="option"], div[role="option"], ul[role="listbox"] li {
        background-color: #161b22 !important;
        color: #ffffff !important;
    }
    li[role="option"]:hover, div[role="option"]:hover, li[aria-selected="true"] {
        background-color: #1f6feb !important;
        color: #ffffff !important;
    }
    li[role="option"] *, div[role="option"] * {
        color: #ffffff !important;
    }

    /* Custom Citation Badges */
    .citation-badge {
        background-color: #1f6feb !important;
        color: #ffffff !important;
        padding: 2px 8px !important;
        border-radius: 4px !important;
        font-weight: bold !important;
        font-family: monospace !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Verifiable Research Agent (with Inline Citations)")
st.caption("Grounding, cosine similarity retrieval, strict bracket citations, and Step-6 regex verification.")

# Sidebar - Corpus Management
st.sidebar.header("📁 Source Corpus Management")

sample_dir = "sample_sources"
if os.path.exists(sample_dir):
    sample_files = [f for f in os.listdir(sample_dir) if os.path.isfile(os.path.join(sample_dir, f))]
else:
    sample_files = []

st.sidebar.markdown(f"**Loaded Sample Files:** ({len(sample_files)})")
for sf in sample_files:
    st.sidebar.text(f"• {sf}")

if st.sidebar.button("⚡ Re-Ingest Corpus Index", type="primary"):
    with st.spinner("Chunking & Embedding documents..."):
        ingest_sources([f"{sample_dir}/*"])
    st.sidebar.success("Index updated successfully!")

# Ensure index exists
if not os.path.exists(INDEX_FILE):
    with st.spinner("Initializing index from sample sources..."):
        ingest_sources([f"{sample_dir}/*"])

# Main Layout Tabs
tab1, tab2, tab3 = st.tabs(["💬 Query & Research", "📊 Benchmark & Evaluation", "📖 System Architecture"])

with tab1:
    st.subheader("Ask a Research Question")
    
    col_q, col_m, col_k = st.columns([3, 1.5, 1])
    with col_q:
        user_query = st.text_input(
            "Enter question:",
            placeholder="e.g. What was Apex's Q3 Year-over-Year revenue growth rate?",
            key="user_query"
        )
    with col_m:
        mode_choice = st.selectbox("Retrieval Mode", ["Hybrid (RRF)", "Dense Vector", "BM25 Keyword"])
        mode_map = {"Hybrid (RRF)": "hybrid", "Dense Vector": "dense", "BM25 Keyword": "bm25"}
        retrieval_mode = mode_map[mode_choice]
    with col_k:
        top_k = st.slider("Top-k Chunks", min_value=2, max_value=8, value=5)

    if st.button("Submit Research Query", use_container_width=True) and user_query:
        with st.spinner("Retrieving top-k contexts & generating grounded answer..."):
            result = query_agent(user_query, top_k=top_k, retrieval_mode=retrieval_mode)

        # Answer Section
        st.markdown("### Answer")
        st.markdown(result["answer"])

        st.divider()

        # Verification & Metrics Section
        ver = result["verification"]
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Verification Status", ver["status"])
        with m2:
            st.metric("Citation Density", f"{ver['citation_density']*100:.1f}%")
        with m3:
            st.metric("Marker Drop Rate", f"{ver['drop_rate']*100:.1f}%")
        with m4:
            st.metric("Repair Pass Executed", "Yes" if result["repaired"] else "No")

        if ver["uncited_sentences"]:
            with st.expander("⚠️ Missing Citation Sentences Detected"):
                for s in ver["uncited_sentences"]:
                    st.warning(s)

        # Retrieved Contexts Section
        st.markdown("### Retrieved Context Chunks (Top-k)")
        for idx, (chunk, score) in enumerate(result["retrieved_chunks"], start=1):
            with st.expander(f"Chunk {idx}: {chunk['tag']} {chunk['doc_title']} (Similarity: {score:.4f})"):
                st.markdown(f"**Tag:** `{chunk['tag']}` | **Doc:** {chunk['doc_title']}")
                st.info(chunk["chunk_text"])

with tab2:
    st.subheader("Evaluation Suite (questions.json)")
    if os.path.exists("questions.json"):
        with open("questions.json", "r") as f:
            q_set = json.load(f)
        
        st.write(f"Loaded **{len(q_set)}** test questions covering Answerable, Partial, Refusal, and Conflicting scenarios.")
        
        if st.button("🚀 Run Full Evaluation Benchmark"):
            with st.spinner("Executing evaluation across all test cases..."):
                from run_eval import run_evaluation
                run_evaluation()
            st.success("Evaluation benchmark complete!")

        if os.path.exists("eval_results.json"):
            with open("eval_results.json", "r") as f:
                eval_data = json.load(f)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Verification Pass Rate", f"{eval_data['verification_pass_rate_percent']}%")
            c2.metric("Avg Citation Density", f"{eval_data['average_citation_density']*100:.1f}%")
            c3.metric("Avg Marker Drop Rate", f"{eval_data['average_marker_drop_rate']*100:.1f}%")
            c4.metric("Repair Passes", eval_data['repair_passes_executed'])

            st.markdown("#### Test Case Transcript Results")
            for res in eval_data["results"]:
                with st.expander(f"{res['id']} [{res['type']}] Status: {res['verification_status']} — {res['question']}"):
                    st.markdown(f"**Expected Behavior:** {res['expected_behavior']}")
                    st.markdown("**Generated Output:**")
                    st.code(res["generated_answer"])

with tab3:
    st.subheader("System Architecture & Tradeoffs")
    st.markdown("""
    - **Closed-Corpus Mode**: Deterministic retrieval over ingested local vector indices (`index.json`).
    - **Chunking Strategy**: 300–500 tokens sentence-aware boundaries tagged with `[S<doc_id>:<chunk_id>]`.
    - **Embeddings**: Local `sentence-transformers` (`all-MiniLM-L6-v2`) / normalized cosine similarity.
    - **System Prompt Constraints**: Strict requirement for bracket markers, paraphrase cap (~15 words), explicit gap refusal, and conflict reporting.
    - **Step 6 Verification & Section 8 Repair Pass**: Post-generation regex parser flags uncited claims and executes an automated repair prompt pass if drop rate > 0.
    """)
