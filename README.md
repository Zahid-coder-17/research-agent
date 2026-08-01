# 🔍 Verifiable Research Agent (with Inline Citations & Hybrid Retrieval)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Groq API](https://img.shields.io/badge/LLM-Groq--Llama3.3--70B-orange.svg)](https://groq.com/)
[![Streamlit UI](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Retrieval](https://img.shields.io/badge/Retrieval-Hybrid%20%28Dense%2BBM25%20RRF%29-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An agentic, closed-corpus research assistant that answers complex technical and financial questions using **ONLY** ingested source documents. Every factual claim is verifiably grounded with exact inline bracket citations (`[S<doc_id>:<chunk_id>]`), backed by hybrid RRF retrieval, sentence-level regex post-verification, automated repair passes, and cross-document conflict reporting.

---

## 🌟 Key Features

- 🎯 **Closed-Corpus Grounding**: Answers questions strictly using ingested source documents with zero pretrained memory hallucination. Refuses unanswerable questions explicitly.
- 🔀 **Hybrid Retrieval (Dense + BM25Okapi RRF)**: Combines TF-IDF vector similarity with BM25 keyword search, merged via Reciprocal Rank Fusion ($score = \sum \frac{1}{60 + rank}$). Outperforms dense-only retrieval on exact codes, proper nouns, and numerical figures.
- 📌 **Inline Bracket Citations**: Attaches exact citation markers (e.g. `[S1:00]`, stacked `[S1:00][S2:01]`) to every factual sentence.
- 🛡️ **Step-6 Regex Post-Verification**: Computes sentence-level citation density and marker drop rates on every model response.
- 🔧 **Section 8 Automated Repair Pass**: Detects uncited claims and automatically triggers a targeted repair re-prompt to achieve maximum citation coverage.
- ⚖️ **Cross-Document Conflict Resolution**: Detects and cites contradictory factual statements across different documents without picking sides.
- 🧠 **Self-Healing & Cached Indexing**: BM25 index is cached on disk (`bm25.pkl`) for ~15ms query loading. Vector embedder auto-heals dimension mismatches at query time.
- 📊 **15-Question Evaluation Suite**: Comprehensive benchmark covering directly answerable, partially answerable, unanswerable, and conflicting cases across 10 documents.
- 🎨 **Interactive Streamlit Web UI**: High-contrast visual interface to select retrieval modes, test queries, view similarity scores, and review evaluation transcripts.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph 1. Ingestion & Indexing Pipeline
        A["Source Docs (10 .md files)"] --> B["Sentence-Aware Chunker (300-500 words)"]
        B --> C["Metadata Tagging [S<doc_id>:<chunk_id>]"]
        C --> D["Dense Vector Embedder"]
        C --> E["BM25 Keyword Indexer (BM25Okapi)"]
        D --> F["vectorizer.pkl & index.json"]
        E --> G["bm25.pkl (Cached)"]
    end

    subgraph 2. Hybrid Retrieval Layer (RRF)
        H["User Question"] --> I1["Dense Search (Cosine Sim)"]
        H --> I2["BM25 Search (Keyword Match)"]
        F --> I1
        G --> I2
        I1 --> J1["Ranked List 1 (Dense)"]
        I2 --> J2["Ranked List 2 (BM25)"]
        J1 & J2 --> K["Reciprocal Rank Fusion (RRF k=60)<br/>score = ∑ 1 / (60 + rank)"]
        K --> L["Top-k Deduplicated Context Chunks"]
    end

    subgraph 3. Grounded LLM Execution & Auto-Fallback
        L --> M["Assemble SOURCES Block + System Prompt"]
        M --> N["Groq API (llama-3.3-70b-versatile)"]
        N -- "If Rate Limit 429" --> O["Auto-Fallback (llama-3.1-8b-instant)"]
    end

    subgraph 4. Step-6 Verification & Automated Repair
        N & O --> Q["Raw Model Answer"]
        Q --> R["Step-6 Regex Verifier (r'\[S\d+:\d{2}\]')"]
        R -- "Citation Drop Rate > 0%" --> S["Section 8 Automated Repair Pass"]
        S --> R
        R -- "Citation Drop Rate = 0%" --> T["Verified Answer + Confidence + Sources Used"]
    end
```

---

## 📁 Repository Structure

```text
research-agent/
├── sample_sources/            # Expanded 10 Source Documents
│   ├── doc1_company_q3_report.md
│   ├── doc2_market_analysis.md
│   ├── doc3_sustainability_policy.md
│   ├── doc4_security_whitepaper.md
│   ├── doc5_ai_governance_policy.md
│   ├── doc6_product_roadmap_2026.md
│   ├── doc7_legal_terms_of_service.md
│   ├── doc8_quarterly_audit_notes.md
│   ├── doc9_disaster_recovery_plan.md
│   └── doc10_competitor_landscape.md
├── src/                       # Core engine modules
│   ├── agent.py               # LLM execution, system prompt, and repair pass logic
│   ├── chunker.py             # Sentence-boundary chunker (300-500 words)
│   └── embeddings.py          # Vector embedder & BM25Okapi Hybrid RRF retrieval
├── config.py                  # System prompt, repair template, and .env loader
├── ingest.py                  # Document ingestion & index generator CLI
├── ask.py                     # Research query CLI with --retrieval=dense|bm25|hybrid flag
├── verify.py                  # Step-6 regex citation post-verifier
├── test_hybrid.py             # A/B retrieval benchmark test script
├── run_eval.py                # 15-question evaluation suite runner
├── app.py                     # Interactive Streamlit Web Application
├── questions.json             # Evaluation benchmark test set (15 questions)
├── eval_results.json          # Output evaluation transcript & metrics
├── submission_notes.md        # Technical approach report & trade-off notes
├── requirements.txt           # Pinned python dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Installation & Setup
```bash
git clone https://github.com/Zahid-coder-17/research-agent.git
cd research-agent
pip install -r requirements.txt
```

Create `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Ingest Expanded Source Corpus (10 Documents)
```bash
python ingest.py sample_sources/*
```

### 3. Run A/B Retrieval Test (Dense vs BM25 vs Hybrid)
```bash
python test_hybrid.py
```

### 4. Query Agent via CLI
```bash
python ask.py "What post-quantum cryptography algorithms will Apex adopt by Q2 2026?" --retrieval=hybrid
```

### 5. Run 15-Question Evaluation Suite
```bash
python run_eval.py
```

### 6. Launch Visual Web UI
```bash
streamlit run app.py
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
