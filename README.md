# HR Recognition Analytics — Agentic RAG Pipeline

> Northeastern University × Workhuman Capstone — Spring 2026

End-to-end NLP pipeline that classifies 1,000 employee recognition messages into a hierarchical topic taxonomy using LLM-driven grounded theory, then builds an agentic RAG chatbot on top for natural language HR analytics.

---

## Demo

[![Watch the demo](https://img.shields.io/badge/Loom-Watch_Demo-625DF5?style=for-the-badge&logo=loom&logoColor=white)](https://www.loom.com/share/04c68b3e16d9421f86810523502f52b3)

**Demo video:** https://www.loom.com/share/04c68b3e16d9421f86810523502f52b3

## Highlights

- **Data-driven taxonomy:** an LLM grounded-theory pipeline built a 4-category, 26-subcategory taxonomy from 1,000 raw recognition messages, with no predefined labels.
- **Reliable batch annotation:** classified 999 records in 50 batches with count validation, retries, and API key rotation on rate limits, so the full run completes on free-tier quotas.
- **Hybrid retrieval:** combines ChromaDB semantic search, pandas analytics, and a 413-node recognition graph (Neo4j with a NetworkX fallback) behind a 6-route query router.
- **Retrieval check before answering:** a CRAG-style relevance gate reranks or falls back to another tool when retrieved context is weak.
- **Full stack:** FastAPI backend plus a React + Chart.js frontend, including natural-language chart generation from live data.
- **Evaluated:** RAGAS answer relevancy of 0.776, plus a pytest unit test suite.

## What it does

**Stage 1 — Taxonomy creation:** Processes raw recognition messages in batches of 50 through Gemini. Uses grounded theory — no predefined labels. The taxonomy is compressed every 5 batches to remove redundancy. Output: 4 categories, 26 subcategories with hierarchical IDs (A, A1, A2...).

**Stage 2 — Annotation:** Classifies all 999 deduplicated records against the taxonomy in batches of 20, with per-batch count validation and retry logic. Failed batches are saved separately and retried. Output: `annotated_results.csv`.

**Stage 3 — Agentic RAG:** Chatbot answers natural language HR analytics questions using 5 tools, a 6-route query router, CRAG-style retrieval checking, a Neo4j graph (with a NetworkX fallback), and session memory.

**Stage 4 — Visual analytics:** Natural language → LLM picks the chart type → builds a Chart.js config from real CSV aggregates → renders in the browser. No hardcoded charts.

---

## Architecture

```
Raw CSV (1000 messages)
       │
       ▼
┌─────────────────────────────────────┐
│         PIPELINE                    │
│  Dedup → Taxonomy Creation          │
│  (Gemini, batch=50, compress/5)     │
│         ↓                           │
│  Annotation (999 records, batch=20) │
│  annotated_results.csv              │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
  ChromaDB          Neo4j AuraDB
  (384-dim           (413 nodes)
  embeddings)       NetworkX fallback
       │                │
       └───────┬────────┘
               ▼
┌─────────────────────────────────────┐
│         QUERY ROUTER                │
│  DIRECT · SEMANTIC · ANALYTICAL     │
│  HYBRID · GRAPH · SUBSTITUTION      │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│         5 AGENT TOOLS               │
│  vector_search    pandas_analytics  │
│  substitution     neo4j_query       │
│  graph_query (NetworkX fallback)    │
└──────────────┬──────────────────────┘
               ▼
         CRAG checker
    (keyword overlap scoring)
               ▼
        Groq LLM answer
               ▼
┌─────────────────────────────────────┐
│         SERVING LAYER               │
│  FastAPI backend  (/api/chat,       │
│  /api/stats, /api/visualize)        │
│  React + Chart.js frontend          │
└─────────────────────────────────────┘
```

---

## Taxonomy

Built from the data — not predefined:

```
A. Strategic Business Outcomes
   A1. Business Growth & Client Success
   A2. Customer Experience & Engagement
   A3. Strategic Direction & Innovation

B. Operational Excellence & Delivery
   B1. Project & Process Management
   B2. Technical & System Operations
   B3–B9. (7 more subcategories)

C. Creative & Brand Impact
   C1. Content Strategy & Creation
   C2. Visual Design & Branding
   C3–C5. (3 more subcategories)

D. Organizational & Team Enablement
   D1. Team Collaboration & Integration
   D2. Learning & Professional Development
   D3–D9. (7 more subcategories)
```

**Distribution across 999 records:**

| Category | Records | % |
|---|---|---|
| Organizational & Team Enablement | 442 | 44.2% |
| Operational Excellence & Delivery | 340 | 34.0% |
| Strategic Business Outcomes | 164 | 16.4% |
| Creative & Brand Impact | 53 | 5.3% |

---

## RAG System

### Query Router — 6 routes

Rule-based classification (regex signal banks with confidence levels) in `src/rag/router.py`:

| Route | Trigger | Tools |
|---|---|---|
| `DIRECT` | Greetings, meta questions | None |
| `SEMANTIC` | Themes, examples, qualitative | `vector_search` |
| `ANALYTICAL` | Counts, rankings, distributions | `pandas_analytics` |
| `HYBRID` | Role + qualitative combined | `pandas_analytics` + `vector_search` |
| `GRAPH` | Team relationships, network | `neo4j_query` / `graph_query` |
| `SUBSTITUTION` | "Who could replace X" | `substitution` |

### CRAG-style retrieval check

A lightweight relevance gate based on keyword overlap between the query and the retrieved context:
- Score ≥ 0.25 → proceed
- Score ≥ 0.125 → rerank
- Score < 0.125 → fallback (adds `vector_search` as a backup tool)

### Graph

- 413 nodes: 4 categories, 26 subcategories, 11 role groups, 372 individual roles
- Relationships: `RECOGNIZED` between role groups (with `count`, `top_category`) and `BELONGS_TO` for roles → groups and subcategories → categories
- Most central: Engineering (4,655 connections)
- Strongest cross-team flow: Product/Program → Engineering (21)
- If Neo4j is unavailable, the same graph is rebuilt in memory with NetworkX from the CSV

### RAGAS Evaluation

| Metric | Score |
|---|---|
| Answer Relevancy | **0.776** |
| Faithfulness | Not completed (free-tier rate limits) |
| Context Precision | Not completed (free-tier rate limits) |

---

## Visual Analytics

`/api/visualize` endpoint:
1. Loads `annotated_results.csv`
2. Builds a data summary (category counts, top 20 subcategories, give/receive by role group, monthly trend)
3. Sends it to the Groq LLM with a Chart.js prompt — the model returns a pure JSON config
4. The frontend renders any chart type: bar, horizontal bar, doughnut, line, radar, scatter

---

## Project Structure

```
├── src/
│   ├── pipeline/
│   │   ├── create_taxonomy.py     # Stage 1: build taxonomy
│   │   ├── annotate.py            # Stage 2: classify records
│   │   └── retry_failed.py        # Retry failed batches
│   ├── llm/                       # Gemini client + prompts
│   ├── rag/
│   │   ├── ingest.py              # ChromaDB ingestion
│   │   ├── tools.py               # vector_search, pandas_analytics, substitution
│   │   ├── router.py              # Query router + CRAG checker
│   │   ├── agent.py               # Agent: route → tools → check → answer
│   │   ├── graph_tool.py          # NetworkX in-memory graph
│   │   ├── neo4j_loader.py        # Load graph into Neo4j
│   │   ├── neo4j_tool.py          # 7 Cypher query types
│   │   └── ragas_eval.py          # RAGAS evaluation
│   ├── api/
│   │   └── server.py              # FastAPI: /chat /stats /visualize
│   ├── utils/                     # Config, logging, JSON + data helpers
│   └── visualization/
│       ├── dashboard.jsx          # React chat UI
│       └── VisualAnalytics.jsx    # Chart generation UI
├── tests/                         # pytest unit tests
├── data/raw/
│   └── mockup_awards.csv          # 1,000 recognition records
├── config/
│   └── config.yaml                # Pipeline settings
├── output/reports/                # Generated by the pipeline (not committed)
├── package.json                   # Frontend (React + Vite)
└── requirements.txt               # Backend (Python 3.11)
```

---

## Running Locally

```bash
# 1. Clone and set up the environment
git clone https://github.com/monu4959/HR-Chatbot-2.git
cd HR-Chatbot-2
conda create -n hr-analytics python=3.11
conda activate hr-analytics
pip install -r requirements.txt

# 2. Add environment variables
cp .env.example .env
# Fill in GEMINI_API_KEY and GROQ_API_KEY (Neo4j is optional)

# 3. Run the pipeline (writes to output/reports/)
python -m src.pipeline.create_taxonomy
python -m src.pipeline.annotate
python -m src.pipeline.retry_failed      # only if some batches failed

# 4. Build the vector store
python -m src.rag.ingest

# 5. Load the Neo4j graph (optional — NetworkX is used otherwise)
python -m src.rag.neo4j_loader

# 6. Start the FastAPI backend
uvicorn src.api.server:app --reload --port 8000

# 7. Start the React frontend (separate terminal)
npm install
npm run dev        # http://localhost:5173

# Tests
pytest
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM (taxonomy, annotation, RAGAS judge) | Google Gemini 2.5 Flash |
| LLM (chat agent, chart generation) | Groq — Llama 3.3 70B Versatile (configurable via `GROQ_MODEL`) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, local) |
| Vector DB | ChromaDB (local persistent, cosine similarity) |
| Graph DB | Neo4j AuraDB (cloud) + NetworkX (fallback) |
| Backend | FastAPI + Uvicorn |
| Frontend | React + Vite + Chart.js |
| Evaluation | RAGAS 0.4.3 |
| Environment | Conda, Python 3.11 |

---

## Key Technical Decisions

**Raw urllib for annotation (not the SDK)** — more predictable on free-tier APIs with rate limits; fewer layers between the request and the retry logic.

**API key rotation** — reads `GEMINI_API_KEY`, `GEMINI_API_KEY_2`, ... from `.env` and rotates on 429, so the 50 annotation batches finish on free-tier quotas without manual intervention.

**Grounded theory taxonomy** — labels emerge from the data rather than being predefined. Each batch only adds new categories; compression every 5 batches keeps the schema clean.

**NetworkX fallback for Neo4j** — the app works fully even if the cloud DB is down. The graph is rebuilt in memory from the CSV on startup.

**Retrieval check before generation** — retrieval quality is scored before context reaches the LLM. Low-scoring retrievals trigger a fallback tool call instead of answering from weak context.

---

*Northeastern University — MS Data Analytics Engineering · Workhuman Capstone · Spring 2026*
