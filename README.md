# AI Engineer Portfolio

70-day intensive program building production-grade AI engineering skills.

**Stack:** Python · LangChain · LangGraph · ChromaDB · FAISS · 
FastAPI · HuggingFace · Sentence Transformers · RAGAS

---

## Phase 1 — Foundations (Days 1–10)

Core AI engineering concepts implemented from scratch.

| Day | Topic | Key files |
|-----|-------|-----------|
| 1 | Python AI stack, schemas, Pydantic | `schemas.py` |
| 2 | Vector DBs — Chroma, FAISS, Pinecone, Weaviate | `vectordb_compare.py` |
| 3 | RAG chunking strategies + document loaders | `chunking_strategies.py` |
| 4 | BM25, hybrid search, RRF, CrossEncoder reranking | `retrieval_hybrid.py` |
| 5 | RAG evaluation — RAGAS 5 metrics | `rag_evaluation_ragas.py` |
| 6 | LangGraph — StateGraph, nodes, edges, checkpointing | `langgraph_basics.py` |
| 7 | Agentic AI — ReAct, plan-execute, reflection, memory | `agent_react.py` |
| 8 | GenAI internals — attention, tokenization, prompt engineering | `prompt_engineering.py` |

## Phase 2 — Projects (Days 11–40) 🔄 in progress

- [ ] Production RAG chatbot (LangChain + ChromaDB + FastAPI)
- [ ] Multi-agent research assistant (LangGraph)

## Phase 3 — Advanced Projects (Days 41–70) 🔜

- [ ] Document intelligence agent with memory
- [ ] GenAI-powered data analyst

---

## Setup

```bash
git clone https://github.com/poojaramakrishna930/ai-engineer-portfolio
cd ai-engineer-portfolio
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

## Running any file

```bash
python src/filename.py
```

---

*Updated almost daily. Each commit = one day of work.*
