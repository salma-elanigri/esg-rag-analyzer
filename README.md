# 🌱 ESG Insight RAG Assistant

A RAG application for querying complex ESG reports — turning dense sustainability PDFs into an interactive Q&A interface.

**Python:** 3.11 | **Architecture:** Hexagonal (Clean Architecture) | **LLM:** Claude (Anthropic API) | **Stack:** LangChain · ChromaDB · RAGAS

## 📹 Demo
[▶ Watch the demo](https://www.loom.com/share/337e70ef92c7402c8dafa58ebed20587)

---

## 🏗️ Architecture

Hexagonal Architecture (Ports & Adapters) — the core `RagService` has zero knowledge of external tools. Swapping ChromaDB for Pinecone, or Claude for GPT-4, requires only a new Adapter.

```mermaid
graph TD
    UI[Streamlit UI] --> Service[RAG Service]
    Service -.-> Port1[IDocumentLoader]
    Service -.-> Port2[IDocumentSplitter]
    Service -.-> Port3[IEmbedder]
    Service -.-> Port4[IVectorStore]
    Service -.-> Port5[ILLMService]
    Port1 --> Loader[PDF Loader]
    Port2 --> Splitter[Document Splitter]
    Port3 --> Embedder[SentenceTransformers]
    Port4 --> Store[ChromaDB]
    Port5 --> LLM[Claude API]
```
---

## 🧠 Design Decisions

**Why Hexagonal Architecture?**
Business logic is fully decoupled from infrastructure. The core `RagService` is testable by mocking adapters without a real DB or LLM running.

**Why build RAG manually first, then migrate to LangChain?**
Building the pipeline from scratch proves understanding of *why* RAG works — vector math, retrieval logic, prompt construction. LangChain was added deliberately in v1 to enable LCEL chains and RAGAS evaluation — not as a shortcut.

**Chunking strategy**
Recursive character splitting (chunk size 1000, overlap 200). The overlap handles chunk-boundary answers — semantic splitting was tested but not justified given the embeddings model's robustness.

---

## 🔄 LangChain Migration

### LLM Adapter
- Swapped raw Anthropic client → `ChatAnthropic` (required for LCEL `|` operator)
- Added `get_llm() → BaseChatModel` to `ILLMService` — exposes LangChain object while preserving hexagonal architecture
- Using `claude-haiku-4-5-20251001` for dev — one line to swap to Sonnet

### Embedder
- `Embedder` inherits from both `IEmbedder` and LangChain `Embeddings` (multiple inheritance)
- Renamed `embed_text` → `embed_query` to satisfy LangChain's interface contract

### Vector Store
- Moved `Chroma` initialization to `__init__`
- Added `as_retriever()` to `IVectorStore` interface — avoids reaching inside the adapter from `RagService`
- Removed redundant `embed_documents()` call — `Chroma` handles embedding via `embedding_function`

### RAG Chain
- Replaced manual `_build_prompt()` with `ChatPromptTemplate` — defined once in `__init__`
- Wrapped `format_docs` with `RunnableLambda` — plain Python functions can't be used in `|` chains
- `RunnablePassthrough` pattern passes question to both retriever and prompt simultaneously
- Using `chain.stream()` for Streamlit's `st.write_stream()` — token-by-token typing effect

### RAGAS Evaluation
- Used `litellm` provider via `llm_factory` — RAGAS 0.4.3 dropped `LangchainLLMWrapper` support
- Pre-instantiated `faithfulness` / `answer_relevancy` from `ragas.metrics` with LLM set manually
- Eval script is standalone (`src/evaluation/ragas_eval.py`) — not wired into app architecture

---
## 📊 Evaluation Results (PwC Sustainability Report 2025)

 > **For evaluation:** Download the PwC Network Sustainability Report 2025 and place it in the project root as `2025-pwc-network-sustainability-report.pdf` 
 > before running `python -m src.evaluation.ragas_eval`

Evaluated using the **RAG Triad** — faithfulness, answer relevancy, context recall and context precision — to separate retrieval failures from generation failures.

| Metric | v1 Baseline (dense) | v2 Hybrid Search | v3 Hybrid + Reranking | Δ v1→v3 |
|---|---|---|---|---|
| Faithfulness | 0.48 | 0.71 | 0.66 | +0.18 |
| Answer Relevancy | 0.54 | 0.55 | 0.56 | +0.02 |
| Context Recall | 0.12 | 0.21 | 0.30 | +0.18 |
| Context Precision | 0.20 | 0.14 | 0.33 | +0.13 |

---

### v1 — Baseline (dense only)
Pure ChromaDB dense retrieval using `all-MiniLM-L6-v2` embeddings.

**Diagnosis:** Context recall of 0.12 revealed the retriever as the bottleneck — only 12% of relevant information was being found. Pure semantic search struggled with exact technical terms like *"scope 1"*, *"FY30"*, *"SBTi"*.

---

### v2 — Hybrid Search (BM25 + dense)
Added BM25 alongside ChromaDB via `EnsembleRetriever` (weights 0.5/0.5).

**Results:**
- Faithfulness +0.23 — better context = less hallucination
- Context recall +0.09 — BM25 finds exact technical terms
- Context precision -0.06 — known tradeoff, BM25 introduces keyword-matched noise

---

### v3 — Hybrid Search + Re-ranking (cross-encoder)
Added `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranker via `ContextualCompressionRetriever` (k=5, top_n=5).

**Results:**
- Context precision +0.13 vs baseline — re-ranker filters BM25 noise ✅
- Context recall +0.18 vs baseline — best chunks kept after re-ranking ✅
- Faithfulness +0.18 vs baseline — overall pipeline improvement ✅
- Increasing k from 5 to 10 showed no significant improvement — re-ranker effectively identifies relevant chunks within initial candidate set

**Key insight:** Hybrid search improved recall at the cost of precision. Re-ranking recovered precision while maintaining recall gains — confirming the standard retrieve-then-rerank pipeline as optimal for ESG document QA.

---

### Evaluation dataset
- Ground truth dataset: `src/evaluation/eval_dataset.json` (5 questions, PwC Network Sustainability Report 2025)
- Evaluation script: `src/evaluation/ragas_eval.py`

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Anthropic API key

### Install & Run

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=your_key_here" > .env
streamlit run src/ui/app.py
```

---

## 💡 Usage

Upload a PDF ESG report via the sidebar, then ask questions in the chat:

```
What are the company's carbon emission targets?
How does the company approach supply chain transparency?
```

---

## 📄 License
Portfolio and demonstration purposes.

## 👩‍💻 Author
**Salma EL ANIGRI** — Applied AI Engineer (NLP/LLM)
