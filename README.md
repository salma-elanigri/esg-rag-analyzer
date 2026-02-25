# 🌱 ESG Insight RAG Assistant

A **privacy-focused Retrieval-Augmented Generation (RAG)** application
designed to query complex **ESG (Environmental, Social, and
Governance)** reports using **local LLMs**.

This project demonstrates **Hexagonal (Clean) Architecture**, separating
**domain logic from infrastructure** to ensure **modularity,
scalability, and testability**.

**Python:** 3.11\
**Architecture:** Hexagonal (Clean Architecture)\
**LLM:** Mistral via Ollama

------------------------------------------------------------------------

# 🌟 Features

### 📄 Intelligent Ingestion

-   Parses **PDF ESG reports**
-   Splits text into **semantic chunks**
-   Generates **embeddings locally**

### 🔍 Semantic Search

Uses **ChromaDB** to retrieve context based on **semantic similarity**,
not just keywords.

### 🤖 Local AI

Powered by **Ollama** with the **Mistral 7B model**.

-   No API keys\
-   No external data sharing\
-   Fully local inference

### 🏗️ Clean Architecture

Fully decoupled layers:

-   **Domain**
-   **Infrastructure**
-   **Interface**

### 💬 Interactive UI

Real-time chat interface built with **Streamlit**.

------------------------------------------------------------------------

# 🏗️ Architecture

This project follows **Hexagonal Architecture (Ports & Adapters)**.

-   The **Core Domain (`RagService`)** defines the business logic.
-   **Ports (interfaces)** define contracts.
-   **Adapters** implement those contracts for external services.

``` mermaid
graph TD

    subgraph Interface Layer
        UI[Streamlit UI]
    end

    subgraph Core Domain
        Service[RAG Service]
        Port1[IDocumentLoader]
        Port2[ITextSplitter]
        Port3[IVectorStore]
        Port4[ILLMService]
    end

    subgraph Infrastructure
        Loader[PDF Loader Adapter]
        Splitter[Text Splitter Adapter]
        Store[ChromaDB Adapter]
        LLM[Ollama Adapter]
        Embedder[Sentence Transformers]
    end

    UI --> Service

    Service -.-> Port1
    Service -.-> Port2
    Service -.-> Port3
    Service -.-> Port4

    Loader ..> Port1
    Splitter ..> Port2
    Store ..> Port3
    LLM ..> Port4

    Embedder --> Store
```

------------------------------------------------------------------------

# 🛠️ Tech Stack

  Category          Technology
  ----------------- -----------------------
  Language          Python 3.11
  Package Manager   uv
  LLM Inference     Ollama
  Embeddings        Sentence Transformers
  Vector Database   ChromaDB
  PDF Processing    PyMuPDF
  Web Framework     Streamlit
  Data Validation   Pydantic

------------------------------------------------------------------------

# 📂 Project Structure

    esg-rag-portfolio/
    │
    ├── src/
    │   ├── core/
    │   │   ├── interfaces.py
    │   │   ├── models.py
    │   │   └── rag_service.py
    │   │
    │   ├── infrastructure/
    │   │   ├── chroma_store.py
    │   │   ├── embedder.py
    │   │   ├── ollama_service.py
    │   │   ├── pdf_loader.py
    │   │   └── text_splitter.py
    │   │
    │   └── ui/
    │       └── app.py
    │
    ├── tests/
    ├── chroma_db/
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

# 🚀 Getting Started

## Prerequisites

-   Python **3.11+**
-   Homebrew (Mac users)
-   Ollama

------------------------------------------------------------------------

# 1️⃣ Install Dependencies

``` bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

------------------------------------------------------------------------

# 2️⃣ Setup Local LLM

``` bash
# Install Ollama
brew install ollama

# Download Mistral model
ollama pull mistral

# Start server
ollama serve
```

Keep this running in another terminal.

------------------------------------------------------------------------

# 3️⃣ Run the Application

``` bash
streamlit run src/ui/app.py
```

Open:

    http://localhost:8501

------------------------------------------------------------------------

# 💡 Usage

### Upload

Upload a **PDF ESG report** (for example a sustainability report).

### Process

The system will automatically:

-   Extract text
-   Create embeddings
-   Store vectors in the database

### Chat

Ask questions like:

    What are the company's carbon emission targets?

The assistant retrieves relevant sections and generates an answer.

------------------------------------------------------------------------

# 🔮 Future Enhancements

-   Support **DOCX and TXT**
-   ESG metric extraction
-   Page citations in answers
-   Docker deployment

------------------------------------------------------------------------

# 📄 License

This project is for **portfolio and demonstration purposes**.

------------------------------------------------------------------------

## 👩‍💻 Author

**Salma EL ANIGRI**\
NLP & Machine Learning Engineer
