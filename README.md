🌱 ESG Insight RAG Assistant

A professional, privacy-focused RAG (Retrieval-Augmented Generation) application designed to query complex ESG (Environmental, Social, and Governance) reports using local LLMs.

This project demonstrates a Clean Architecture (Hexagonal) implementation, separating domain logic from infrastructure to ensure testability, modularity, and scalability.

Python Version: 3.11
Architecture: Hexagonal (Clean Architecture)
LLM: Mistral 7B via Ollama

🌟 Features
📄 Intelligent Ingestion

Parses PDF ESG reports

Splits text into semantic chunks

Generates embeddings locally

🔍 Semantic Search

Uses ChromaDB to retrieve context based on semantic similarity, not just keyword matching.

🤖 Local AI

Powered by Ollama using the Mistral AI Mistral 7B model.

No API keys

No external data sharing

Fully local inference

🏗️ Clean Architecture

Fully decoupled layers:

Domain

Infrastructure

UI

💬 Interactive UI

Real-time conversational interface built with Streamlit.

🏗️ Architecture

This project follows Hexagonal Architecture (Ports & Adapters).

The Core Domain (RagService) defines business logic.

Ports (interfaces) define how external services interact.

Adapters implement those interfaces for tools such as ChromaDB and Ollama.

Diagram is not supported.
🛠️ Tech Stack
Category	Technology
Language	Python 3.11
Package Manager	uv
LLM Inference	Ollama (Model: Mistral)
Embeddings	Sentence-Transformers (all-MiniLM-L6-v2)
Vector Database	ChromaDB
PDF Processing	PyMuPDF
Web Framework	Streamlit
Data Validation	Pydantic
📂 Project Structure
esg-rag-portfolio/
│
├── src/
│   ├── core/                 # Domain Logic (The Hexagon)
│   │   ├── interfaces.py     # Abstract Base Classes (Ports)
│   │   ├── models.py         # Pydantic Models
│   │   └── rag_service.py    # Business Logic Orchestration
│   │
│   ├── infrastructure/       # Adapters (External Services)
│   │   ├── chroma_store.py
│   │   ├── embedder.py
│   │   ├── ollama_service.py
│   │   ├── pdf_loader.py
│   │   └── text_splitter.py
│   │
│   └── ui/                   # Interface Layer
│       └── app.py            # Streamlit Application
│
├── tests/                    # Unit Tests
├── chroma_db/                # Local Vector DB Storage
├── requirements.txt
└── README.md
🚀 Getting Started
Prerequisites

Python 3.11+

Homebrew (for macOS users)

Ollama

1️⃣ Install Dependencies

We use uv for fast dependency management.

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
2️⃣ Setup Local LLM (Ollama)

Ollama runs the AI model locally on your machine.

# Install Ollama
brew install ollama

# Download Mistral model
ollama pull mistral

# Start Ollama server
ollama serve

Keep the server running in a separate terminal.

3️⃣ Run the Application
streamlit run src/ui/app.py

The app will open at:

http://localhost:8501
💡 Usage
Upload

Upload a PDF ESG report from the sidebar
(e.g., Sustainability Report).

Process

The system will automatically:

Extract text

Generate embeddings

Store vectors in the database

Chat

Ask questions in natural language:

What are the company's carbon emission targets?
Analyze

The assistant retrieves relevant document context and generates a response.

🔮 Future Enhancements

 Support additional document formats (DOCX, TXT)

 ESG-specific metric extraction

 Page-level citations in answers

 Docker deployment

📄 License

This project is for portfolio and demonstration purposes.

👩‍💻 Author

Salma EL ANIGRI
NLP & Machine Learning Engineer