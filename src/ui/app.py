import io

import streamlit as st
import logging
import os
import sys


# Add the project root directory to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Imports from your architecture
from src.infrastructure.pdf_loader import PdfLoader
from src.infrastructure.document_splitter import DocumentSplitter
from src.infrastructure.embedder import Embedder
from src.infrastructure.chroma_store import ChromaStore
from src.infrastructure.anthropic_service import AnthropicService
from src.application.rag_service import RagService

# Configure Logging
logging.basicConfig(level=logging.INFO)


# --- 1. Initialize Services (Cached) ---
@st.cache_resource
def initialize_services():
    """
    Initialize heavy resources (Models, DB connections) only once.
    """
    logging.info("Initializing Services...")
    loader = PdfLoader()
    splitter = DocumentSplitter()
    embedder = Embedder()
    vectorstore = ChromaStore(embedder=embedder)
    llm = AnthropicService()
    rag = RagService(loader, splitter, vectorstore, llm)
    return rag


# Initialize the RAG service
rag_service = initialize_services()

# --- 2. Page Config ---
st.set_page_config(page_title="ESG RAG Assistant", page_icon="🌱", layout="wide")
# --- 3. Sidebar (The Input) ---
with st.sidebar:
    st.header("📄 Document Ingestion")

    # File Uploader
    uploaded_file = st.file_uploader("Upload an ESG Report (PDF)", type="pdf")

    if uploaded_file is not None:
        try:
            # Convert uploaded file to a BytesIO object (Virtual File in RAM)
            bytes_data = io.BytesIO(uploaded_file.getvalue())

            # Pass directly to the service
            with st.spinner("Processing PDF... This might take a moment."):
                rag_service.ingest(bytes_data)

            st.success("PDF Processed successfully! You can now ask questions.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

    else:
        st.info("👆 Upload a PDF to start analyzing.")

# --- 4. Main Chat Interface ---
st.markdown(
    "<h1 style='text-align: center; color: #2E8B57;'>🌿 ESG Assistant</h1>",
    unsafe_allow_html=True,
)
# If no file is uploaded, show a Welcome Hero
if uploaded_file is None:
    st.markdown(
        """
    <div style='text-align: center; padding: 50px;'>
        <h2 style='color: #2E8B57;'>Welcome to ESG Insight</h2>
        <p style='color: #AAAAAA;'>Upload a Sustainability Report to unlock AI-driven insights.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Stop execution here so chat doesn't show
    st.stop()

# Chat History placeholder
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about the report"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        response = st.write_stream(rag_service.ask(prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})
