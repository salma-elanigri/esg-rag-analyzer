from unittest.mock import MagicMock

import pytest

from src.application.rag_service import RagService
from src.core.models import DocumentChunk


# ------------------------------- Fixtures --------------------------------------------------
@pytest.fixture
def mock_loader() -> MagicMock:
    mock_loader = MagicMock()
    mock_loader.load.return_value = [
        DocumentChunk(content="ESG REPORT PAGE 1", page_number=1),
        DocumentChunk(content="ESG REPORT PAGE 2", page_number=2),
    ]
    return mock_loader


@pytest.fixture
def mock_splitter() -> MagicMock:
    mock_splitter = MagicMock()
    mock_splitter.split.return_value = [
        DocumentChunk(content="ESG REPORT CHUNK 1", page_number=1),
        DocumentChunk(content="ESG REPORT CHUNK 2", page_number=1),
        DocumentChunk(content="ESG REPORT CHUNK 3", page_number=1),
        DocumentChunk(content="ESG REPORT CHUNK 1", page_number=2),
        DocumentChunk(content="ESG REPORT CHUNK 2", page_number=2),
        DocumentChunk(content="ESG REPORT CHUNK 3", page_number=2),
    ]
    return mock_splitter


@pytest.fixture
def mock_vectorstore() -> MagicMock:
    mock_vectorstore = MagicMock()
    return mock_vectorstore


@pytest.fixture
def mock_llm() -> MagicMock:
    mock_llm = MagicMock()
    return mock_llm


@pytest.fixture
def rag_service(mock_loader, mock_splitter, mock_vectorstore, mock_llm) -> RagService:
    return RagService(
        loader=mock_loader,
        splitter=mock_splitter,
        vectorstore=mock_vectorstore,
        llm_service=mock_llm,
    )


# ---------------------------- Tests ---------------------------------------------------------
def test_ingest_calls_loader(rag_service, mock_loader):
    """Verify that the loader load is called with the correct path"""
    rag_service.ingest("report.pdf")
    mock_loader.load.assert_called_once_with("report.pdf")


def test_ingest_calls_splitter(rag_service, mock_splitter):
    """Verify that the splitter split is called with each page"""
    rag_service.ingest("report.pdf")
    assert mock_splitter.split.call_count == 2


def test_ingest_calls_vectorstore(rag_service, mock_vectorstore):
    """Verify that the vectorstore add_documents is called once when ingest called"""
    rag_service.ingest("report.pdf")
    mock_vectorstore.add_documents.assert_called_once()


def test_ask_calls_llm(rag_service, mock_llm):
    """Verify that the get_llm() is called when ask called"""
    rag_service.ask("what are the scope 3 targets set for 2030?")
    mock_llm.get_llm.assert_called()


def test_ask_calls_vectorstore(rag_service, mock_vectorstore):
    """Verify that the vectorstore as_retriever is called when ask called"""
    rag_service.ask("what are the scope 3 targets set for 2030?")
    mock_vectorstore.as_retriever.assert_called()
