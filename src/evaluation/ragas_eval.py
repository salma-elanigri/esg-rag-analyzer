
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from datasets import Dataset

from src.application.rag_service import RagService
from src.infrastructure.anthropic_service import AnthropicService
from src.infrastructure.chroma_store import ChromaStore
from src.infrastructure.document_splitter import DocumentSplitter
from src.infrastructure.embedder import Embedder
from src.infrastructure.pdf_loader import PdfLoader
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
import os
from dotenv import load_dotenv
import litellm
load_dotenv(override=True)

evaluator_llm = llm_factory(
    model="anthropic/claude-haiku-4-5-20251001",
    provider="litellm",
    client=litellm.completion,
    temperature=0.01,
    top_p=None,  # explicitly disable top_p
    max_tokens=4096
)

# wrap your existing embedder for RAGAS
evaluator_embeddings = Embedder()
# Rag service

# initialize RAG
rag = RagService(
    loader=PdfLoader(),
    splitter=DocumentSplitter(),
    vectorstore=ChromaStore(embedder=Embedder()),
    llm_service=AnthropicService()
)
# Dataset
rag.ingest("/Users/salmaelanigri/Documents/portfolio/esg-rag-analyzer/2025-pwc-network-sustainability-report.pdf")

questions = [
    "What are PwC's carbon emission targets?",
    "How does PwC approach diversity and inclusion?",
    "What is PwC's renewable energy strategy?",
    "How does PwC measure its scope 3 emissions?",
    "What social initiatives does PwC support?"
]
answers = []
contexts = []

for question in questions:
    # get answer from RAG
    answer = answer = "".join(rag.ask(question))
    answers.append(answer)

    # get retrieved chunks
    chunks = rag.vectorstore.similarity_search(question, k=5)
    contexts.append([chunk.content for chunk in chunks])

data = {
    "question": questions,
    "answer": answers,  # your RAG's answer
    "contexts": contexts,  # retrieved chunk
}
dataset = Dataset.from_dict(data)

# Metrics now accept the native factory objects directly
faithfulness = Faithfulness(llm=evaluator_llm)
answer_relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)


result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy],
)
print(result)