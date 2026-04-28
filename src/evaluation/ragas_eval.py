from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from datasets import Dataset
from src.application.rag_service import RagService
from src.infrastructure.anthropic_service import AnthropicService
from src.infrastructure.chroma_store import ChromaStore
from src.infrastructure.document_splitter import DocumentSplitter
from src.infrastructure.embedder import Embedder
from src.infrastructure.pdf_loader import PdfLoader
from ragas.llms import llm_factory
from dotenv import load_dotenv
import litellm
import os

load_dotenv(override=True)

evaluator_llm = llm_factory(
    model="anthropic/claude-haiku-4-5-20251001",
    provider="litellm",
    client=litellm.completion,
    temperature=0.01,
    top_p=None,  # explicitly disable top_p
    max_tokens=4096,
)

# wrap your existing embedder for RAGAS
evaluator_embeddings = Embedder()

# initialize RAG service
rag = RagService(
    loader=PdfLoader(),
    splitter=DocumentSplitter(),
    vectorstore=ChromaStore(embedder=Embedder()),
    llm_service=AnthropicService(),
)

# ingest report
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pdf_path = os.path.join(project_root, "2025-pwc-network-sustainability-report.pdf")
rag.ingest(pdf_path)

# prepare eval dataset
questions = [
    "What are PwC's carbon emission targets?",
    "How does PwC approach diversity and inclusion?",
    "What is PwC's renewable energy strategy?",
    "How does PwC measure its scope 3 emissions?",
    "What social initiatives does PwC support?",
]
ground_truths = [
    # Q1: Carbon emission targets
    "PwC has near-term targets for FY30: reduce scope 1 and 2 absolute emissions by 50% from FY19 base, transition to 100% renewable electricity, reduce business travel emissions by 50% from FY19 base, and have 50% of suppliers set science-based targets. Long-term target for FY50: reach net zero and reduce scope 1, 2 and 3 absolute emissions by 90% from FY19 base. Both validated by SBTi.",
    # Q2: Diversity and inclusion
    "PwC's materiality assessment identifies 'Own workforce' as a key topic, focused on maintaining an inclusive workplace, developing people skills, supporting wellbeing, and attracting and retaining talent. In FY25, 158,298 PwC employees (nearly 44% of people) participated in sustainability upskilling activities.",
    # Q3: Renewable energy strategy
    "PwC procures 99% of electricity from renewable sources globally using a mix of self-generation (1%), bundled Energy Attribute Certificates (2%), and unbundled EACs (63%). On-site generation is often not feasible as most facilities are leased. Target is 100% renewable electricity across all PwC firms by FY30.",
    # Q4: Scope 3 measurement
    "PwC calculates scope 3 using the indirect measurement method with a combination of primary and secondary data. Primary data is prioritized for categories 1, 2, 3 and 6. Secondary data via a third-party model is used for category 7 (employee commuting). PwC follows the GHG Protocol Corporate Value Chain Scope 3 Standard and extends beyond minimum boundaries for categories 6 and 7.",
    # Q5: Social initiatives
    "PwC invested approximately US$36 million in FY25 in climate mitigation including sustainability-focused professionals, renewable energy procurement, supplier engagement and sustainable procurement funding. Sustainability-related projects accounted for approximately US$1.5 billion of revenue in FY25, spanning services supporting clients on climate impacts and energy strategies.",
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
    "ground_truth": ground_truths,
}
dataset = Dataset.from_dict(data)

# Metrics now accept the native factory objects directly
faithfulness = Faithfulness(llm=evaluator_llm)
answer_relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
context_recall = ContextRecall(llm=evaluator_llm)
context_precision = ContextPrecision(llm=evaluator_llm)

result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)
print(result)
