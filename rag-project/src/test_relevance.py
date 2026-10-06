from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text

from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.reranker import Reranker
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "documents"
    / "rag_sample_knowledge_base.pdf"
)


pages = load_pdf(PDF_PATH)

chunks = chunk_text(
    pages,
    chunk_size=500,
    chunk_overlap=100,
    source="rag_sample_knowledge_base.pdf",
)

embedder = Embedder()

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedder.embed_texts(
    texts
)

vector_store = VectorStore(
    dimension=embeddings.shape[1]
)

vector_store.add(
    embeddings,
    chunks
)

reranker = Reranker()


questions = [
    "What is the purpose of chunk overlap?",
    "What are embeddings?",
    "What is FAISS?",
    "What is the capital of India?",
    "How do I cook pasta?",
]


for question in questions:

    print("\n" + "=" * 80)
    print(question)
    print("=" * 80)

    query_embedding = embedder.embed_text(
        question
    )

    candidates = vector_store.search(
        query_embedding,
        top_k=10,
    )

    results = reranker.rerank(
        question,
        candidates,
        top_k=3,
    )

    for result in results:

        print(
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"Rerank score: "
            f"{result['rerank_score']:.4f}"
        )

        print(
            result["chunk"]["text"][:150]
        )

        print()