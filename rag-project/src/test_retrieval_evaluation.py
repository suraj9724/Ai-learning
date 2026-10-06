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


def build_test_store():
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

    embeddings = embedder.embed_texts(texts)

    store = VectorStore(
        dimension=embeddings.shape[1]
    )

    store.add(
        embeddings,
        chunks,
    )

    return store, embedder


def evaluate_query(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    reranker: Reranker,
):
    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    query_embedding = embedder.embed_text(question)

    candidates = store.search(
        query_embedding,
        top_k=10,
    )

    results = reranker.rerank(
        question=question,
        results=candidates,
        top_k=5,
    )

    print("\nTOP RESULTS")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        chunk = result["chunk"]

        print("\n" + "-" * 80)

        print(f"Rank: {rank}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Document: {chunk['document']}")
        print(f"Page: {chunk['page_number']}")
        print(
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )
        print(
            f"Rerank score: "
            f"{result['rerank_score']:.4f}"
        )

        print("\nText:")
        print(chunk["text"][:500])


def main():
    store, embedder = build_test_store()

    reranker = Reranker()

    questions = [
        "What are embeddings?",
        "Why is chunk overlap useful?",
        "What is FAISS?",
        "What is retrieval in RAG?",
        "How does a RAG system reduce hallucinations?",
        "What is the capital of India?",
    ]

    for question in questions:
        evaluate_query(
            question,
            store,
            embedder,
            reranker,
        )


if __name__ == "__main__":
    main()