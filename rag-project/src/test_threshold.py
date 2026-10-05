from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "documents" / "rag_sample_knowledge_base.pdf"


# Load document
pages = load_pdf(PDF_PATH)


# Create chunks
chunks = chunk_text(
    pages,
    chunk_size=500,
    chunk_overlap=100,
)


# Create embedder
embedder = Embedder()


# Create embeddings
texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedder.embed_texts(texts)


# Create vector store
vector_store = VectorStore(
    dimension=embeddings.shape[1]
)


vector_store.add(
    embeddings,
    chunks,
)


questions = [
    "What is the purpose of chunk overlap?",
    "What are embeddings?",
    "What is FAISS used for?",
    "What is the capital of India?",
    "How do I cook pasta?",
]


for question in questions:

    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    query_embedding = embedder.embed_text(
        question
    )

    results = vector_store.search(
        query_embedding,
        top_k=3,
    )

    for i, result in enumerate(
        results,
        start=1,
    ):

        chunk = result["chunk"]

        print(
            f"\nResult {i}"
        )

        print(
            f"Distance: "
            f"{result['distance']:.4f}"
        )

        print(
            f"Page: "
            f"{chunk['page_number']}"
        )

        print(
            f"Text: "
            f"{chunk['text'][:200]}..."
        )