from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text

from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "documents" / "rag_sample_knowledge_base.pdf"
VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "vector_store"


def ingest():

    print("Loading PDF...")

    pages = load_pdf(
        PDF_PATH
    )

    print(
        f"Loaded {len(pages)} pages."
    )


    print("Creating chunks...")

    chunks = chunk_text(
        pages,
        chunk_size=500,
        chunk_overlap=100,
        source="rag_sample_knowledge_base.pdf",
    )

    print(
        f"Created {len(chunks)} chunks."
    )


    print("Loading embedding model...")

    embedder = Embedder()


    texts = [
        chunk["text"]
        for chunk in chunks
    ]


    print("Generating embeddings...")

    embeddings = embedder.embed_texts(
        texts
    )


    print(
        f"Embedding shape: {embeddings.shape}"
    )


    print("Building vector store...")

    vector_store = VectorStore(
        dimension=embeddings.shape[1]
    )


    vector_store.add(
        embeddings,
        chunks,
    )


    print("Saving vector store...")

    vector_store.save(
        VECTOR_STORE_PATH
    )


    print(
        f"Vector store saved to: "
        f"{VECTOR_STORE_PATH}"
    )


if __name__ == "__main__":
    ingest()