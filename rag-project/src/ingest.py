from pathlib import Path
import hashlib

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = PROJECT_ROOT / "data" / "documents"
VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "vector_store"



def create_document_id(filename: str) -> str:
    return hashlib.sha256(
        filename.encode("utf-8")
    ).hexdigest()[:12]


def ingest():
    pdf_files = sorted(DOCUMENTS_PATH.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCUMENTS_PATH}"
        )

    all_chunks = []

    for pdf_path in pdf_files:

        document_id = create_document_id(
            pdf_path.name
        )

        print("\n" + "=" * 60)
        print(f"Processing: {pdf_path.name}")
        print(f"Document ID: {document_id}")
        print("=" * 60)

        pages = load_pdf(str(pdf_path))

        chunks = chunk_text(
            pages,
            chunk_size=500,
            chunk_overlap=100,
            source=pdf_path.name,
        )

        # Add document identity to every chunk
        for chunk in chunks:
            chunk["document_id"] = document_id
            chunk["document"] = pdf_path.name

        print(f"Created {len(chunks)} chunks.")

        all_chunks.extend(chunks)

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)

    print(f"Documents: {len(pdf_files)}")
    print(f"Total chunks: {len(all_chunks)}")

    embedder = Embedder()

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = embedder.embed_texts(texts)

    print(f"Embedding shape: {embeddings.shape}")

    vector_store = VectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add(
        embeddings,
        all_chunks,
    )

    vector_store.save(VECTOR_STORE_PATH)

    print(
        f"\nVector store saved to: "
        f"{VECTOR_STORE_PATH}"
    )


if __name__ == "__main__":
    ingest()