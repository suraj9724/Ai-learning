from pathlib import Path

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "documents" / "rag_sample_knowledge_base.pdf"


# 1. Load PDF
pages = load_pdf(PDF_PATH)

# 2. Create chunks
chunks = chunk_text(
    pages,
    chunk_size=500,
    chunk_overlap=100,
)

print(f"Total chunks: {len(chunks)}")


# 3. Create embedding model
embedder = Embedder()


# 4. Extract chunk text
texts = [chunk["text"] for chunk in chunks]


# 5. Generate embeddings
embeddings = embedder.embed_texts(texts)


print("\nEmbedding information:")
print(f"Shape: {embeddings.shape}")
print(f"Number of chunks: {len(embeddings)}")
print(f"Vector dimensions: {len(embeddings[0])}")


print("\nFirst embedding:")
print(embeddings[0])