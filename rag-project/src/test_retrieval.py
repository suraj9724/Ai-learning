from pathlib import Path

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "documents" / "rag_sample_knowledge_base.pdf"

# --------------------------------------------------
# 1. Load PDF
# --------------------------------------------------

pages = load_pdf(PDF_PATH)


# --------------------------------------------------
# 2. Create chunks
# --------------------------------------------------

chunks = chunk_text(
    pages,
    chunk_size=500,
    chunk_overlap=100,
)

print(f"Total chunks: {len(chunks)}")


# --------------------------------------------------
# 3. Create embedding model
# --------------------------------------------------

embedder = Embedder()


# --------------------------------------------------
# 4. Generate chunk embeddings
# --------------------------------------------------

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedder.embed_texts(texts)


print(
    f"Embedding shape: {embeddings.shape}"
)


# --------------------------------------------------
# 5. Create FAISS vector store
# --------------------------------------------------

dimension = embeddings.shape[1]

vector_store = VectorStore(
    dimension=dimension
)


# --------------------------------------------------
# 6. Store embeddings
# --------------------------------------------------

vector_store.add(
    embeddings,
    chunks,
)


print("Embeddings added to FAISS.")


# --------------------------------------------------
# 7. Ask a question
# --------------------------------------------------

question = "What is the capital of India?"


# --------------------------------------------------
# 8. Embed the question
# --------------------------------------------------

query_embedding = embedder.embed_text(
    question
)


# --------------------------------------------------
# 9. Search
# --------------------------------------------------

results = vector_store.search(
    query_embedding,
    top_k=3,
)


# --------------------------------------------------
# 10. Display results
# --------------------------------------------------

print("\n" + "=" * 80)
print("SEARCH RESULTS")
print("=" * 80)


for i, result in enumerate(results, start=1):

    chunk = result["chunk"]

    print(f"\nRESULT {i}")
    print("-" * 80)

    print(
        f"Chunk ID: {chunk['chunk_id']}"
    )

    print(
        f"Page: {chunk['page_number']}"
    )

    print(
        f"Similarity: {result['similarity']:.4f}"
    )

    print("\nText:")

    print(chunk["text"])