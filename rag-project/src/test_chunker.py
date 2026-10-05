from pathlib import Path

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "documents" / "rag_sample_knowledge_base.pdf"


pages = load_pdf(PDF_PATH)

chunks = chunk_text(
    pages,
    chunk_size=500,
    chunk_overlap=100,
)

print(f"Total pages: {len(pages)}")
print(f"Total chunks: {len(chunks)}")


for chunk in chunks:
    print("\n" + "=" * 80)
    print(
        f"CHUNK {chunk['chunk_id']} | "
        f"PAGE {chunk['page_number']}"
    )
    print("=" * 80)

    print(chunk["text"])