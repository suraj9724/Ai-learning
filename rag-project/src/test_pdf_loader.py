from pathlib import Path
from ingestion.pdf_loader import load_pdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "documents" / "rag_sample_knowledge_base.pdf"

pages = load_pdf(PDF_PATH)

print(f"Total pages: {len(pages)}")

for page in pages:
    print("\n" + "=" * 80)
    print(f"PAGE {page['page_number']}")
    print("=" * 80)
    print(page["text"])