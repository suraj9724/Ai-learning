from pathlib import Path

from pypdf import PdfReader


def load_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a PDF and return one dictionary per page.

    Each page contains:
    - page_number
    - text
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    reader = PdfReader(path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append({
            "page_number": page_number,
            "text": text.strip(),
        })

    return pages