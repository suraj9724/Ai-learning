def chunk_text(
    pages: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    source: str = "unknown",
    section: str | None = None,
) -> list[dict]:

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        if not text:
            continue

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append({
                    "chunk_id": len(chunks),
                    "document": source,
                    "page_number": page_number,
                    "section": section,
                    "text": chunk,
                })

            if end >= len(text):
                break

            start = end - chunk_overlap

    return chunks