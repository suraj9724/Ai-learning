def build_context(
    results: list[dict],
    max_characters: int = 6000,
) -> str:

    if not results:
        return ""

    context_parts = []
    current_length = 0

    for result in results:
        chunk = result["chunk"]

        section = chunk.get("section")

        chunk_context = f"""
[Source ID: {chunk["chunk_id"]}]
Document: {chunk["document"]}
Document ID: {chunk["document_id"]}
Page: {chunk["page_number"]}
Section: {section or "Unknown"}

{chunk["text"]}
"""

        if current_length + len(chunk_context) > max_characters:
            break

        context_parts.append(chunk_context)
        current_length += len(chunk_context)

    return "\n\n".join(context_parts)