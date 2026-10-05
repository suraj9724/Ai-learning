from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text

from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore

from generation.llm import LLM

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "documents" / "rag_sample_knowledge_base.pdf"


class RAG:

    def __init__(self):

        print("Loading document...")

        pages = load_pdf(PDF_PATH)

        print("Creating chunks...")

        self.chunks = chunk_text(
            pages,
            chunk_size=500,
            chunk_overlap=100,
            source=PDF_PATH.name,
        )

        print(
            f"Created {len(self.chunks)} chunks."
        )

        print("Loading embedding model...")

        self.embedder = Embedder()

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        print("Generating embeddings...")

        embeddings = self.embedder.embed_texts(
            texts
        )

        print("Creating vector store...")

        dimension = embeddings.shape[1]

        self.vector_store = VectorStore(
            dimension=dimension
        )

        self.vector_store.add(
            embeddings,
            self.chunks
        )

        print("Loading LLM...")

        self.llm = LLM()

        print("RAG system ready!")


    def ask(
        self,
        question: str,
        top_k: int = 3,
    ):

        # -----------------------------------------
        # 1. Convert question into embedding
        # -----------------------------------------

        query_embedding = self.embedder.embed_text(
            question
        )


        # -----------------------------------------
        # 2. Retrieve relevant chunks
        # -----------------------------------------

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )


        # -----------------------------------------
        # 3. Build context
        # -----------------------------------------

        context_parts = []

        for result in results:

            chunk = result["chunk"]

            context_parts.append(
                f"""
[Source ID: {chunk['chunk_id']}]

Document: {chunk['source']}
Page: {chunk['page_number']}

Content:
{chunk['text']}
"""
            )

        context = "\n\n".join(
            context_parts
        )

        # -----------------------------------------
        # 4. Generate answer
        # -----------------------------------------

        answer = self.llm.generate(
            question=question,
            context=context,
        )

        # -----------------------------------------
        # 5. Return answer + sources
        # -----------------------------------------

        sources = []

        for result in results:
            chunk = result["chunk"]

            sources.append({
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "page": chunk["page_number"],
                "similarity": result["similarity"],
            })

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }