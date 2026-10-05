from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from generation.llm import LLM
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTOR_STORE_PATH = (
    PROJECT_ROOT
    / "data"
    / "vector_store"
)


class RAG:

    def __init__(self):

        print("Loading embedding model...")

        self.embedder = Embedder()


        print("Loading vector store...")

        self.vector_store = VectorStore.load(
            VECTOR_STORE_PATH
        )


        print("Loading LLM...")

        self.llm = LLM()


        print("RAG system ready!")


    def ask(
        self,
        question: str,
        top_k: int = 3,
    ):

        # -------------------------------
        # 1. Embed question
        # -------------------------------

        query_embedding = (
            self.embedder.embed_text(
                question
            )
        )


        # -------------------------------
        # 2. Retrieve chunks
        # -------------------------------

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )


        # -------------------------------
        # 3. Build context
        # -------------------------------

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


        # -------------------------------
        # 4. Generate answer
        # -------------------------------

        answer = self.llm.generate(
            question=question,
            context=context,
        )


        # -------------------------------
        # 5. Build sources
        # -------------------------------

        sources = []

        for result in results:

            chunk = result["chunk"]

            sources.append({
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "page": chunk["page_number"],
                "similarity": result[
                    "similarity"
                ],
            })


        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }