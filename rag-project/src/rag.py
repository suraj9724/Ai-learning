from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from generation.llm import LLM
from pathlib import Path
from retrieval.reranker import Reranker
from generation.context_builder import build_context
from retrieval.relevance import RelevanceGate
from retrieval.query_rewriter import QueryRewriter
from generation.citation_validator import CitationValidator

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


        print("Loading Reranker...")

        self.reranker = Reranker()

        print("Loading Relevance Gate...")

        self.relevance_gate = RelevanceGate(
            threshold=0.0
        )
        
        print("Loading query rewriter...")

        self.query_rewriter = QueryRewriter()

        print("Loading citation validator...")

        self.citation_validator = CitationValidator()

        print("RAG system ready!")


    def ask(
        self,
        question: str,
        top_k: int = 3,
        conversation: list[dict] | None = None,
        document_id: str | None = None
    ):

        # -------------------------------
        # 0. Rewrite question
        # -------------------------------

        search_query = self.query_rewriter.rewrite(
            question=question,
            conversation=conversation,
        )

        print(
            f"\nOriginal question: {question}"
        )

        print(
            f"Search query: {search_query}"
        )

        # -------------------------------
        # 1. Embed question
        # -------------------------------

        query_embedding = (
            self.embedder.embed_text(
                search_query
            )
        )


        # -------------------------------
        # 2. Retrieve chunks
        # -------------------------------

        candidate_results = self.vector_store.search(
            query_embedding,
            top_k=10,
            document_id=document_id,
        )

        # -------------------------------
        # 2.1 Rerank results
        # -------------------------------

        results = self.reranker.rerank(
            question,
            candidate_results,
            top_k=top_k,
        )
        
        # -----------------------------------------
        # 4. Apply relevance gate
        # -----------------------------------------

        results = self.relevance_gate.filter(
            results
        )
        
        if not results:

            return {
                "question": question,
                "answer": (
                    "I couldn't find relevant information "
                    "in the knowledge base to answer that."
                ),
        "sources": [],
    }

        # -----------------------------------------
        # DEBUG: Inspect retrieval pipeline
        # -----------------------------------------

        # print("\n" + "=" * 80)
        # print("CANDIDATE RESULTS FROM FAISS")
        # print("=" * 80)

        # for result in candidate_results:
        #     print(
        #         f"Similarity: "
        #         f"{result['similarity']:.4f}"
        #     )
        #     print(
        #         result["chunk"]["text"][:200]
        #     )
        #     print()

        # print("\n" + "=" * 80)
        # print("RERANKED RESULTS")
        # print("=" * 80)

        # for result in results:
        #     print(
        #         f"Similarity: "
        #         f"{result['similarity']:.4f}"
        #     )
        #     print(
        #         f"Rerank: "
        #         f"{result['rerank_score']:.4f}"
        #     )
        #     print(
        #         result["chunk"]["text"][:200]
        #     )
        #     print()
        
        # -------------------------------
        # 3. Build context
        # -------------------------------

        context = build_context(
            results,
            max_characters=6000,
        )


        # -------------------------------
        # 4. Generate answer
        # -------------------------------

        answer = self.llm.generate(
            question=question,
            context=context,
        )

        citation_validation = (
    self.citation_validator.validate(
        answer=answer,
        results=results,
    )
)
        # -------------------------------
        # 5. Build sources
        # -------------------------------

        sources = []

        for result in results:
            chunk = result["chunk"]

            sources.append({
                "chunk_id": chunk["chunk_id"],
                "document": chunk["document"],
                "page": chunk["page_number"],
                "section": chunk.get("section"),
                "similarity": round(result["similarity"], 4),
                "rerank_score": round(result["rerank_score"], 4),
            })


        return {
            "question": question,
            "search_query": search_query,
            "answer": answer,
            "sources": sources,
            "citation_validation": citation_validation,
        }