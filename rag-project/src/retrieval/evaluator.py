class RetrievalEvaluator:

    def __init__(
        self,
        vector_store,
        embedder,
        reranker,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ):
        query_embedding = (
            self.embedder.embed_text(question)
        )

        candidates = self.vector_store.search(
            query_embedding,
            top_k=10,
        )

        return self.reranker.rerank(
            question=question,
            results=candidates,
            top_k=top_k,
        )

    @staticmethod
    def contains_relevant_keywords(
        text: str,
        keywords: list[str],
    ) -> bool:

        text = text.lower()

        matches = 0

        for keyword in keywords:
            if keyword.lower() in text:
                matches += 1

        # Consider the chunk relevant when
        # at least half of the expected keywords
        # appear in it.
        required = max(
            1,
            len(keywords) // 2,
        )

        return matches >= required

    def evaluate(
        self,
        dataset: list[dict],
        k: int = 5,
    ):

        results = []

        for item in dataset:

            question = item["question"]
            keywords = item["relevant_keywords"]

            retrieved = self.retrieve(
                question,
                top_k=k,
            )

            relevant_found = any(
                self.contains_relevant_keywords(
                    result["chunk"]["text"],
                    keywords,
                )
                for result in retrieved
            )

            results.append({
                "question": question,
                "relevant_found": relevant_found,
                "retrieved_count": len(retrieved),
            })

        return results