from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):

        print(
            f"Loading reranker: {model_name}"
        )

        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        question: str,
        results: list[dict],
        top_k: int = 3,
    ):

        if not results:
            return []

        pairs = []

        for result in results:

            chunk = result["chunk"]

            pairs.append([
                question,
                chunk["text"],
            ])

        scores = self.model.predict(
            pairs
        )

        reranked = []

        for result, score in zip(
            results,
            scores,
        ):

            reranked.append({
                **result,
                "rerank_score": float(score),
            })

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]