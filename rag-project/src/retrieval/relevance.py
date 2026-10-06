class RelevanceGate:

    def __init__(
        self,
        threshold: float = 0.0,
    ):
        self.threshold = threshold

    def filter(
        self,
        results: list[dict],
    ) -> list[dict]:

        relevant_results = []

        for result in results:

            score = result["rerank_score"]

            if score >= self.threshold:
                relevant_results.append(
                    result
                )

        return relevant_results