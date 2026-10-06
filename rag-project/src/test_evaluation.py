from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.reranker import Reranker
from retrieval.evaluator import RetrievalEvaluator
from retrieval.evaluation_dataset import (
    EVALUATION_DATASET,
)
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTOR_STORE_PATH = (
    PROJECT_ROOT
    / "data"
    / "vector_store"
)


def main():

    print("Loading vector store...")

    vector_store = VectorStore.load(
        VECTOR_STORE_PATH
    )

    print("Loading embedding model...")

    embedder = Embedder()

    print("Loading reranker...")

    reranker = Reranker()

    evaluator = RetrievalEvaluator(
        vector_store=vector_store,
        embedder=embedder,
        reranker=reranker,
    )

    print("\nRunning retrieval evaluation...")

    results = evaluator.evaluate(
        dataset=EVALUATION_DATASET,
        k=5,
    )

    successful = sum(
        1
        for result in results
        if result["relevant_found"]
    )

    total = len(results)

    recall = successful / total

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    for result in results:

        status = (
            "PASS"
            if result["relevant_found"]
            else "FAIL"
        )

        print(
            f"{status} | "
            f"{result['question']}"
        )

    print("\n" + "=" * 60)

    print(
        f"Recall@5: "
        f"{recall:.2%}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()