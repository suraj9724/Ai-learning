import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        """
        Create a FAISS index using inner product (cosine similarity with normalized vectors).
        """

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(dimension)

        self.chunks = []

    def add(
        self,
        embeddings: np.ndarray,
        chunks: list[dict],
    ):
        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings must match number of chunks"
            )

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        similarity_threshold: float | None = None,
    ):
        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        query_embedding = query_embedding.reshape(
            1, -1
        )

        faiss.normalize_L2(query_embedding)

        similarities, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for similarity, index in zip(
            similarities[0],
            indices[0],
        ):
            if index == -1:
                continue

            similarity = float(similarity)

            if (
                similarity_threshold is not None
                and similarity < similarity_threshold
            ):
                continue

            results.append({
                "chunk": self.chunks[index],
                "similarity": similarity,
            })

        return results