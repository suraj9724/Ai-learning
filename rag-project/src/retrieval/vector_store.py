import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension: int | None = None):

        self.index = None
        self.chunks = []

        if dimension is not None:
            self.index = faiss.IndexFlatIP(dimension)

    def add(
        self,
        embeddings: np.ndarray,
        chunks: list[dict],
    ):

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings must match "
                "number of chunks"
            )

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        # Normalize vectors so inner product
        # behaves like cosine similarity.
        faiss.normalize_L2(embeddings)

        if self.index is None:
            self.index = faiss.IndexFlatIP(
                embeddings.shape[1]
            )

        self.index.add(embeddings)

        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        similarity_threshold: float | None = None,
    ):

        if self.index is None:
            raise ValueError(
                "Vector store has not been initialized."
            )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        query_embedding = query_embedding.reshape(
            1,
            -1,
        )

        faiss.normalize_L2(
            query_embedding
        )

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

    def save(self, directory: str):

        path = Path(directory)

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.index is None:
            raise ValueError(
                "Cannot save an empty vector store."
            )

        faiss.write_index(
            self.index,
            str(path / "index.faiss"),
        )

        with open(
            path / "chunks.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.chunks,
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, directory: str):

        path = Path(directory)

        index_path = path / "index.faiss"
        chunks_path = path / "chunks.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {chunks_path}"
            )

        store = cls()

        store.index = faiss.read_index(
            str(index_path)
        )

        with open(
            chunks_path,
            "r",
            encoding="utf-8",
        ) as file:

            store.chunks = json.load(file)

        return store