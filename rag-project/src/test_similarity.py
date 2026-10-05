from embeddings.embedder import Embedder
import numpy as np


embedder = Embedder()


sentences = [
    "What is the purpose of document retrieval?",
    "Why do we retrieve relevant documents?",
    "The weather is very cold today.",
]


embeddings = embedder.embed_texts(sentences)


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


similarity_01 = cosine_similarity(
    embeddings[0],
    embeddings[1],
)

similarity_02 = cosine_similarity(
    embeddings[0],
    embeddings[2],
)


print("Similarity between sentence 1 and 2:")
print(similarity_01)

print("\nSimilarity between sentence 1 and 3:")
print(similarity_02)