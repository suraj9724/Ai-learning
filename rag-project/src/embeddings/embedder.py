from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str):
        return self.model.encode(
            text,
            convert_to_numpy=True,
        )

    def embed_texts(self, texts: list[str]):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        )