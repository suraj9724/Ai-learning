from ollama import Client


class LLM:

    def __init__(
        self,
        model: str = "llama3.2:3b",
    ):
        self.model = model

        self.client = Client(
            host="http://localhost:11434"
        )

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:

        prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

Rules:

1. Do not use outside knowledge.
2. Do not make up information.
3. If the context does not contain enough information,
   say that you don't have enough information.
4. When making a factual claim, include the relevant
   source ID in brackets.
5. Use the format [Source ID: X].

Context:
--------------------
{context}
--------------------

Question:
{question}

Answer:
"""

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response["message"]["content"]