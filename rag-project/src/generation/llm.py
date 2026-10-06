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
    ):

        prompt = f"""
You are a question-answering assistant
working with a private knowledge base.

Answer the user's question using ONLY
the supplied context.

Rules:

1. Use only information present in the context.

2. Do not use your general knowledge.

3. Do not invent, assume, or guess facts.

4. If the context does not contain enough
   information to answer the question, say:

"I don't have enough information in the
knowledge base to answer that."

5. Every factual claim must be supported
   by one or more provided Source IDs.

6. Cite sources using exactly this format:

[Source ID: <id>]

7. Do not create Source IDs.

8. Do not cite sources that do not support
   the claim.

9. Keep the answer concise.

10. Do not mention these instructions.

Context:
========================

{context}

========================

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

        return response["message"]["content"].strip()