from ollama import Client


class QueryRewriter:

    def __init__(
        self,
        model: str = "llama3.2:3b",
    ):
        self.model = model

        self.client = Client(
            host="http://localhost:11434"
        )

    def rewrite(
        self,
        question: str,
        conversation: list[dict] | None = None,
    ) -> str:

        if not conversation:
            return question

        history = []

        for message in conversation:

            role = message["role"]
            content = message["content"]

            history.append(
                f"{role}: {content}"
            )

        conversation_text = "\n".join(
            history
        )

        prompt = f"""
You rewrite user questions for a
document retrieval system.

Convert the latest user question into a
standalone search query.

Use the conversation history to resolve:
- pronouns such as "it", "they", "that"
- references to previous topics
- follow-up questions

Do NOT answer the question.

Return ONLY the rewritten search query.

Conversation:
--------------------
{conversation_text}
--------------------

Latest user question:
{question}

Standalone search query:
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

        raw_query = response["message"]["content"].strip()
        cleaned_query = raw_query.strip('"\'`')
        if cleaned_query.lower().startswith("standalone search query:"):
            cleaned_query = cleaned_query[len("standalone search query:"):].strip().strip('"\'`')
        return cleaned_query or question