from rag import RAG


rag = RAG()


while True:

    question = input(
        "\nAsk a question (or type 'exit'): "
    )

    if question.lower() == "exit":
        break


    result = rag.ask(
        question,
        top_k=3,
    )


    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(result["answer"])


    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)


    for source in result["sources"]:

        print(
            f"\nChunk: {source['chunk_id']}"
        )

        print(
            f"Document: {source['source']}"
        )

        print(
            f"Page: {source['page']}"
        )

        print(
            f"Similarity: {source['similarity']:.4f}"
        )