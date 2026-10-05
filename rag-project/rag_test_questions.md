# RAG Test Questions

Use these after the basic retrieval pipeline is working.

## Direct retrieval questions

1. What is Retrieval-Augmented Generation?
2. What are the major stages of a typical RAG pipeline?
3. Why is document ingestion required?
4. Why are documents divided into chunks?
5. What is the purpose of chunk overlap?
6. What is an embedding?
7. Why can embeddings find semantically similar text even when exact keywords differ?
8. What is FAISS used for?
9. What happens when a user submits a question to the retriever?
10. Why is retrieval quality important?

## Multi-step questions

11. Explain the flow from a PDF to a final RAG answer.
12. What happens if the correct information is not retrieved?
13. What are some ways to improve retrieval quality?
14. Why should document metadata be preserved?
15. How should a RAG application be evaluated?

## Questions for testing grounding

16. According to the knowledge base, what is chunk overlap used for?
17. According to the knowledge base, why can too much context be harmful?
18. What does the knowledge base recommend doing before introducing LangChain or LlamaIndex?

## Expected behavior

For every answer, the application should eventually be able to identify the source document and, ideally, the relevant page or chunk.
