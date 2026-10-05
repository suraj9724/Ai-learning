# RAG Project

## Purpose

This project is our first practical AI/ML project focused on Retrieval-Augmented Generation (RAG).

We will build the core pipeline ourselves before introducing higher-level frameworks.

## Initial stack

- Python
- FastAPI
- PyPDF
- Sentence Transformers
- FAISS
- OpenAI API
- python-dotenv

## Initial data

Place the provided knowledge-base PDF inside:

```text
data/documents/rag_sample_knowledge_base.pdf
```

## Pipeline

```text
PDF
  -> Text extraction
  -> Chunking
  -> Embeddings
  -> FAISS vector index
  -> Query embedding
  -> Similarity retrieval
  -> Context construction
  -> LLM
  -> Answer + sources
```

## Development order

1. PDF text extraction
2. Chunking
3. Embedding generation
4. FAISS indexing
5. Similarity search
6. RAG prompt construction
7. LLM generation
8. Source/citation handling
9. FastAPI endpoint
10. Evaluation
11. Retrieval improvements
12. Frontend

## Important rule

Do not add LangChain or LlamaIndex at the beginning. First understand and implement the core RAG flow directly.

## First success condition

Given a question such as:

> What is the purpose of chunk overlap?

the system should retrieve the relevant chunk from the PDF and use it to generate a grounded answer.
