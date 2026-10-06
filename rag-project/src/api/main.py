import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Ensure the 'src' directory is in Python's import search path
# so that imports inside rag.py (e.g. `from embeddings...`) succeed seamlessly
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rag import RAG

# ---------------------------------------------------------------------------
# 1. FastAPI Application Initialization
# ---------------------------------------------------------------------------
# FastAPI is the core web framework. It creates an ASGI app instance that
# handles HTTP routing, request parsing, response serialization, and Swagger docs.
app = FastAPI(
    title="RAG Assistant API",
    description="FastAPI backend connecting a React frontend to the RAG pipeline.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# 2. CORS (Cross-Origin Resource Sharing) Middleware
# ---------------------------------------------------------------------------
# Why CORS?
# Browsers prevent web apps running on one origin (e.g. React at http://localhost:5173)
# from making HTTP requests to another origin (e.g. FastAPI at http://localhost:8000).
# Adding CORSMiddleware tells the browser: "Allow the React frontend to make requests here".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local development (or ["http://localhost:5173"])
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Allows headers like Content-Type
)

# ---------------------------------------------------------------------------
# 3. Initialize RAG System (Singleton)
# ---------------------------------------------------------------------------
# Loading the embedder, vector store, reranker, and LLM takes a few seconds.
# We initialize it ONCE at application startup so that incoming API requests
# get answered instantly without reloading the models on every request.
print("[Server Startup] Initializing RAG pipeline...")
rag = RAG()
print("[Server Startup] RAG pipeline is ready!")

# ---------------------------------------------------------------------------
# 4. Pydantic Models for Data Validation
# ---------------------------------------------------------------------------
# What is Pydantic?
# Pydantic validates incoming JSON data automatically.
# If a client sends an invalid payload (e.g. missing question or wrong data type),
# FastAPI automatically returns an HTTP 422 Unprocessable Entity error with details,
# without us needing to write dozens of manual `if` checks.

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="The message content")

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's query")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of source chunks to retrieve")
    conversation: Optional[List[ChatMessage]] = Field(default=None, description="Previous conversation turns for context")
    document_id: Optional[str] = Field(default=None, description="Optional document filter")

class QueryResponse(BaseModel):
    question: str
    search_query: str
    answer: str
    sources: List[Dict[str, Any]]
    citation_validation: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------------
# 5. API Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Simple root endpoint to confirm API is running."""
    return {
        "status": "online",
        "message": "Welcome to RAG Assistant API! Visit /docs for Swagger documentation.",
    }

@app.get("/api/health")
def health_check():
    """Health check endpoint to verify backend status from frontend."""
    return {"status": "ok", "service": "rag-backend"}

@app.post("/api/ask", response_model=QueryResponse)
def ask_question(payload: QueryRequest):
    """
    Main endpoint for frontend:
    1. Receives question, top_k, and optional conversation history.
    2. Calls rag.ask() which rewrites the query, searches Faiss, reranker, context builder,
       and generates answer via LLM with citation validation.
    3. Returns the structured answer and sources back as JSON.
    """
    try:
        conv_list = None
        if payload.conversation:
            conv_list = [msg.model_dump() for msg in payload.conversation]

        result = rag.ask(
            question=payload.question,
            top_k=payload.top_k,
            conversation=conv_list,
            document_id=payload.document_id,
        )

        return result

    except Exception as e:
        print(f"Error during rag.ask execution: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating the answer: {str(e)}",
        )

# ---------------------------------------------------------------------------
# 6. Direct Runner Support
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # Start the server on http://localhost:8000
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
