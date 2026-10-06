import sys
from pathlib import Path
import uvicorn

# Ensure 'src' is in Python's search path
SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

if __name__ == "__main__":
    print("----------------------------------------------------------")
    print("Starting FastAPI RAG Backend on http://127.0.0.1:8000")
    print("Swagger UI Docs: http://127.0.0.1:8000/docs")
    print("Health check:    http://127.0.0.1:8000/api/health")
    print("----------------------------------------------------------")
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
