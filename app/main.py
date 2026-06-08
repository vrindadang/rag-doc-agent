from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.ingestor import ingest_pdf, list_ingested_files
from app.retriever import retrieve_similar_chunks
from app.generator import stream_answer
import json

app = FastAPI(
    title="RAG Doc Agent (Azure OpenAI)",
    description="Upload PDFs, ask questions, get streamed answers with citations.",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 8


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def list_documents():
    """List all ingested documents."""
    return {"documents": list_ingested_files()}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload and index a PDF."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    result = ingest_pdf(await file.read(), filename=file.filename)
    return {"message": "Ingested successfully", "chunks_indexed": result["chunks_indexed"]}


@app.post("/query")
async def query(req: QueryRequest):
    """
    Query the knowledge base.
    Returns a Server-Sent Events stream.
    Each event: data: {"token": "..."}
    """
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    chunks = retrieve_similar_chunks(req.question, top_k=req.top_k)
    if not chunks:
        raise HTTPException(404, "No documents found. Please ingest a PDF first.")

    def event_stream():
        for token in stream_answer(req.question, chunks):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")