import os
import shutil
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import db, ingestion, vectorstore, rag
from . import quiz as quiz_module

app = FastAPI(title="Study Assistant RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.on_event("startup")
def startup():
    db.init_db()


# ---------- Documents ----------

@app.post("/documents/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")
    doc_id = db.create_document(file.filename, "pdf")
    filepath = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    records = ingestion.process_pdf(filepath, doc_id, file.filename)
    if not records:
        db.delete_document_db(doc_id)
        raise HTTPException(status_code=400, detail="Could not extract any text from this PDF (it may be scanned/image-only).")
    vectorstore.add_records(records)
    return {"document_id": doc_id, "name": file.filename, "chunks_indexed": len(records)}


class YouTubeRequest(BaseModel):
    url: str
    title: Optional[str] = None


@app.post("/documents/upload-youtube")
async def upload_youtube(payload: YouTubeRequest):
    name = payload.title or payload.url
    doc_id = db.create_document(name, "youtube")
    try:
        records = ingestion.process_youtube(payload.url, doc_id, name)
    except Exception as e:
        db.delete_document_db(doc_id)
        raise HTTPException(status_code=400, detail=f"Could not process YouTube URL: {e}")
    if not records:
        db.delete_document_db(doc_id)
        raise HTTPException(status_code=400, detail="No transcript could be found for this video.")
    vectorstore.add_records(records)
    return {"document_id": doc_id, "name": name, "chunks_indexed": len(records)}


@app.get("/documents")
async def get_documents():
    return db.list_documents_db()


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    vectorstore.delete_document(document_id)
    db.delete_document_db(document_id)
    return {"status": "deleted"}


# ---------- Chat ----------

@app.post("/sessions")
async def new_session():
    return {"session_id": db.create_session()}


@app.get("/sessions/{session_id}/history")
async def session_history(session_id: str):
    return db.get_history(session_id, limit=200)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    document_ids: Optional[List[str]] = None  # optional: scope chat to specific docs


@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    try:
        return rag.chat(payload.session_id, payload.message, payload.document_ids)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---------- Quiz ----------

class QuizRequest(BaseModel):
    document_ids: Optional[List[str]] = None
    num_questions: int = 5
    difficulty: str = "medium"


@app.post("/quiz")
async def quiz_endpoint(payload: QuizRequest):
    try:
        return quiz_module.generate_quiz(payload.document_ids, payload.num_questions, payload.difficulty)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
