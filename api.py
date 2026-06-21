"""
FastAPI backend for RAG chatbot (optimized for Streamlit + Groq)
"""

import os
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.connections import get_llm, get_embeddings
from src.vector_store import load_vector_store
from src.retriever import get_answer
from src.ingest import ingest_documents

# --------------------------------------------------
# App setup
# --------------------------------------------------

app = FastAPI(
    title="RAG API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Globals (loaded once → FAST)
# --------------------------------------------------

vector_store = None
llm = None
embeddings = None

chat_history = []

# --------------------------------------------------
# Startup (CRITICAL for speed)
# --------------------------------------------------

@app.on_event("startup")
def startup():
    global vector_store, llm, embeddings

    embeddings = get_embeddings()
    llm = get_llm()

    try:
        vector_store = load_vector_store(
            embeddings=embeddings
        )
        print("✅ Vector store loaded")
    except Exception as e:
        print(f"⚠️ No index found yet: {e}")
        vector_store = None

# --------------------------------------------------
# Models
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    top_k: int = 4


class ChatResponse(BaseModel):
    question: str
    answer: str
    timestamp: str

# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

# --------------------------------------------------
# Chat endpoint
# --------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    global vector_store, llm

    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    if vector_store is None:
        raise HTTPException(503, "Vector store not initialized. Run /ingest first.")

    try:
        answer = get_answer(
            question=req.question,
            vector_store=vector_store,
            llm=llm,
            chat_history=chat_history,
            top_k=req.top_k
        )

    except Exception as e:
        raise HTTPException(500, str(e))

    ts = datetime.now().strftime("%H:%M:%S")

    chat_history.append({
        "user": req.question,
        "assistant": answer
    })

    return ChatResponse(
        question=req.question,
        answer=answer,
        timestamp=ts
    )

# --------------------------------------------------
# Ingest (SAFE + reload index)
# --------------------------------------------------

@app.post("/ingest")
def ingest():

    global vector_store, embeddings

    try:
        ingest_documents()

        vector_store = load_vector_store(
            embeddings=embeddings
        )

        return {
            "status": "ok",
            "message": "PDFs ingested and index rebuilt"
        }

    except Exception as e:
        raise HTTPException(500, str(e))

# --------------------------------------------------
# History
# --------------------------------------------------

@app.get("/history")
def history():
    return chat_history


@app.delete("/history")
def clear():
    chat_history.clear()
    return {"status": "cleared"}