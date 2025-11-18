from __future__ import annotations

import time
from uuid import uuid4
from pathlib import Path
from typing import Optional, Iterator

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.ingestion import process_document
from app.vectordb import VectorStore
from app.rag import answer_question
from app.monitoring import log_request


store = VectorStore()
HAS_DOCS = False
SESSION_HISTORY: dict[str, list[dict]] = {}

app = FastAPI(
    title="RAG Asszisztens - Python verzió",
    description="Zárófeladat: RAG alapú asszisztens FastAPI backenddel",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    context: list[dict]
    monitoring: dict

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global HAS_DOCS

    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".txt", ".pdf"]:
        raise HTTPException(status_code=400, detail="Csak .txt vagy .pdf támogatott.")
    
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / file.filename
    raw_path.write_bytes(await file.read())

    chunks = process_document(raw_path)
    if not chunks:
        raise HTTPException(status_code=400, detail="Nem sikerült szöveget kinyerni a dokumentumból.")
    
    store.add_documents(chunks)
    HAS_DOCS = True

    return {
        "status": "ok",
        "filename": file.filename,
        "chunks_indexed": len(chunks),
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global HAS_DOCS

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Üres kérdés nem engedélyezett.")
    
    if not HAS_DOCS:
        raise HTTPException(
            status_code=400,
            detail="Még nincsenek indexelt dokumentumok. Először tölts fel egy TXT/PDF fájlt."
        )
    
    session_id = request.session_id or str(uuid4())
    history = SESSION_HISTORY.get(session_id, [])

    start_time = time.time()
    answer, context = answer_question(store, question, history=history)
    end_time = time.time()

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    SESSION_HISTORY[session_id] = history

    total_latency = end_time - start_time

    input_tokens_est = len(question.split())
    output_tokens_est = len(answer.split())

    metrics = log_request(
        endpoint = "/chat",
        session_id = session_id,
        question = question,
        answer = answer,
        context = context,
        input_tokens_est = input_tokens_est,
        output_tokens_est = output_tokens_est,
        total_latency_sec = total_latency,
        first_token_latency_sec = total_latency,
    )

    return ChatResponse(
        session_id = session_id,
        answer = answer,
        context = context,
        monitoring = metrics,
    )


@app.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    """
    Egyszerű streaming endpoint.
    Csak a választ streameli plain textként.
    """
    global HAS_DOCS

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Üres kérdés nem engedélyezett.")

    if not HAS_DOCS:
        raise HTTPException(
            status_code=400,
            detail="Még nincsenek indexelt dokumentumok. Először tölts fel egy TXT/PDF fájlt."
        )

    session_id = request.session_id or str(uuid4())
    history = SESSION_HISTORY.get(session_id, [])

    candidates = store.search(question, top_k=5)
    if not candidates:
        def gen_empty() -> Iterator[str]:
            yield "Nem találtam releváns információt a dokumentumokban."
        return StreamingResponse(gen_empty(), media_type="text/plain")

    from app.rag import rerank_by_llm, build_prompt

    contexts = rerank_by_llm(question, candidates, top_m=3)
    prompt = build_prompt(question, contexts, history=history)

    def token_generator() -> Iterator[str]:
        import time
        from app.openai_client import client
        from app.monitoring import log_request

        start_time = time.time()
        first_token_time = None
        answer_chunks = []

        stream = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            content = delta.content or ""
            if content:
                answer_chunks.append(content)
                if first_token_time is None:
                    first_token_time = time.time()
                yield content

        end_time = time.time()

        full_answer = "".join(answer_chunks)

        # session history frissítés
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": full_answer})
        SESSION_HISTORY[session_id] = history

        # metrikák becslése
        total_latency = end_time - start_time
        first_token_latency = (
            first_token_time - start_time if first_token_time is not None else None
        )

        input_tokens_est = len(question.split())
        output_tokens_est = len(full_answer.split())

        log_request(
            endpoint="/chat_stream",
            session_id=session_id,
            question=question,
            answer=full_answer,
            context=contexts,
            input_tokens_est=input_tokens_est,
            output_tokens_est=output_tokens_est,
            total_latency_sec=total_latency,
            first_token_latency_sec=first_token_latency,
        )

    return StreamingResponse(token_generator(), media_type="text/plain")
