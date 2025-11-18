import os
import json
from typing import List, Dict, Optional, Tuple

from app.vectordb import VectorStore
from app.openai_client import client


def build_prompt(question: str, contexts: List[Dict], history: Optional[List[Dict]]) -> str:
    
    history_text = ""
    if history:
        for h in history[-6:]:
            role_label = "Felhasználó" if h["role"] == "user" else "Asszisztens"
            history_text += f"{role_label}: {h['content']}\n"
    
    context_text = ""
    for idx, c in enumerate(contexts, start=1):
        src = c.get("source_file", "ismeretlen forrás")
        context_text += f"Részlet #{idx} (forrás: {src}):\n{c['text']}\n\n---\n\n"

    prompt = f"""
Te egy RAG-alapú asszisztens vagy. 
Csak a megadott kontextusból válaszolj.
Ha a válasz nincs benne a kontextusban, mond azt, hogy nem tudod. 

ELŐZMÉNYEK (beszélgetés):
{history_text}

KONTEXTUS:
{context_text}

KÉRDÉS:
{question}

VÁLASZ (magyarul, tömören, de érthetően):
"""
    return prompt.strip()

def rerank_by_llm(question:str, candidates: List[Dict], top_m: int = 3) -> List [Dict]:
    if not candidates:
        return []
        
    items = []
    for i, c in enumerate(candidates, start=1):
        text = c["text"]
        if len(text) > 400:
            text = text [:400] + "..."
        items.append({"id": i, "text": text})
        
    chunks_str = ""
    for it in items:
        chunks_str += f"ID: {it['id']}\nSzöveg: {it['text']}\n\n"

    prompt_rerank = f"""
    Értékeld az alábbi szövegrészletek relevanciáját a kérdéshez.

    KÉRDÉS:
    {question}

    SZÖVEGRÉSZLET:
    {chunks_str}

    Adj vissza egy JSON listát, ahol minden elem:
    {{"id": <ID>, "score": <0-1 közötti szám>}}

    Csak a JSON tömböt add vissza, semmi mást. 
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt_rerank}],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    try:
        scores = json.loads(raw)
        score_map = {int(item["id"]): float(item["score"]) for item in scores}
    except Exception:
        return candidates
        
    scored = []
    for i, c in enumerate(candidates, start =1):
        s =score_map.get(i,0.0)
        cc = dict(c)
        cc["rerank_score"] = s
        scored.append(cc)

    scored_sorted = sorted(scored, key=lambda x: x["rerank_score"], reverse=True)
    return scored_sorted[:top_m]
    

def answer_question(
        store: VectorStore,
        question: str,
        top_k: int = 5,
        use_chunks: int = 3,
        history: Optional[List[Dict]] = None,
        use_rerank: bool = True,
) -> Tuple[str, List[Dict]]:
    candidates = store.search(question, top_k=top_k)

    if not candidates:
        return "Nem találtam releváns információt a dokumentumokban.", []
    
    if use_rerank:
        contexts = rerank_by_llm(question, candidates, top_m=use_chunks)
    else:
        contexts = candidates[:use_chunks]

    prompt = build_prompt(question, contexts, history=history)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content.strip()
    return answer, contexts
