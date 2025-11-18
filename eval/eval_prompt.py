import json
from pathlib import Path
from typing import Dict, List, Any

from app.openai_client import client

from app.ingestion import process_document
from app.vectordb import VectorStore
from app.rag import answer_question

def load_cases(path: Path) -> List[Dict[str,Any]]:
    return json.loads(path.read_text(encoding="utf-8"))

def judge_answer(
        question: str,
        gold: str,
        model_answer: str,
        model: str = "gpt-4.1-mini",
) -> Dict[str, float]:
    """
    LLM-as-judge:
    Visszaad egy dict-et: {"relevance": x, "hallucination": y, "correctness": z}
    mind 0-1 között.
    """

    prompt = f"""
Te egy értékelő vagy. 
A feladatod: értékeld egy asszisztens válaszát egy kérdésre az elvárt (gold) válasz alapján.

Add meg a pontos adatokat 0 és 1 között:

- "relevance": mennyire releváns a válasz a kérdéshez? (0 = egyáltalán nem, 1 = teljesen)
- "hallucination": mennyire tartalmaz a válasz olyan állítást, ami nincs benne a gold válaszban? (0 = semennyire, 1 = nagyon sok hallucináció)
- "correctness": mennyire egyezik a válasz a gold válasszal? (0 = hibás, 1 = nagyon pontosan egyezik.)

Csak egy JSON-t adj vissza, minden szöveg nélkül. Példa:
{{"relevance": 0.9, "hallucination": 0.1, "correctness": 0.85}}

KÉRDÉS:
{question}

GOLD VÁLASZ:
{gold}

ASSZISZTENS VÁLASZA: 
{model_answer}
"""
    
    response = client.chat.completions.create(
        model = model,
        messages = [{"role": "user", "content": prompt}],
        temperature = 0.0,
    )
    raw = response.choices[0].message.content.strip()

    try:
        scores = json.loads(raw)
        return {
            "relevance": float(scores.get("relevance", 0.0)),
            "hallucination": float(scores.get("hallucination", 0.0)),
            "correctness": float(scores.get("correctness", 0.0)),
        }
    except Exception:
        print("Nem sikert JSON-t parse-olni a judge válaszából:", raw)
        return {"relevance": 0.0, "hallucination": 1.0, "correctness": 0.0}
    
def main():
    eval_path = Path("data/eval/prompt.json")
    if not eval_path.exists():
        print(f"Hiányzik a tesztfájl: {eval_path}")
        return
    
    cases = load_cases(eval_path)
    print(f"{len(cases)} prompt-szintű teszteset.")

    relevance_scores = []
    halluc_scores = []
    correctness_scores = []

    for case in cases:
        cid = case["id"]
        question = case["question"]
        gold = case["gold_answer"]
        doc_path = Path(case["doc_path"])

        if not doc_path.exists():
            print(f"[{cid}] Hiányzik a doc: {doc_path}, kihagyva.")
            continue

        # indexelése adott dokumentummal
        chunks = process_document(doc_path)
        store = VectorStore()
        store.add_documents(chunks)

        # RAG rendszer válasza
        answer, _ctx = answer_question(store, question)

        # LLM-as-judge értékelése
        scores = judge_answer(question, gold, answer)
        relevance_scores.append(scores["relevance"])
        halluc_scores.append(scores["hallucination"])
        correctness_scores.append(scores["correctness"])

        print (
            f"[{cid}] relevance={scores['relevance']:.2f}, "
            f"hallucination={scores['hallucination']:.2f}, "
            f"correctness={scores['correctness']:.2f}"
        )

    if not relevance_scores:
        print("Nem volt értékelhető eset.")
        return
    
    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    print("\n=== Összefoglaló (prompt-szint) ===")
    print(f"Átlag relevance: {avg(relevance_scores):.3f}")
    print(f"Átlag hallucination: {avg(halluc_scores):.3f}")
    print(f"Átlag correctness: {avg(correctness_scores):.3f}")

if __name__ == "__main__":
    main()