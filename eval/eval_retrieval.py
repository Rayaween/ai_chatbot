import json
from pathlib import Path
from typing import List, Dict

from app.ingestion import process_document
from app.vectordb import VectorStore


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    r = retrieved[:k]
    if not r:
        return 0.0
    tp = sum(1 for x in r if x in relevant)
    return tp / len(r)

def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    r = retrieved[:k]
    tp = sum(1 for x in r if x in relevant)
    return tp / len(relevant)

def mrr(retrieved: List[str], relevant: List[str]) -> float:
    for i, rid in enumerate(retrieved):
        if rid in relevant:
            return 1.0 / (i + 1)
    return 0.0

def load_eval_cases(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data

def main():
    doc_path = Path("data/raw/belivek_39-45.pdf") #frissíteni ha más a forrás
    if not doc_path.exists():
        print(f"Hiányzik a dokumentum a korpuszhoz: {doc_path}")
        return
    
    print(f"Korpuszdokumentum feldolgozása: {doc_path}")
    chunks = process_document(doc_path)
    print(f"{len(chunks)} chunk elkészült.")

    store = VectorStore()
    store.add_documents(chunks)
    print("Vectortár feltöltve.")

    # Tesztesetek betöltése
    eval_path = Path("data/eval/retrieval.json") 
    if not eval_path.exists():
        print(f"Hiányzik a tesztfájl: {eval_path}")
        return
    
    cases = load_eval_cases(eval_path)
    print(f"{len(cases)} tesztesetet találtunk.")

    precisions = []
    recalls = []
    mrrs = []

    for case in cases:
        query = case["query"]
        relevant_ids = case["relevant_ids"]

        results = store.search(query, top_k=10)
        retrieved_ids = [r["id"] for r in results]

        p5 = precision_at_k(retrieved_ids, relevant_ids, k=5)
        r5 = recall_at_k(retrieved_ids, relevant_ids, k=5)
        rr = mrr(retrieved_ids, relevant_ids)

        precisions.append(p5)
        recalls.append(r5)
        mrrs.append(rr)

        print(f"Case {case['id']}: P@5={p5:.3f}, R@5={r5:.3f}, MRR={rr:.3f}")

    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    avg_r = sum(recalls) / len(recalls) if recalls else 0.0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0

    print("\n=== Összefoglaló ===")
    print(f"Átlag precision@5: {avg_p:.3f}")
    print(f"Átlag recalls@5: {avg_r:.3f}")
    print(f"Átlag MRR: {avg_mrr:.3f}")

if __name__ == "__main__":
    main()