import json
from pathlib import Path
from typing import List, Dict, Tuple

from app.ingestion import process_document
from app.vectordb import VectorStore

from .eval_retrieval import precision_at_k, recall_at_k, mrr, load_eval_cases


DOC_PATH = Path("data/raw/belivek_39-45.pdf")
EVAL_PATH = Path("data/eval/retrieval.json")

CHUNK_CONFIGS = [
    {"name": "small_chunks", "chunk_size": 200, "overlap": 50},
    {"name": "medium_chunks", "chunk_size": 500, "overlap": 100},
    {"name": "large_chunks", "chunk_size": 800, "overlap": 200},
]


def run_one_config(config: Dict) -> Tuple[float, float, float]:
    print(f"\n=== Chunk config: {config['name']} ===")
    chunks = process_document(DOC_PATH, chunk_size=config["chunk_size"], overlap=config["overlap"])
    print(f"Chunkok száma: {len(chunks)}")

    store = VectorStore()
    store.add_documents(chunks)

    cases = load_eval_cases(EVAL_PATH)
    precisions: List[float] = []
    recalls: List[float] = []
    mrrs: List[float] = []

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

    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    avg_r = sum(recalls) / len(recalls) if recalls else 0.0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0

    print(f"Átlag precision@5: {avg_p:.3f}")
    print(f"Átlag recall@5: {avg_r:.3f}")
    print(f"Átlag MRR: {avg_mrr:.3f}")

    return avg_p, avg_r, avg_mrr


def main():
    if not DOC_PATH.exists():
        print(f"Hiányzik a dokumentum: {DOC_PATH}")
        return
    if not EVAL_PATH.exists():
        print(f"Hiányzik a retrieval eval fájl: {EVAL_PATH}")
        return

    results = []
    for cfg in CHUNK_CONFIGS:
        p, r, m = run_one_config(cfg)
        results.append({"config": cfg, "precision": p, "recall": r, "mrr": m})

    print("\n=== Összefoglaló (chunking stratégia) ===")
    for r in results:
        name = r["config"]["name"]
        print(f"{name}: P@5={r['precision']:.3f}, R@5={r['recall']:.3f}, MRR={r['mrr']:.3f}")


if __name__ == "__main__":
    main()
