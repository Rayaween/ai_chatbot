import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests

from eval.eval_prompt import judge_answer

API_BASE = "http://127.0.0.1:8000"


def load_scenarios(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))

def upload_document_via_api(doc_path: Path) -> bool:
    files = {"file": (doc_path.name, doc_path.read_bytes())}
    resp = requests.post(f"{API_BASE}/upload", files=files, timeout=60)
    if resp.status_code == 200:
        data = resp.json()
        print(
            f"  [UPLOAD] OK: {data['filename']} - {data['chunks_indexed']} chunk indexelve."
        )
        return True
    else:
        print(f"  [UPLOAD] HIBA ({resp.status_code}): {resp.text}")
        return False
    
def chat_via_api(question: str, session_id: Optional[str],
) -> tuple[Optional[str], Optional[str], float]:
    payload = {"question": question, "session_id": session_id}
    start = time.time()
    resp = requests.post(f"{API_BASE}/chat", json=payload, timeout=120)
    end = time.time()
    latency = end - start

    if resp.status_code == 200:
        data = resp.json()
        answer = data["answer"]
        new_session_id = data["session_id"]
        return answer, new_session_id, latency
    else:
        print(f"  [CHAT] HIBA ({resp.status_code})? {resp.text}")
        return None, session_id, latency
    
def main():
    eval_path = Path("data/eval/app.json")
    if not eval_path.exists():
        print(f"Hiányzik az app-szintű eval fájl: {eval_path}")
        return
    
    scenarios = load_scenarios(eval_path)
    print(f"{len(scenarios)} alkalmazás szintű scenario.")

    all_latencies: List[float] = []
    correctness_scores: List[float] = []

    for scenario in scenarios:
        sid = scenario["id"]
        doc_path = Path(scenario["doc_path"])
        questions = scenario["questions"]

        print(f"\n=== Scenarios: {sid} ===")

        if not doc_path.exists():
            print(f"  Dokumentum hiányzik: {doc_path}, kihagyjuk.")
            continue

        # Upload ha kell
        if scenario.get("upload", True):
            ok = upload_document_via_api(doc_path)
            if not ok:
                continue

        # Chat kérdések
        session_id: Optional[str] = None
        for q in questions:
            question = q["q"]
            gold = q["gold_answer"]
            max_latency = q.get("max_latency_sec", 10.0)

            print(f"  Kérdés: {question}")

            answer, session_id, latency = chat_via_api(question, session_id)
            all_latencies.append(latency)

            if answer is None:
                print("   Nincs válasz, kihagyjuk az értékelést.")
                continue

            print(f"   Latency: {latency:.2f} s (max: {max_latency:.2f} s)")

            # LLM-as-judge alapján correctness (0-1)
            scores = judge_answer(question, gold, answer)
            correctness = scores["correctness"]
            correctness_scores.append(correctness)

            print(f"   Correctness: {correctness:.2f}")

            if latency > max_latency:
                print("   ⚠️ FIGYELEM: a latency meghaladta a megengedett maximumot")
    
    if not all_latencies:
        print("Nem volt sikeres a hívás.")
        return
    
    avg_latency = sum(all_latencies) / len(all_latencies)
    avg_correctness = sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0

    print("\n=== Összefoglaló (alkalmazás-szint) ===")
    print(f"Átlag latency: {avg_latency:.2f} s")
    print(f"Átlag correctness: {avg_correctness:.3f}")

    # User satisfaction szimuláció
    avg_satisfaction = avg_correctness
    print(f"Átlag szimulált ügyfél elégedettség: {avg_satisfaction:.3f}")


if __name__ == "__main__":
    main()