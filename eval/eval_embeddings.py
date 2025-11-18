import math
from typing import List, Tuple

from app.embeddings import embed_texts


def cosine_sim(v1: List[float], v2: List[float]) -> float:
    dot = sum(a*b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a*a for a in v1))
    n2 = math.sqrt(sum(b*b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def main():
    similar_pairs: List[Tuple[str, str]] = [
        ("A RAG rendszerek dokumentumokból keresnek információt.", "A RAG külső dokumentumok alapján válaszol."),
        ("Az embedding vektorokat készít a szövegből.", "A szöveg vektorrá alakítása embeddinggel történik."),
    ]

    dissimilar_pairs: List[Tuple[str, str]] = [
        ("A RAG rendszerek dokumentumokból keresnek információt.", "A macskák szeretnek aludni."),
        ("Az embedding vektorokat készít a szövegből.", "Ma esik az eső."),
    ]

    texts = [t for pair in similar_pairs + dissimilar_pairs for t in pair]
    vectors = embed_texts(texts)
    text_to_vec = {t: v for t, v in zip(texts, vectors)}

    sims_similar = []
    for a, b in similar_pairs:
        sims_similar.append(cosine_sim(text_to_vec[a], text_to_vec[b]))

    sims_dissimilar = []
    for a, b in dissimilar_pairs:
        sims_dissimilar.append(cosine_sim(text_to_vec[a], text_to_vec[b]))

    avg_similar = sum(sims_similar) / len(sims_similar) if sims_similar else 0.0
    avg_dissimilar = sum(sims_dissimilar) / len(sims_dissimilar) if sims_dissimilar else 0.0

    print("=== Embedding modell teszt ===")
    print(f"Átlagos hasonlóság 'hasonló' párokra:   {avg_similar:.3f}")
    print(f"Átlagos hasonlóság 'nem hasonló' párokra: {avg_dissimilar:.3f}")


if __name__ == "__main__":
    main()
