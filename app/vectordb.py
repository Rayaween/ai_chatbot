from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from app.embeddings import embed_texts

class VectorStore:
    def __init__(self, collection_name: str = "documents", dim: int = 1536):
        self.qdrant = QdrantClient(":memory:")
        self.collection_name = collection_name

        # gyűjtemény létrehozása, vagy ha már létezik, újbóli létrehozása
        self.qdrant.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        
    def add_documents(self, chunks: List[Dict]):
        texts = [c["text"] for c in chunks]
        vectors = embed_texts(texts)

        points: List[PointStruct] = []
        for c, v in zip(chunks, vectors):
            chunk_id = c["id"]
            payload = {
                "text": c ["text"],
                "source_file": c.get("source_file", None),
            }
            points.append(
                PointStruct(
                    id=chunk_id,
                    vector=v,
                    payload=payload,
                )
            )

        self.qdrant.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        [query_vec] = embed_texts([query])

        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            limit=top_k,
        )

        hits: List[Dict] = []
        for r in results:
            hits.append(
                {
                    "id": r.id,
                    "text": r.payload["text"],
                    "score": r.score,
                    "source_file": r.payload.get("source_file")
                }
            )
        return hits