import os
from typing import List

from app.openai_client import client

def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    if not texts:
        return []
    
    response = client.embeddings.create(
        model=model,
        input=texts,
    )

    vectors: List[List[float]] = [item.embedding for item in response.data]
    return vectors