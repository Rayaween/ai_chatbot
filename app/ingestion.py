from pathlib import Path
from typing import List, Dict

def load_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        import fitz
        doc = fitz.open(path)
        pages = [page.get_text() for page in doc]
        return "\n".join(pages)
    else:
        raise ValueError(f"Nem támogatott fájltípus: {suffix}")
    
def simple_word_chunk(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    words = text.split()
    chunks: List[Dict] = []

    start = 0
    idx = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append(
            {
                "id": idx,
                "text": chunk_text,
            }
        )

        idx += 1
        if end == len(words):
            break

        start = max(0, end - overlap)

    return chunks

def process_document(
        path: Path,
        chunk_size: int =  500,
        overlap: int = 100,
) -> List[Dict]:
    text = load_text_from_file(path)
    chunks = simple_word_chunk(text, chunk_size=chunk_size, overlap=overlap)

    for c in chunks:
        c["source_file"] = path.name

    return chunks