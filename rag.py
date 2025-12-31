import os
import glob
import sqlite3
import numpy as np
import config

from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from functools import lru_cache

# Open SQLite connection
def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_DB_PATH, check_same_thread=False)
    return conn

# Create the docs table for first time
def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()

# Load embedding model once at startup
model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

@lru_cache(maxsize=512)
def _encode_single(text: str) -> np.ndarray:
     # Cache embeddings for repeated queries
    return model.encode([text], normalize_embeddings=True)[0]


def _encode(texts: List[str]) -> np.ndarray:
    # Encode a list of texts
    return np.vstack([_encode_single(t) for t in texts])


def index_docs():
     # Skip indexing if DB already has data
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM docs;")
    count = cur.fetchone()[0]
    if count > 0:
        conn.close()
        return
    # Load files from the knowledge folder
    paths = glob.glob(os.path.join(config.DOC_FOLDER, "*.md")) + \
            glob.glob(os.path.join(config.DOC_FOLDER, "*.txt"))
    all_texts = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        # Split long docs into smaller chunks    
        chunks = _chunk_text(text, chunk_size=400, overlap=50)
        all_texts.extend(chunks)

    if not all_texts:
        conn.close()
        return

    # Store embeddings in the database
    embs = _encode(all_texts).astype(np.float32)
    for text, emb in zip(all_texts, embs):
        cur.execute(
            "INSERT INTO docs (content, embedding) VALUES (?, ?);",
            (text, emb.tobytes()),
        )

    conn.commit()
    conn.close()
    _encode_single.cache_clear()
    retrieve_cached.cache_clear()

def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    # Simple chunking with a bit of overlap for context
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def retrieve(query: str, k: int) -> List[str]:
    # Score all chunks by similarity to the query
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT content, embedding FROM docs;")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return []

    q_emb = _encode([query])[0]
    scored: List[Tuple[float, str]] = []
    for content, emb_blob in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        score = float(np.dot(q_emb, emb))
        scored.append((score, content))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in scored[:k]]
    return top

@lru_cache(maxsize=256)
def retrieve_cached(query: str) -> Tuple[str, ...]:
    k = config.TOP_K
    # Cache retrieval results for repeated questions
    results = retrieve(query, k)
    return tuple(results)
