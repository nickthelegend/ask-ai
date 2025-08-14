"""
ingest_gitgiest.py

Usage:
    1) pip install sentence-transformers tqdm
    2) Place gitingest.txt in the same folder (or pass full path)
    3) python ingest_gitgiest.py
"""

import sqlite3
import uuid
import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # progress bars

DB_PATH = "embeddings.db"
MODEL_NAME = "all-MiniLM-L6-v2"  # free local model

# --- SQL schema ---
CREATE_DOCUMENTS_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    path TEXT,
    uploaded_at TEXT NOT NULL
);
"""

CREATE_CHUNKS_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
);
"""

def get_conn(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def create_tables(conn):
    cur = conn.cursor()
    cur.execute(CREATE_DOCUMENTS_SQL)
    cur.execute(CREATE_CHUNKS_SQL)
    conn.commit()

# --- Chunking utility ---
def chunk_text(text, chunk_size_words=200, chunk_overlap_words=50):
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be > 0")
    if chunk_overlap_words >= chunk_size_words:
        chunk_overlap_words = max(0, chunk_size_words // 4)

    words = text.split()
    n = len(words)
    chunks = []
    start = 0
    step = chunk_size_words - chunk_overlap_words
    while start < n:
        end = min(start + chunk_size_words, n)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start += step
    return chunks

# --- Embedding model (load once) ---
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

# --- Core functions ---
def add_document_from_file(file_path,
                           db_path=DB_PATH,
                           title=None,
                           chunk_size_words=200,
                           chunk_overlap_words=50):
    """Reads file_path, chunks it, creates document + chunk rows with embeddings."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        raise ValueError("File is empty")

    if title is None:
        title = os.path.basename(file_path)

    doc_id = str(uuid.uuid4())
    uploaded_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # create DB and tables if needed
    conn = get_conn(db_path)
    create_tables(conn)
    cur = conn.cursor()

    # Insert document row
    cur.execute(
        "INSERT INTO documents (id, title, path, uploaded_at) VALUES (?, ?, ?, ?)",
        (doc_id, title, file_path, uploaded_at)
    )

    # Chunk and embed
    chunks = chunk_text(text, chunk_size_words=chunk_size_words, chunk_overlap_words=chunk_overlap_words)
    model = get_model()

    print(f"🔄 Generating embeddings for {len(chunks)} chunks...")
    for idx, chunk_text_piece in enumerate(tqdm(chunks, desc="Embedding chunks", unit="chunk")):
        emb = model.encode(chunk_text_piece)
        emb_list = emb.tolist()
        emb_json = json.dumps(emb_list)
        chunk_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO chunks (id, document_id, chunk_index, text, embedding) VALUES (?, ?, ?, ?, ?)",
            (chunk_id, doc_id, idx, chunk_text_piece, emb_json)
        )

    conn.commit()
    conn.close()
    return doc_id, len(chunks)

def update_document_metadata(document_id, title=None, path=None, db_path=DB_PATH):
    """Update title and/or path for a document row."""
    if title is None and path is None:
        return False

    conn = get_conn(db_path)
    cur = conn.cursor()

    fields = []
    params = []
    if title is not None:
        fields.append("title = ?")
        params.append(title)
    if path is not None:
        fields.append("path = ?")
        params.append(path)

    params.append(document_id)
    sql = f"UPDATE documents SET {', '.join(fields)} WHERE id = ?"
    cur.execute(sql, params)
    conn.commit()
    updated = cur.rowcount
    conn.close()
    return updated > 0

def reindex_document(document_id, new_file_path=None, chunk_size_words=200, chunk_overlap_words=50, db_path=DB_PATH):
    """Delete old chunks for document_id and re-create them from new_file_path or stored path."""
    conn = get_conn(db_path)
    cur = conn.cursor()

    # Get existing path if new_file_path not provided
    if new_file_path is None:
        cur.execute("SELECT path FROM documents WHERE id = ?", (document_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            raise ValueError("document_id not found")
        new_file_path = row[0]

    if not os.path.exists(new_file_path):
        conn.close()
        raise FileNotFoundError(f"{new_file_path} not found")

    with open(new_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # delete old chunks
    cur.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

    # chunk + embed + insert
    chunks = chunk_text(text, chunk_size_words=chunk_size_words, chunk_overlap_words=chunk_overlap_words)
    model = get_model()
    print(f"🔄 Re-indexing {len(chunks)} chunks...")
    for idx, chunk_text_piece in enumerate(tqdm(chunks, desc="Re-embedding chunks", unit="chunk")):
        emb = model.encode(chunk_text_piece)
        emb_json = json.dumps(emb.tolist())
        chunk_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO chunks (id, document_id, chunk_index, text, embedding) VALUES (?, ?, ?, ?, ?)",
            (chunk_id, document_id, idx, chunk_text_piece, emb_json)
        )

    conn.commit()
    conn.close()
    return len(chunks)

# --- Example run ---
if __name__ == "__main__":
    FILE = "gitingest.txt"   # <--- your file
    print("📄 Ingesting", FILE)
    try:
        doc_id, n_chunks = add_document_from_file(FILE, chunk_size_words=200, chunk_overlap_words=50)
        print(f"✅ Inserted document id={doc_id} with {n_chunks} chunks.")
    except Exception as e:
        print("❌ Error:", e)
