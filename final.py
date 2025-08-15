"""
Standalone Gradio UI for Document Ingestion & Embedding
-------------------------------------------------------
Features:
- Upload .txt files and embed
- Paste text directly and embed
- Update existing documents from file or text
- Adjustable chunk size and overlap
- Stores in SQLite database
"""

import os
import uuid
import json
import tempfile
import sqlite3
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import gradio as gr

# ====== CONFIG ======
DB_PATH = "embeddings.db"
MODEL_NAME = "all-MiniLM-L6-v2"

# ====== DB SETUP ======
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

# ====== TEXT CHUNKING ======
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

# ====== EMBEDDING MODEL ======
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

# ====== CORE INGESTION ======
def add_document_from_file(file_path, title=None, chunk_size_words=200, chunk_overlap_words=50):
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

    conn = get_conn()
    create_tables(conn)
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO documents (id, title, path, uploaded_at) VALUES (?, ?, ?, ?)",
        (doc_id, title, file_path, uploaded_at)
    )

    chunks = chunk_text(text, chunk_size_words=chunk_size_words, chunk_overlap_words=chunk_overlap_words)
    model = get_model()

    for idx, chunk_text_piece in enumerate(tqdm(chunks, desc="Embedding chunks", unit="chunk")):
        emb = model.encode(chunk_text_piece)
        emb_json = json.dumps(emb.tolist())
        chunk_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO chunks (id, document_id, chunk_index, text, embedding) VALUES (?, ?, ?, ?, ?)",
            (chunk_id, doc_id, idx, chunk_text_piece, emb_json)
        )

    conn.commit()
    conn.close()
    return doc_id, len(chunks)

def add_document_from_text(text, title="pasted_text", chunk_size_words=200, chunk_overlap_words=50):
    if not text.strip():
        return "❌ No text provided"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    doc_id, n_chunks = add_document_from_file(tmp_path, title=title, chunk_size_words=chunk_size_words, chunk_overlap_words=chunk_overlap_words)
    os.remove(tmp_path)
    return f"✅ Added text as document ID {doc_id} with {n_chunks} chunks."

def list_documents():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, title FROM documents")
    docs = cur.fetchall()
    conn.close()
    return {f"{title} ({doc_id})": doc_id for doc_id, title in docs}

def reindex_document(document_id, new_file_path=None, chunk_size_words=200, chunk_overlap_words=50):
    conn = get_conn()
    cur = conn.cursor()

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

    cur.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

    chunks = chunk_text(text, chunk_size_words=chunk_size_words, chunk_overlap_words=chunk_overlap_words)
    model = get_model()
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

# ====== GRADIO HANDLERS ======
def handle_file_upload(file, chunk_size, chunk_overlap):
    if not file:
        return "❌ No file uploaded"
    doc_id, n_chunks = add_document_from_file(file.name, chunk_size_words=int(chunk_size), chunk_overlap_words=int(chunk_overlap))
    return f"✅ Added {file.name} as document ID {doc_id} with {n_chunks} chunks."

def handle_text_upload(text, title, chunk_size, chunk_overlap):
    return add_document_from_text(text, title=title, chunk_size_words=int(chunk_size), chunk_overlap_words=int(chunk_overlap))

def handle_update(doc_id, new_text, new_file, chunk_size, chunk_overlap):
    if new_file:
        n_chunks = reindex_document(doc_id, new_file_path=new_file.name, chunk_size_words=int(chunk_size), chunk_overlap_words=int(chunk_overlap))
        return f"🔄 Updated document {doc_id} with {n_chunks} chunks from new file."
    elif new_text.strip():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
            tmp.write(new_text)
            tmp_path = tmp.name
        n_chunks = reindex_document(doc_id, new_file_path=tmp_path, chunk_size_words=int(chunk_size), chunk_overlap_words=int(chunk_overlap))
        os.remove(tmp_path)
        return f"🔄 Updated document {doc_id} with {n_chunks} chunks from new text."
    else:
        return "❌ No new content provided."
# ====== Ensure tables exist before UI loads ======
conn = get_conn()
create_tables(conn)
conn.close()

# ====== BUILD GRADIO UI ======
with gr.Blocks() as demo:
    gr.Markdown("# 📄 Document Ingestion & Embedding")

    with gr.Tab("📂 Upload File"):
        file_input = gr.File(file_types=[".txt"], label="Upload a text file")
        chunk_size = gr.Number(value=200, label="Chunk Size (words)")
        chunk_overlap = gr.Number(value=50, label="Chunk Overlap (words)")
        file_btn = gr.Button("Ingest File")
        file_output = gr.Textbox(label="Result")
        file_btn.click(handle_file_upload, inputs=[file_input, chunk_size, chunk_overlap], outputs=file_output)

    with gr.Tab("📝 Paste Text"):
        text_input = gr.Textbox(lines=10, placeholder="Paste your text here...")
        text_title = gr.Textbox(value="pasted_text", label="Title")
        chunk_size2 = gr.Number(value=200, label="Chunk Size (words)")
        chunk_overlap2 = gr.Number(value=50, label="Chunk Overlap (words)")
        text_btn = gr.Button("Ingest Text")
        text_output = gr.Textbox(label="Result")
        text_btn.click(handle_text_upload, inputs=[text_input, text_title, chunk_size2, chunk_overlap2], outputs=text_output)

    with gr.Tab("♻️ Update Document"):
        doc_list = gr.Dropdown(choices=list_documents(), label="Select Document")
        update_text = gr.Textbox(lines=10, placeholder="Enter new text (or leave blank if uploading a file)")
        update_file = gr.File(file_types=[".txt"], label="Upload new file (optional)")
        chunk_size3 = gr.Number(value=200, label="Chunk Size (words)")
        chunk_overlap3 = gr.Number(value=50, label="Chunk Overlap (words)")
        update_btn = gr.Button("Update Document")
        update_output = gr.Textbox(label="Result")
        update_btn.click(handle_update, inputs=[doc_list, update_text, update_file, chunk_size3, chunk_overlap3], outputs=update_output)

demo.launch()