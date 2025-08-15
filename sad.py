import sqlite3
import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ----------------------------
# CONFIG
# ----------------------------
DB_PATH = "embeddings.db"
MODEL_NAME = "all-MiniLM-L6-v2"  # your embedding model
TOP_K = 5
OPENROUTER_API_KEY = "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"
MODEL_ID = "deepseek/deepseek-r1-0528:free"

# ----------------------------
# INIT MODELS
# ----------------------------
model = SentenceTransformer(MODEL_NAME)

# ----------------------------
# Cosine Similarity
# ----------------------------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ----------------------------
# Get embedding for query
# ----------------------------
def get_embedding(text):
    return model.encode(text).tolist()

# ----------------------------
# Retrieve top K chunks
# ----------------------------
def retrieve_chunks(query, k=TOP_K):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query_embedding = get_embedding(query)

    cursor.execute("SELECT document_id, chunk_index, text, embedding FROM chunks")
    rows = cursor.fetchall()

    similarities = []
    for doc_id, chunk_idx, chunk_text, embedding_str in rows:
        try:
            embedding = json.loads(embedding_str)
        except json.JSONDecodeError:
            embedding = [float(x) for x in embedding_str.split(",")]

        sim = cosine_similarity(query_embedding, embedding)
        similarities.append((sim, doc_id, chunk_idx, chunk_text))

    similarities.sort(key=lambda x: x[0], reverse=True)
    top_chunks = similarities[:k]

    # Also fetch "nearby" chunks for better context
    context_chunks = []
    for _, doc_id, chunk_idx, chunk_text in top_chunks:
        context_chunks.append(chunk_text)
        cursor.execute("""
            SELECT text FROM chunks 
            WHERE document_id=? AND chunk_index BETWEEN ? AND ?
        """, (doc_id, chunk_idx - 1, chunk_idx + 1))
        nearby = [r[0] for r in cursor.fetchall()]
        context_chunks.extend(nearby)

    conn.close()
    seen = set()
    final_context = []
    for c in context_chunks:
        if c not in seen:
            seen.add(c)
            final_context.append(c)
    return final_context

# ----------------------------
# Ask OpenRouter
# ----------------------------
def ask_openrouter(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ----------------------------
# Main Question Answering
# ----------------------------
def answer_question(query):
    chunks = retrieve_chunks(query)
    context = "\n\n".join(chunks)
    prompt = f"""
Answer the following question using only the provided context.

Context:
{context}

Question: {query}

Answer:
    """
    return ask_openrouter(prompt)

# ----------------------------
# Main loop
# ----------------------------
if __name__ == "__main__":
    question = input("Ask a question: ")
    answer = answer_question(question)
    print("\n--- Answer ---\n")
    print(answer)
