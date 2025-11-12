import os,glob,json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
# from fastapi import FastAPI
from pydantic import BaseModel  

app=FastAPI()


def load_corpus(corpus_dir = "docs"):   # Load text files from a directory into a dictionary and return it
    corpus = {}
    for filepath in glob.glob(os.path.join(corpus_dir, '*.txt')):
        with open(filepath, 'r', encoding='utf-8') as file:
            doc_id = os.path.basename(filepath)
            corpus[doc_id] = file.read()
    return corpus

def chunk_text(text, chunk_size=500, overlap=50): # Split text into overlapping chunks
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def build_chunks(corpus, chunk_size=500, overlap=50): # Build overlapping chunks for each document in the corpus
    chunked_corpus = []
    for doc_id, text in corpus.items():
        chunks = chunk_text(text, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            chunked_corpus.append({
                'doc_id': doc_id,
                'chunk_id': f"{doc_id}_chunk_{i}",
                'text': chunk
            })
    return chunked_corpus

model: SentenceTransformer | None = None
faiss_index: faiss.Index | None = None
index_metadata: list = []
index_dim: int = 0

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
def startup_build_index():
    global model, faiss_index, index_metadata, index_dim
    # load model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # load corpus and chunk
    corpus = load_corpus("docs")
    chunked = build_chunks(corpus, chunk_size=500, overlap=50)
    index_metadata = chunked
    if not chunked:
        faiss_index = None
        return
    texts = [c['text'] for c in chunked]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype('float32')
    # L2 normalize for cosine similarity using inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    embeddings = embeddings / norms
    index_dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(index_dim)  # inner product on normalized vectors = cosine similarity
    faiss_index.add(embeddings)

@app.post("/query")
async def query_top2(req: QueryRequest):
    global model, faiss_index, index_metadata, index_dim
    if faiss_index is None:
        return {"results": []}
    q = req.query
    q_emb = model.encode([q], convert_to_numpy=True)[0].astype('float32')
    # normalize
    q_norm = np.linalg.norm(q_emb)
    if q_norm == 0:
        q_norm = 1e-10
    q_emb = q_emb / q_norm
    q_emb = q_emb.reshape(1, -1)
    k = 2
    distances, indices = faiss_index.search(q_emb, k)
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        meta = index_metadata[idx]
        results.append({
            "doc_id": meta["doc_id"],
            "chunk_id": meta["chunk_id"],
            "text": meta["text"],
            "score": float(score)  # cosine similarity in [-1,1]
        })
    return {"results": results}

# optional health endpoint
@app.get("/health")
def health():
    return {"ready": faiss_index is not None}