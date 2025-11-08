import os,glob,json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_corpus(corpus_dir = "docs"):
    corpus = {}
    for filepath in glob.glob(os.path.join(corpus_dir, '*.txt')):
        with open(filepath, 'r', encoding='utf-8') as file:
            doc_id = os.path.basename(filepath)
            corpus[doc_id] = file.read()
    return corpus

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def build_chunks(corpus, chunk_size=500, overlap=50):
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