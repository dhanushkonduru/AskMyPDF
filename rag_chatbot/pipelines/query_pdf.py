import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
from typing import List

from utils.groq_client import get_groq_client

# Import QdrantVectorStore if available in your project or from a library
from langchain.vectorstores import Qdrant as QdrantVectorStore

EMBED_DIM = 384

def ask_pdf(question: str, collections: list, top_k=6) -> str:
    from operator import itemgetter

    qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    q_emb = model.encode(question)

    all_hits = []
    for collection in collections:
        hits = qdrant.search(collection_name=collection, query_vector=q_emb, limit=top_k)
        all_hits.extend(hits)

    # Sort by score and pick top overall
    sorted_hits = sorted(all_hits, key=lambda h: h.score, reverse=True)[:top_k]

    context = "\n---\n".join([f"[{h.payload.get('source', '')}] {h.payload['text']}" for h in sorted_hits])

    prompt = f"""
You are an intelligent PDF assistant. Answer the following question using the information below.

Question: {question}

Relevant excerpts:
{context}

Provide a clear, complete and well-explained answer. If multiple points are relevant, summarize them all.
"""

    groq = get_groq_client()
    resp = groq.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content
