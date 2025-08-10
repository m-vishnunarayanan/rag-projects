import os
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DOCUMENTS = [
    "RAG stands for Retrieval-Augmented Generation. It retrieves relevant text chunks and feeds them to a language model to ground answers in external knowledge.",
    "Cosine similarity is often used to measure closeness between embeddings. Higher cosine similarity means more relevant matches.",
    "Sentence-Transformers provides easy-to-use embedding models like 'all-MiniLM-L6-v2' that are fast and good for small RAG prototypes.",
]

class SimpleRAG:
    def __init__(self, documents: List[str], model_name: str="all-MiniLM-L6-v2"):
        self.docs = documents
        self.embedder = SentenceTransformer(model_name)
        self.doc_embeddings = self.embedder.encode(
            self.docs, convert_to_numpy=True, normalize_embeddings=True
        )

    
    def retrieve(self, query: str, k: int = 3)-> List[Tuple[int, float]]:
        q = self.embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        scores = self.doc_embeddings @ q
        top_idx = np.argsort(-scores)[:k]

        return [(int(i), float(scores[i])) for i in top_idx]

    def answer(self, query: str, k: int = 3, model: str="gpt-4o-mini") -> str:
        hits = self.retrieve(query, k=k)

        context = "\n\n---\n\n".join(self.docs[i] for i, _ in hits)

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context. If the answer isn't in the context, say you don't know."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]

        client = OpenAI()
        resp = client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content


if __name__ == "__main__":
    rag = SimpleRAG(DOCUMENTS)
    user_query = "What is RAG?"
    print(f"Question asked {user_query}")
    print(f"Result : {rag.answer(user_query)}")
