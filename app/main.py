import json
from typing import List, Tuple

import httpx

from app.config import LLM_MODEL, OLLAMA_BASE_URL, MAX_CONTEXT_CHUNKS
from app.ingestion import RAGIndex


class RAGChatbot:
    def __init__(self, docs_dir: str):
        self.index = RAGIndex(docs_dir, "data/vector_store")

    def build_prompt(self, query: str, context_chunks: List[Tuple[str, float]]) -> str:
        if not context_chunks:
            return f"Answer the user question using only general knowledge.\n\nUser question: {query}\n\nIf you do not know the answer, say you do not know."

        context_text = "\n\n".join(f"Context {idx + 1}: {chunk}" for idx, (chunk, _) in enumerate(context_chunks))
        return (
            "You are a helpful assistant. Answer the user question using the provided context. "
            "If the answer is not in the context, say that the answer is not available in the provided documents.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {query}"
        )

    def ask(self, question: str) -> str:
        context_chunks = self.index.retrieve(question, top_k=MAX_CONTEXT_CHUNKS)
        prompt = self.build_prompt(question, context_chunks)

        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            return payload.get("response", "I could not generate a response.")
        except Exception as exc:
            return f"Error connecting to Ollama. Please make sure Ollama is running and the model '{LLM_MODEL}' is downloaded. Details: {exc}"
