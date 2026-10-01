import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL, VECTOR_STORE_DIR, MAX_CONTEXT_CHUNKS
from app.ingestion import chunk_text, load_documents, save_chunks


class RAGIndex:
    def __init__(self, docs_dir: str, vector_store_dir: str):
        self.docs_dir = docs_dir
        self.vector_store_dir = Path(vector_store_dir)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.documents: List[str] = []
        self.metadata_path = self.vector_store_dir / "metadata.json"
        self.index_path = self.vector_store_dir / "index.faiss"
        self.load_or_build()

    def load_or_build(self):
        if self.index_path.exists() and self.metadata_path.exists():
            self.load()
            return

        text_documents = load_documents(self.docs_dir)
        if not text_documents:
            self.documents = []
            self.index = faiss.IndexFlatL2(384)
            return

        chunks: List[str] = []
        for doc in text_documents:
            chunks.extend(chunk_text(doc))

        if not chunks:
            self.documents = []
            self.index = faiss.IndexFlatL2(384)
            return

        self.documents = chunks
        save_chunks(self.documents, str(self.metadata_path))
        self.index = self._build_index(self.documents)

    def _build_index(self, chunks: List[str]) -> faiss.IndexFlatL2:
        embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.asarray(embeddings, dtype="float32"))
        faiss.write_index(index, str(self.index_path))
        return index

    def load(self):
        try:
            self.documents = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except Exception:
            self.documents = []

        self.index = faiss.read_index(str(self.index_path)) if self.index_path.exists() else None

        if self.index is None or not self.documents:
            self.index = faiss.IndexFlatL2(384)

    def retrieve(self, query: str, top_k: int = MAX_CONTEXT_CHUNKS) -> List[Tuple[str, float]]:
        if self.index is None or not self.documents:
            return []

        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = self.index.search(np.asarray(query_embedding, dtype="float32"), min(top_k, len(self.documents)))

        results: List[Tuple[str, float]] = []
        for index, distance in zip(indices[0], distances[0]):
            if index < 0 or index >= len(self.documents):
                continue
            results.append((self.documents[int(index)], float(distance)))
        return results
