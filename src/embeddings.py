import os
import re
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("embeddings")

class SimpleTfidfVectorizer:
    """
    Pure Python & Numpy TF-IDF Vectorizer with n-grams and L2 normalization.
    Zero external dependencies beyond numpy.
    """
    def __init__(self, ngram_range=(1, 2)):
        self.ngram_range = ngram_range
        self.vocab = {}
        self.idf = np.array([], dtype=np.float32)

    def _extract_ngrams(self, text: str) -> List[str]:
        words = re.findall(r'\w+', text.lower())
        tokens = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(words) - n + 1):
                tokens.append(" ".join(words[i:i + n]))
        return tokens

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        doc_tokens = [self._extract_ngrams(t) for t in texts]
        unique_tokens = sorted(list(set(token for doc in doc_tokens for token in doc)))
        if not unique_tokens:
            return np.zeros((len(texts), 1), dtype=np.float32)

        self.vocab = {token: idx for idx, token in enumerate(unique_tokens)}
        num_docs = len(texts)
        doc_freq = np.zeros(len(unique_tokens), dtype=np.float32)
        
        for doc in doc_tokens:
            seen = set(doc)
            for token in seen:
                doc_freq[self.vocab[token]] += 1.0
                
        self.idf = (np.log((1.0 + num_docs) / (1.0 + doc_freq)) + 1.0).astype(np.float32)
        
        matrix = np.zeros((num_docs, len(unique_tokens)), dtype=np.float32)
        for d_idx, doc in enumerate(doc_tokens):
            for token in doc:
                if token in self.vocab:
                    matrix[d_idx, self.vocab[token]] += 1.0
                    
        tf = np.where(matrix > 0, 1.0 + np.log(np.maximum(matrix, 1.0)), 0.0).astype(np.float32)
        return tf * self.idf

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.vocab:
            return np.zeros((len(texts), 1), dtype=np.float32)
            
        doc_tokens = [self._extract_ngrams(t) for t in texts]
        num_docs = len(texts)
        matrix = np.zeros((num_docs, len(self.vocab)), dtype=np.float32)
        for d_idx, doc in enumerate(doc_tokens):
            for token in doc:
                if token in self.vocab:
                    matrix[d_idx, self.vocab[token]] += 1.0
        tf = np.where(matrix > 0, 1.0 + np.log(np.maximum(matrix, 1.0)), 0.0).astype(np.float32)
        return tf * self.idf


class EmbeddingEngine:
    """
    High-performance vector embedding engine supporting standalone Numpy TF-IDF cosine similarity
    and SentenceTransformers.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", force_tfidf: bool = True, vectorizer_path: str = "vectorizer.pkl"):
        self.model_name = model_name
        self.model = None
        self.use_fallback = True
        self.vectorizer_path = vectorizer_path
        self.vectorizer = None
        
        if not force_tfidf and os.environ.get("USE_SENTENCE_TRANSFORMERS") == "1":
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                self.use_fallback = False
            except Exception as e:
                logger.warning(f"SentenceTransformers failed: {e}. Using TF-IDF fallback.")

        if self.use_fallback and os.path.exists(self.vectorizer_path):
            try:
                with open(self.vectorizer_path, "rb") as f:
                    self.vectorizer = pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load saved vectorizer: {e}")
                self.vectorizer = None

    def embed_texts(self, texts: List[str], save_vectorizer: bool = False) -> np.ndarray:
        if not self.use_fallback and self.model is not None:
            embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return embeddings / norms
        else:
            if self.vectorizer is None:
                self.vectorizer = SimpleTfidfVectorizer()
                dense_matrix = self.vectorizer.fit_transform(texts)
            else:
                dense_matrix = self.vectorizer.transform(texts)

            if hasattr(dense_matrix, "toarray"):
                dense_matrix = dense_matrix.toarray()

            dense_matrix = np.asarray(dense_matrix, dtype=np.float32)

            if save_vectorizer:
                try:
                    with open(self.vectorizer_path, "wb") as f:
                        pickle.dump(self.vectorizer, f)
                except Exception as e:
                    logger.warning(f"Failed saving vectorizer: {e}")

            norms = np.linalg.norm(dense_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return dense_matrix / norms

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query], save_vectorizer=False)[0]

    @staticmethod
    def cosine_similarity(query_vector: np.ndarray, chunk_vectors: np.ndarray) -> np.ndarray:
        """Computes dot product assuming normalized vectors."""
        return np.dot(chunk_vectors, query_vector)

    def retrieve_top_k(self, query: str, chunks: List[Dict[str, Any]], chunk_vectors: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if len(chunks) == 0:
            return []
        
        query_vector = self.embed_query(query)
        
        # Auto self-healing dimension check: if loaded chunk vectors don't match query vector dimension, re-embed chunks
        if chunk_vectors.ndim != 2 or chunk_vectors.shape[1] != query_vector.shape[0]:
            logger.warning(f"Embedding dimension mismatch: chunks {chunk_vectors.shape} vs query {query_vector.shape}. Re-embedding chunks on the fly...")
            chunk_texts = [c.get("chunk_text", "") for c in chunks]
            chunk_vectors = self.embed_texts(chunk_texts, save_vectorizer=True)
            
        scores = self.cosine_similarity(query_vector, chunk_vectors)
        
        # Rank by highest score
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append((chunks[idx], float(scores[idx])))
        return results
