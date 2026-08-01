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

    def build_bm25_index(self, chunks: List[Dict[str, Any]], save_bm25: bool = True):
        """Builds BM25Okapi index over whitespace + lowercase tokenized chunks."""
        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [c.get("chunk_text", "").lower().split() for c in chunks]
            # Handle empty/short documents gracefully
            if not tokenized_corpus or all(len(doc) == 0 for doc in tokenized_corpus):
                self.bm25_index = None
                return
            self.bm25_index = BM25Okapi(tokenized_corpus)
            if save_bm25:
                bm25_path = "bm25.pkl"
                with open(bm25_path, "wb") as f:
                    pickle.dump(self.bm25_index, f)
        except Exception as e:
            logger.warning(f"Could not build BM25 index: {e}")
            self.bm25_index = None

    def bm25_retrieve(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Keyword search using cached BM25Okapi index."""
        if not chunks:
            return []

        # Load BM25 index if not in memory
        if getattr(self, "bm25_index", None) is None:
            bm25_path = "bm25.pkl"
            if os.path.exists(bm25_path):
                try:
                    with open(bm25_path, "rb") as f:
                        self.bm25_index = pickle.load(f)
                except Exception as e:
                    logger.warning(f"Could not load bm25.pkl: {e}")
                    self.build_bm25_index(chunks, save_bm25=False)
            else:
                self.build_bm25_index(chunks, save_bm25=False)

        if getattr(self, "bm25_index", None) is None:
            return []

        tokenized_query = query.lower().split()
        if not tokenized_query:
            return []

        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((chunks[idx], float(scores[idx])))
        return results

    def hybrid_retrieve(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        chunk_vectors: np.ndarray, 
        top_k: int = 8, 
        rrf_k: int = 60
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Hybrid Retrieval merging Dense Embedding Search and BM25 Keyword Search
        via Reciprocal Rank Fusion (RRF):
        score(chunk) = sum(1 / (rrf_k + rank_in_list))
        """
        if not chunks:
            return []

        fetch_k = max(top_k * 2, len(chunks))

        # 1. Dense retrieval candidates
        dense_results = self.retrieve_top_k(query, chunks, chunk_vectors, top_k=fetch_k)

        # 2. BM25 keyword retrieval candidates
        bm25_results = self.bm25_retrieve(query, chunks, top_k=fetch_k)

        # Map chunk tags to chunks and calculate RRF scores
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # Process Dense Ranks (1-indexed)
        for rank, (chunk, _) in enumerate(dense_results, start=1):
            tag = chunk.get("tag", f"{chunk.get('doc_id')}:{chunk.get('chunk_id')}")
            chunk_map[tag] = chunk
            rrf_scores[tag] = rrf_scores.get(tag, 0.0) + (1.0 / (rrf_k + rank))

        # Process BM25 Ranks (1-indexed)
        for rank, (chunk, _) in enumerate(bm25_results, start=1):
            tag = chunk.get("tag", f"{chunk.get('doc_id')}:{chunk.get('chunk_id')}")
            chunk_map[tag] = chunk
            rrf_scores[tag] = rrf_scores.get(tag, 0.0) + (1.0 / (rrf_k + rank))

        # Sort deduplicated chunks by highest fused RRF score
        sorted_tags = sorted(rrf_scores.keys(), key=lambda t: rrf_scores[t], reverse=True)[:top_k]

        fused_results = [(chunk_map[tag], rrf_scores[tag]) for tag in sorted_tags]
        return fused_results

