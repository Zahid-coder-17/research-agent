import os
import sys
import json
import unittest
import numpy as np

# Ensure root directory is on path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from src.chunker import DocumentChunker
from src.embeddings import EmbeddingEngine, SimpleTfidfVectorizer
from src.web_search import fetch_page_content, log_web_fetch, LOG_FILE
from verify import CitationVerifier, CITATION_REGEX
from config import SYSTEM_PROMPT, REPAIR_PROMPT_TEMPLATE

class TestDocumentChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = DocumentChunker(min_chunk_words=20, max_chunk_words=50)

    def test_sentence_splitting(self):
        text = "First sentence here. Second sentence follows! Third sentence is here?"
        sentences = self.chunker.split_into_sentences(text)
        self.assertEqual(len(sentences), 3)

    def test_chunking_and_tagging(self):
        doc_text = " ".join([f"Word{i} is in this sentence." for i in range(100)])
        chunks = self.chunker.chunk_document(doc_text, doc_id=1, doc_title="test_doc.md")
        self.assertGreater(len(chunks), 0)
        self.assertTrue(chunks[0]["tag"].startswith("[S1:"))
        self.assertIn("chunk_text", chunks[0])

class TestEmbeddingEngine(unittest.TestCase):
    def setUp(self):
        self.embedder = EmbeddingEngine()
        self.embedder.vectorizer = None
        self.sample_chunks = [
            {"doc_id": "S1", "chunk_id": "00", "tag": "[S1:00]", "doc_title": "Doc1", "chunk_text": "AES-256 encryption at rest"},
            {"doc_id": "S2", "chunk_id": "00", "tag": "[S2:00]", "doc_title": "Doc2", "chunk_text": "Net-Zero carbon emissions target year 2040"},
            {"doc_id": "S3", "chunk_id": "00", "tag": "[S3:00]", "doc_title": "Doc3", "chunk_text": "Q3 revenue growth reached 12 percent"}
        ]

    def test_vectorizer_normalization(self):
        vec = SimpleTfidfVectorizer()
        matrix = vec.fit_transform(["hello world", "test hello"])
        matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.linalg.norm(matrix, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_bm25_retrieval(self):
        self.embedder.build_bm25_index(self.sample_chunks, save_bm25=False)
        results = self.embedder.bm25_retrieve("AES-256 encryption", self.sample_chunks, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0]["tag"], "[S1:00]")

    def test_hybrid_rrf_retrieval(self):
        texts = [c["chunk_text"] for c in self.sample_chunks]
        vectors = self.embedder.embed_texts(texts, save_vectorizer=True)
        self.embedder.build_bm25_index(self.sample_chunks, save_bm25=False)
        
        fused = self.embedder.hybrid_retrieve("AES-256 encryption", self.sample_chunks, vectors, top_k=2, rrf_k=60)
        self.assertEqual(len(fused), 2)
        # Check top result is [S1:00]
        self.assertEqual(fused[0][0]["tag"], "[S1:00]")
        self.assertGreater(fused[0][1], 0.0)

class TestCitationVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = CitationVerifier()

    def test_regex_marker_matching(self):
        self.assertTrue(CITATION_REGEX.search("[S1:00]"))
        self.assertTrue(CITATION_REGEX.search("[W2:05]"))
        self.assertTrue(CITATION_REGEX.search("[S10:00:01]"))
        self.assertFalse(CITATION_REGEX.search("[X1:00]"))

    def test_verification_passing(self):
        text = "Revenue grew 12% YoY [S1:00]. Encryption uses AES-256 [S4:00].\nConfidence: Fully supported\nSources used:\n- [S1:00] doc1"
        res = self.verifier.verify_citations(text)
        self.assertTrue(res["is_verified"])
        self.assertEqual(res["citation_density"], 1.0)
        self.assertEqual(res["drop_rate"], 0.0)

    def test_verification_failing_missing_claim(self):
        text = "Revenue grew 12% YoY [S1:00]. Operating margin reached 21.5%.\nConfidence: Fully supported\nSources used:\n- [S1:00] doc1"
        res = self.verifier.verify_citations(text)
        self.assertFalse(res["is_verified"])
        self.assertEqual(len(res["uncited_sentences"]), 1)

class TestWebFetchLogging(unittest.TestCase):
    def test_fetch_logging(self):
        log_web_fetch("https://example.com/test", 200, 250, "Test Page")
        self.assertTrue(os.path.exists(LOG_FILE))
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        last = json.loads(lines[-1])
        self.assertEqual(last["url"], "https://example.com/test")
        self.assertEqual(last["word_count"], 250)

if __name__ == "__main__":
    unittest.main()
