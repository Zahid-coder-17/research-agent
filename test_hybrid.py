import os
import sys
import json
import time
from ask import load_index
from src.embeddings import EmbeddingEngine

def run_hybrid_ab_test():
    """
    A/B Retrieval Test Script comparing Dense, BM25, and Hybrid (RRF) retrieval.
    Verifies that BM25 index is cached and Hybrid RRF correctly merges ranked lists.
    """
    print(f"\n=======================================================")
    print(f"  HYBRID RETRIEVAL A/B TEST (Dense vs BM25 vs Hybrid)")
    print(f"=======================================================\n")

    chunks, embeddings = load_index()
    embedder = EmbeddingEngine()

    # 1. Verify BM25 Cache loading latency
    start_cache_check = time.time()
    bm25_res = embedder.bm25_retrieve("AES-256", chunks, top_k=3)
    cache_latency_ms = (time.time() - start_cache_check) * 1000
    
    bm25_cached = os.path.exists("bm25.pkl")
    print(f"[CACHE CHECK] bm25.pkl exists: {bm25_cached} | Retrieval latency: {cache_latency_ms:.2f}ms")

    # 3 Target Test Questions
    test_questions = [
        {
            "id": "Q01 (Exact Proper Noun / Security Standard)",
            "query": "What encryption protocols and algorithms like AES-256 and TLS 1.3 are used for data at rest?"
        },
        {
            "id": "Q03 (Financial Metric / R&D Spending)",
            "query": "How much did Apex invest in Research & Development (R&D) during Q3?"
        },
        {
            "id": "Q02 (ESG / Net-Zero Policy)",
            "query": "What is Apex's target year to achieve Net-Zero carbon emissions?"
        }
    ]

    for q_data in test_questions:
        q_id = q_data["id"]
        query = q_data["query"]

        print(f"\n-------------------------------------------------------")
        print(f" TEST CASE: {q_id}")
        print(f" QUERY:     '{query}'")
        print(f"-------------------------------------------------------")

        dense_top = embedder.retrieve_top_k(query, chunks, embeddings, top_k=5)
        bm25_top = embedder.bm25_retrieve(query, chunks, top_k=5)
        hybrid_top = embedder.hybrid_retrieve(query, chunks, embeddings, top_k=5, rrf_k=60)

        dense_tags = [f"{c['tag']} (sim: {score:.3f})" for c, score in dense_top]
        bm25_tags = [f"{c['tag']} (bm25: {score:.3f})" for c, score in bm25_top]
        hybrid_tags = [f"{c['tag']} (rrf: {score:.4f})" for c, score in hybrid_top]

        print(f"\n  [1] DENSE RETRIEVAL (Cosine):")
        for rank, item in enumerate(dense_tags, 1):
            print(f"      Rank {rank}: {item}")

        print(f"\n  [2] BM25 RETRIEVAL (Okapi):")
        for rank, item in enumerate(bm25_tags, 1):
            print(f"      Rank {rank}: {item}")

        print(f"\n  [3] HYBRID RETRIEVAL (RRF k=60):")
        for rank, item in enumerate(hybrid_tags, 1):
            print(f"      Rank {rank}: {item}")

    print(f"\n=======================================================")
    print(f" A/B TEST COMPLETE")
    print(f"=======================================================\n")

if __name__ == "__main__":
    run_hybrid_ab_test()
