import os
import sys
import json
import argparse
import numpy as np
from src.embeddings import EmbeddingEngine
from src.agent import ResearchAgent

INDEX_FILE = "index.json"

def load_index():
    if not os.path.exists(INDEX_FILE):
        print(f"Error: Index file '{INDEX_FILE}' not found. Please run 'python ingest.py' first.")
        sys.exit(1)
    
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    chunks = data["chunks"]
    embeddings = np.array(data["embeddings"], dtype=np.float32)
    return chunks, embeddings

def query_agent(question: str, top_k: int = 5):
    chunks, embeddings = load_index()
    embedder = EmbeddingEngine()
    
    # Cosine similarity retrieval
    retrieved = embedder.retrieve_top_k(question, chunks, embeddings, top_k=top_k)
    
    agent = ResearchAgent()
    response = agent.answer_question(question, retrieved)
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Query Research Agent with inline bracket citations")
    parser.add_argument("question", type=str, help="The research question to answer")
    parser.add_argument("--k", type=int, default=5, help="Number of top chunks to retrieve (default: 5)")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f" QUESTION: {args.question}")
    print(f"=======================================================\n")
    
    result = query_agent(args.question, top_k=args.k)
    
    print(result["answer"])
    print(f"\n-------------------------------------------------------")
    print(f" [POST-PROCESS VERIFICATION SUMMARY]")
    print(f" Status:              {result['verification']['status']}")
    print(f" Citation Density:    {result['verification']['citation_density'] * 100:.1f}%")
    print(f" Marker Drop Rate:    {result['verification']['drop_rate'] * 100:.1f}%")
    print(f" Repair Pass Applied: {result['repaired']}")
    print(f" Total Markers Found: {len(result['verification']['all_markers_found'])}")
    print(f"-------------------------------------------------------\n")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python ask.py \"What was Q3 revenue growth?\" [--k 5]")
        sys.exit(1)
    main()
