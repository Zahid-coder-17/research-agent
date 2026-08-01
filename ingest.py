import os
import sys
import glob
import json
import numpy as np
from pathlib import Path
from src.chunker import DocumentChunker
from src.embeddings import EmbeddingEngine

INDEX_FILE = "index.json"

def read_file_content(file_path: str) -> str:
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(str(path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

def ingest_sources(file_patterns: list):
    files = []
    for pat in file_patterns:
        matched = glob.glob(pat)
        if matched:
            files.extend(matched)
        elif os.path.exists(pat):
            files.append(pat)

    files = sorted(list(set(files)))
    if not files:
        print("No files found matching the pattern.")
        return

    print(f"--- Ingesting {len(files)} Source Documents ---")
    chunker = DocumentChunker(min_chunk_words=150, max_chunk_words=400)
    all_chunks = []

    for doc_idx, file_path in enumerate(files, start=1):
        filename = os.path.basename(file_path)
        content = read_file_content(file_path)
        if not content.strip():
            print(f"Skipping empty or unreadable file: {filename}")
            continue

        doc_chunks = chunker.chunk_document(content, doc_id=doc_idx, doc_title=filename)
        print(f"  Doc S{doc_idx} [{filename}]: Generated {len(doc_chunks)} chunks.")
        all_chunks.extend(doc_chunks)

    if not all_chunks:
        print("No chunks were generated.")
        return

    print(f"\nBuilding BM25 index & computing vector embeddings for {len(all_chunks)} total chunks...")
    embedder = EmbeddingEngine()
    chunk_texts = [c["chunk_text"] for c in all_chunks]
    embeddings_matrix = embedder.embed_texts(chunk_texts, save_vectorizer=True)
    embedder.build_bm25_index(all_chunks, save_bm25=True)

    # Save to index.json
    index_data = {
        "chunks": all_chunks,
        "embeddings": embeddings_matrix.tolist(),
        "total_documents": len(files),
        "total_chunks": len(all_chunks)
    }

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    print(f"Ingestion Complete! Local index saved to '{INDEX_FILE}'.\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        file_args = ["sample_sources/*"]
    else:
        file_args = sys.argv[1:]
    ingest_sources(file_args)
