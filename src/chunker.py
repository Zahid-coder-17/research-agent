import re
from typing import List, Dict, Any

class DocumentChunker:
    """
    Sentence-boundary aware chunker that splits source documents into chunks
    of target size 300-500 words/tokens, attaching citation tags [S<doc_id>:<chunk_id>].
    """
    def __init__(self, min_chunk_words: int = 150, max_chunk_words: int = 400):
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words

    def split_into_sentences(self, text: str) -> List[str]:
        # Clean lines and split on sentence endings (. ! ?) preserving punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk_document(self, text: str, doc_id: int, doc_title: str) -> List[Dict[str, Any]]:
        sentences = self.split_into_sentences(text)
        chunks = []
        current_sentences = []
        current_word_count = 0
        chunk_index = 0

        for sent in sentences:
            sent_words = len(sent.split())
            if current_word_count + sent_words > self.max_chunk_words and current_sentences:
                # Save current chunk
                chunk_text = " ".join(current_sentences)
                chunk_id_str = f"{chunk_index:02d}"
                tag = f"[S{doc_id}:{chunk_id_str}]"
                chunks.append({
                    "doc_id": f"S{doc_id}",
                    "chunk_id": chunk_id_str,
                    "tag": tag,
                    "doc_title": doc_title,
                    "chunk_text": chunk_text,
                    "word_count": current_word_count
                })
                chunk_index += 1
                current_sentences = [sent]
                current_word_count = sent_words
            else:
                current_sentences.append(sent)
                current_word_count += sent_words

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk_id_str = f"{chunk_index:02d}"
            tag = f"[S{doc_id}:{chunk_id_str}]"
            chunks.append({
                "doc_id": f"S{doc_id}",
                "chunk_id": chunk_id_str,
                "tag": tag,
                "doc_title": doc_title,
                "chunk_text": chunk_text,
                "word_count": current_word_count
            })

        return chunks
