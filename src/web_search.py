import os
import sys
import json
import time
import logging
import requests
import trafilatura
from datetime import datetime
from typing import List, Dict, Any, Tuple
from src.chunker import DocumentChunker

logger = logging.getLogger(__name__)
LOG_FILE = "web_fetch_log.jsonl"

def log_web_fetch(url: str, status_code: int, word_count: int, title: str, error: str = ""):
    """Logs every URL fetch attempt to web_fetch_log.jsonl with a timestamp."""
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "url": url,
        "status_code": status_code,
        "word_count": word_count,
        "title": title,
        "error": error
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"Failed writing to {LOG_FILE}: {e}")

def fetch_page_content(url: str, timeout: int = 10, max_retries: int = 1) -> Tuple[str, str, int]:
    """
    Fetches full page content via requests + trafilatura readability extraction.
    Timeout 10s per page, 1 retry max. Returns (title, clean_text, word_count).
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            status_code = resp.status_code
            if status_code == 200:
                html = resp.text
                extracted_text = trafilatura.extract(html, include_comments=False, include_tables=True) or ""
                words = len(extracted_text.split())
                
                # Extract simple title
                title = ""
                try:
                    metadata = trafilatura.extract_metadata(html)
                    if metadata and metadata.title:
                        title = metadata.title
                except Exception:
                    pass
                if not title:
                    title = url.split("/")[-1] or url

                log_web_fetch(url, status_code, words, title)
                return title, extracted_text, words
            else:
                log_web_fetch(url, status_code, 0, "", f"HTTP status {status_code}")
        except Exception as e:
            if attempt == max_retries:
                log_web_fetch(url, 0, 0, "", str(e))
    
    return "", "", 0

def web_retrieve(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Executes live web search using Tavily API (or DuckDuckGo fallback),
    fetches full page content, extracts text, and returns tagged chunks [W<result_index>:<chunk_id>].
    Fails loudly if the search API call fails.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    serper_key = os.getenv("SERPER_API_KEY")
    search_results = []
    api_error = None

    # Step 1: Execute Search API (Tavily, Serper, or DuckDuckGo)
    if tavily_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            res = client.search(query=query, max_results=max_results, search_depth="basic")
            for item in res.get("results", []):
                search_results.append({
                    "url": item.get("url"),
                    "title": item.get("title", ""),
                    "snippet": item.get("content", "")
                })
        except Exception as e:
            api_error = f"Tavily Search API Failure: {e}"
    elif serper_key:
        try:
            headers = {"X-API-KEY": serper_key, "Content-Type": "application/json"}
            payload = json.dumps({"q": query, "num": max_results})
            resp = requests.post("https://google.serper.dev/search", headers=headers, data=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("organic", []):
                    search_results.append({
                        "url": item.get("link"),
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", "")
                    })
            else:
                api_error = f"Serper API HTTP status {resp.status_code}"
        except Exception as e:
            api_error = f"Serper Search API Failure: {e}"
    else:
        # Fallback to DuckDuckGo search if no API key is provided
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                for item in results:
                    search_results.append({
                        "url": item.get("href"),
                        "title": item.get("title", ""),
                        "snippet": item.get("body", "")
                    })
        except Exception as e:
            api_error = f"Web Search API Failure: {e}. Please set TAVILY_API_KEY or SERPER_API_KEY in your .env file for reliable quota."

    # Query Reformulation (up to 2 attempts if 0 results)
    if not search_results and not api_error:
        reformulations = [
            f"{query} latest analysis",
            " ".join(query.split()[:5])
        ]
        for alt_q in reformulations:
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    results = list(ddgs.text(alt_q, max_results=max_results))
                    if results:
                        for item in results:
                            search_results.append({
                                "url": item.get("href"),
                                "title": item.get("title", ""),
                                "snippet": item.get("body", "")
                            })
                        break
            except Exception as e:
                api_error = f"Web Search Reformulation Failure: {e}"

    if api_error and not search_results:
        print(f"\n[CRITICAL ERROR] {api_error}")
        raise RuntimeError(api_error)

    if not search_results:
        print(f"Warning: Live web search returned 0 results for query: '{query}'")
        return []

    # Step 2 & 3: Fetch full page contents and chunk
    chunker = DocumentChunker(min_chunk_words=150, max_chunk_words=400)
    web_chunks = []
    result_idx = 1

    for item in search_results:
        url = item["url"]
        if not url:
            continue

        title, full_text, word_count = fetch_page_content(url, timeout=10, max_retries=1)

        # Skip pages with < 100 words of content
        if word_count < 100 or not full_text.strip():
            logger.info(f"Skipping URL {url}: content words ({word_count}) < 100")
            continue

        doc_title = title if title else item.get("title", url)
        page_chunks = chunker.chunk_document(full_text, doc_id=result_idx, doc_title=doc_title)
        
        # Override tags with [W<result_index>:<chunk_id>] distinct prefix
        for c in page_chunks:
            chunk_id_str = c["chunk_id"]
            tag = f"[W{result_idx}:{chunk_id_str}]"
            c["doc_id"] = f"W{result_idx}"
            c["tag"] = tag
            c["source_type"] = "web"
            c["url"] = url
            web_chunks.append(c)

        result_idx += 1

    return web_chunks
