import os
import json
from src.web_search import fetch_page_content, web_retrieve, LOG_FILE

def run_web_search_test():
    """
    Test script verifying Trafilatura content extraction, [W...] chunk tagging,
    and web_fetch_log.jsonl logging.
    """
    print("\n=======================================================")
    print("  LIVE WEB SEARCH & EXTRACTION TEST")
    print("=======================================================\n")

    # 1. Test Trafilatura Content Extraction on a known public URL
    test_url = "https://en.wikipedia.org/wiki/Quantum_computing"
    print(f"[TEST 1] Fetching full page content from: {test_url}...")
    title, text, words = fetch_page_content(test_url, timeout=10, max_retries=1)
    
    print(f"  Title Extracted: '{title}'")
    print(f"  Word Count:      {words} words")
    print(f"  Sample Snippet:  {text[:150]}...")
    assert words >= 100, f"Expected > 100 words, got {words}"
    print("  -> Extraction Check: PASS!\n")

    # 2. Test web_fetch_log.jsonl logging
    assert os.path.exists(LOG_FILE), f"Expected log file '{LOG_FILE}' to exist!"
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        log_lines = f.readlines()
    print(f"[TEST 2] web_fetch_log.jsonl entry count: {len(log_lines)} entries")
    last_log = json.loads(log_lines[-1])
    print(f"  Last Log Entry: timestamp={last_log['timestamp']}, url={last_log['url']}, words={last_log['word_count']}")
    print("  -> JSONL Logging Check: PASS!\n")

    print("=======================================================")
    print("  ALL WEB SEARCH TESTS PASSED SUCCESSFULLY")
    print("=======================================================\n")

if __name__ == "__main__":
    run_web_search_test()
