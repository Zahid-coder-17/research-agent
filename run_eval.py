import os
import json
import time
from ask import query_agent

QUESTIONS_FILE = "questions.json"
EVAL_RESULTS_FILE = "eval_results.json"

def run_evaluation():
    if not os.path.exists(QUESTIONS_FILE):
        print(f"Error: '{QUESTIONS_FILE}' not found.")
        return

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"\n=======================================================")
    print(f" RUNNING EVALUATION SUITE ({len(questions)} QUESTIONS)")
    print(f"=======================================================\n")

    results = []
    total_drop_rate = 0.0
    total_density = 0.0
    verified_count = 0
    repaired_count = 0

    for idx, q_item in enumerate(questions, start=1):
        q_id = q_item["id"]
        q_type = q_item["type"]
        question = q_item["question"]

        print(f"[{idx}/{len(questions)}] Testing {q_id} ({q_type}): '{question[:50]}...'")
        start_time = time.time()
        
        agent_resp = query_agent(question, top_k=5)
        elapsed = time.time() - start_time
        
        ver = agent_resp["verification"]
        
        if ver["is_verified"]:
            verified_count += 1
        if agent_resp["repaired"]:
            repaired_count += 1

        total_drop_rate += ver["drop_rate"]
        total_density += ver["citation_density"]

        results.append({
            "id": q_id,
            "type": q_type,
            "question": question,
            "expected_behavior": q_item.get("expected_behavior", ""),
            "generated_answer": agent_resp["answer"],
            "verification_status": ver["status"],
            "citation_density": ver["citation_density"],
            "drop_rate": ver["drop_rate"],
            "repaired": agent_resp["repaired"],
            "markers_found": ver["all_markers_found"],
            "execution_time_sec": round(elapsed, 2)
        })
        
        print(f"   -> Status: {ver['status']} | Drop Rate: {ver['drop_rate']*100:.1f}% | Time: {elapsed:.2f}s\n")

    avg_drop_rate = total_drop_rate / len(questions) if questions else 0.0
    avg_density = total_density / len(questions) if questions else 0.0
    verification_pass_rate = (verified_count / len(questions)) * 100 if questions else 0.0

    eval_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": len(questions),
        "verified_questions_count": verified_count,
        "verification_pass_rate_percent": round(verification_pass_rate, 2),
        "average_citation_density": round(avg_density, 4),
        "average_marker_drop_rate": round(avg_drop_rate, 4),
        "repair_passes_executed": repaired_count,
        "results": results
    }

    with open(EVAL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"=======================================================")
    print(f" EVALUATION SUMMARY")
    print(f" Total Questions:        {len(questions)}")
    print(f" Verification Pass Rate: {verification_pass_rate:.1f}%")
    print(f" Avg Citation Density:   {avg_density * 100:.1f}%")
    print(f" Measured Drop Rate:     {avg_drop_rate * 100:.1f}%")
    print(f" Repair Passes Executed: {repaired_count}")
    print(f" Results Saved To:       '{EVAL_RESULTS_FILE}'")
    print(f"=======================================================\n")

if __name__ == "__main__":
    run_evaluation()
