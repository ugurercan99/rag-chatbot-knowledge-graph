import json
from pathlib import Path
import evaluate_chatbot

def fast_run():
    # Load questions
    questions = evaluate_chatbot.load_questions(Path("evaluation_questions.json"))
    q_map = {q.id: q for q in questions}

    # Load old results to get the raw answers
    with open("evaluation_results.json", "r") as f:
        old_data = json.load(f)
    
    metadata = old_data["metadata"]
    ragas_status = old_data["ragas"]
    old_results = old_data["results"]
    
    new_results = []
    
    for row in old_results:
        q_id = row["id"]
        if q_id not in q_map: continue
        q = q_map[q_id]
        
        answer = row["answer"]
        retrieved_sources = row["retrieved_sources"]
        
        # Recompute source metrics
        hit_rate, recall = evaluate_chatbot.compute_source_metrics(q, retrieved_sources)
        
        # Recompute abstention
        abstention_detected = evaluate_chatbot.detect_abstention(answer)
        
        # Recompute keywords
        kr = evaluate_chatbot.keyword_recall(answer, q.reference_keywords)
        
        new_row = dict(row)
        new_row["category"] = q.category
        new_row["should_answer"] = q.should_answer
        new_row["reference_keywords"] = q.reference_keywords
        new_row["acceptable_sources_any"] = q.acceptable_sources_any
        new_row["required_sources_all"] = q.required_sources_all
        new_row["expected_sources"] = q.expected_sources
        
        new_row["source_hit_rate_at_k"] = evaluate_chatbot.round_metric(hit_rate)
        new_row["source_recall_at_k"] = evaluate_chatbot.round_metric(recall)
        new_row["keyword_recall"] = evaluate_chatbot.round_metric(kr)
        new_row["abstention_detected"] = abstention_detected
        new_row["abstention_correct"] = (not q.should_answer) and abstention_detected
        
        new_row["question_score"] = evaluate_chatbot.round_metric(evaluate_chatbot.build_question_score(new_row))
        new_results.append(new_row)
        
    summary = evaluate_chatbot.build_summary_markdown(new_results, metadata, ragas_status, Path("evaluation_questions.json"))
    
    evaluate_chatbot.write_csv(new_results, Path("evaluation_results.csv"))
    evaluate_chatbot.write_json(new_results, metadata, ragas_status, Path("evaluation_results.json"))
    Path("evaluation_summary.md").write_text(summary, encoding="utf-8")
    print("Fast recalc complete!")

if __name__ == "__main__":
    fast_run()
