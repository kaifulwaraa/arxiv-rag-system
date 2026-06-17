import os
import json
import sys
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieval import load_all_chunks, naive_vector_search, hybrid_search, hybrid_rerank_search
from rag_pipeline import generate_answer

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

def load_test_questions():
    with open("results/test_questions.json", "r") as f:
        return json.load(f)

def call_llm_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = groq_llm.invoke(prompt)
            time.sleep(2)
            return response.content
        except Exception as e:
            if "429" in str(e):
                wait_time = 30 * (attempt + 1)
                print(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Error: {e}")
                return None
    return None

def evaluate_faithfulness(question, answer, contexts):
    context_str = "\n\n".join(contexts)
    prompt = f"""Given the following question, answer, and context, rate the faithfulness of the answer on a scale of 0 to 1.
Faithfulness means: does the answer only contain information that is present in the context?

Question: {question}
Answer: {answer}
Context: {context_str}

Return ONLY a number between 0 and 1. Nothing else."""
    result = call_llm_with_retry(prompt)
    try:
        return float(result.strip())
    except:
        return 0.5

def evaluate_answer_relevancy(question, answer):
    prompt = f"""Rate how relevant this answer is to the question on a scale of 0 to 1.
1 means perfectly relevant, 0 means completely irrelevant.

Question: {question}
Answer: {answer}

Return ONLY a number between 0 and 1. Nothing else."""
    result = call_llm_with_retry(prompt)
    try:
        return float(result.strip())
    except:
        return 0.5

def evaluate_context_precision(question, contexts):
    context_str = "\n\n".join(contexts)
    prompt = f"""Rate the precision of these retrieved contexts for answering the question.
Precision means: what fraction of the retrieved contexts are actually relevant to the question?

Question: {question}
Contexts: {context_str}

Return ONLY a number between 0 and 1. Nothing else."""
    result = call_llm_with_retry(prompt)
    try:
        return float(result.strip())
    except:
        return 0.5

def evaluate_context_recall(question, ground_truth, contexts):
    context_str = "\n\n".join(contexts)
    prompt = f"""Rate the recall of these retrieved contexts.
Recall means: does the context contain all the information needed to answer the question correctly?

Question: {question}
Expected Answer: {ground_truth}
Retrieved Contexts: {context_str}

Return ONLY a number between 0 and 1. Nothing else."""
    result = call_llm_with_retry(prompt)
    try:
        return float(result.strip())
    except:
        return 0.5

def evaluate_strategy(questions, strategy, all_chunks):
    print(f"\n--- Evaluating Strategy: {strategy} ---")
    scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": []
    }

    for i, q in enumerate(questions):
        print(f"  Question {i+1}/{len(questions)}: {q['question'][:50]}...")

        if strategy == "naive":
            chunks = naive_vector_search(q["question"], top_k=5)
        elif strategy == "hybrid":
            chunks = hybrid_search(q["question"], all_chunks, top_k=5)
        elif strategy == "rerank":
            chunks = hybrid_rerank_search(q["question"], all_chunks, top_k=5)

        contexts = [c["text"] for c in chunks]

        answer = generate_answer(q["question"], chunks)
        time.sleep(3)

        print(f"    Scoring faithfulness...")
        f_score = evaluate_faithfulness(q["question"], answer, contexts)
        scores["faithfulness"].append(f_score)
        time.sleep(3)

        print(f"    Scoring answer relevancy...")
        ar_score = evaluate_answer_relevancy(q["question"], answer)
        scores["answer_relevancy"].append(ar_score)
        time.sleep(3)

        print(f"    Scoring context precision...")
        cp_score = evaluate_context_precision(q["question"], contexts)
        scores["context_precision"].append(cp_score)
        time.sleep(3)

        print(f"    Scoring context recall...")
        cr_score = evaluate_context_recall(q["question"], q["ground_truth"], contexts)
        scores["context_recall"].append(cr_score)
        time.sleep(5)

        print(f"    Scores: F={f_score:.2f} AR={ar_score:.2f} CP={cp_score:.2f} CR={cr_score:.2f}")

    return {
        "faithfulness": sum(scores["faithfulness"]) / len(scores["faithfulness"]),
        "answer_relevancy": sum(scores["answer_relevancy"]) / len(scores["answer_relevancy"]),
        "context_precision": sum(scores["context_precision"]) / len(scores["context_precision"]),
        "context_recall": sum(scores["context_recall"]) / len(scores["context_recall"])
    }

def run_evaluation():
    print("=== Starting Sequential Evaluation Pipeline ===\n")

    questions = load_test_questions()
    questions = questions[:5]
    print(f"Loaded {len(questions)} test questions")

    print("Loading chunks for BM25...")
    all_chunks = load_all_chunks()

    all_scores = {}
    strategies = ["naive", "hybrid", "rerank"]

    for strategy in strategies:
        scores = evaluate_strategy(questions, strategy, all_chunks)
        all_scores[strategy] = scores

        print(f"\nResults for {strategy}:")
        for metric, score in scores.items():
            print(f"  {metric}: {score:.4f}")

        with open("results/benchmark_results.json", "w") as f:
            json.dump(all_scores, f, indent=2)
        print(f"Saved results after {strategy}")

        if strategy != "rerank":
            print(f"\nWaiting 2 minutes before next strategy...")
            time.sleep(120)

    print("\n=== FINAL RESULTS ===")
    print(f"{'Strategy':<20} {'Faithfulness':<15} {'Answer Rel.':<15} {'Context Prec.':<15} {'Context Rec.':<15}")
    print("-" * 80)
    for strategy, scores in all_scores.items():
        print(f"{strategy:<20} {scores['faithfulness']:<15.4f} {scores['answer_relevancy']:<15.4f} {scores['context_precision']:<15.4f} {scores['context_recall']:<15.4f}")

if __name__ == "__main__":
    run_evaluation()