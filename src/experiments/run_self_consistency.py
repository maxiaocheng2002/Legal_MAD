"""
Run Self-Consistency Baseline (SC1) for Bar Exam QA (LOCAL DATASET VERSION)
Fully stable version: non-freezing, full JSON output, reasoning included.
"""

import sys
import json
from pathlib import Path
from tqdm import tqdm

# Ensure project root is importable
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.baselines.self_consistency import SelfConsistencyBaseline
from src.utils.data_loader_local import load_bar_exam_qa_local


def run_sc_experiments(
    dataset_name="bar_exam_qa",
    sample_size=50,
    num_samples=10,
    output_dir="results_sc",
    split="train",
):
    print("Running Self-Consistency Baseline (SC1)")
    print(f"Dataset: {dataset_name}")
    print(f"Split: {split}")
    print(f"Sample size: {sample_size}")
    print(f"Samples/question: {num_samples}")

    # Load dataset
    questions = load_bar_exam_qa_local(sample_size=sample_size, split=split)

    sc_agent = SelfConsistencyBaseline(num_samples=num_samples)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    correct_count = 0

    for q in tqdm(questions, desc="Running SC"):
        try:
            response = sc_agent.answer_mcq(
                question=q["question"],
                prompt_context=q["prompt"],
                choices=q["choices"],
            )

            correct = (response["answer"] == q["answer"])

            results.append({
                "question_id": q["id"],
                "question": q["question"],
                "prompt": q["prompt"],
                "choices": q["choices"],
                "gold_answer": q["answer"],

                # SC outputs
                "sc_answer": response["answer"],
                "samples": response["samples"],
                "reasoning_samples": response["reasoning_samples"],
                "majority_count": response["majority_count"],
                "correct": correct,
            })

            if correct:
                correct_count += 1

        except Exception as e:
            print(f"Error on question {q['id']}: {e}")

    # Save JSON
    out_file = output_path / f"sc_{dataset_name}_{split}_{sample_size}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    accuracy = correct_count / len(results) if results else 0

    print("\n=== FINAL RESULTS (SC1) ===")
    print(f"Total questions: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Saved to: {out_file}")

    return results


if __name__ == "__main__":
    run_sc_experiments(
        dataset_name="bar_exam_qa",
        sample_size=None,      # change to None for full dataset
        num_samples=1,       # adjust for speed/accuracy
        output_dir="results_sc",
        split="train",
    )

