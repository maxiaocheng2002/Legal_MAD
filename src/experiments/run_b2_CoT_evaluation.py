import os
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
EVALUATION_FILENAME = "results_b2_cot_evaluation.py"


def run_evaluation(experiment_name: str):
    filename = os.path.join(
        RESULTS_DIR,
        f"results_b2_cot_{experiment_name.replace(' ', '_')}.json"
    )

    if not os.path.exists(filename):
        return {
            "experiment_name": experiment_name,
            "error": "File not found",
            "total": 0,
            "correct": 0,
            "accuracy": "0.00%"
        }

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    correct = 0

    for item in data:
        gold = item.get("gold_answer")
        judge = item.get("judge", {})
        pred = judge.get("decision")

        if pred is not None and gold is not None and pred.upper() == gold.upper():
            correct += 1

    accuracy = (correct / total) * 100 if total > 0 else 0

    return {
        "experiment_name": experiment_name,
        "total": total,
        "correct": correct,
        "accuracy": f"{accuracy:.2f}%"
    }


if __name__ == "__main__":
    experiments = ["IRAC_CoT", "Basic_CoT"]

    all_results = {}

    for exp in experiments:
        result = run_evaluation(exp)
        print(result)
        all_results[exp] = result

    output_path = os.path.join(RESULTS_DIR, EVALUATION_FILENAME)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print("Saved evaluation to", output_path)
