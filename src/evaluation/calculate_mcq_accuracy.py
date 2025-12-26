"""
Calculate accuracy for MAD experiments on Bar Exam MCQ dataset.
Compares judge's decision with gold_answer.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List


def calculate_accuracy(result_file: Path) -> Dict:
    """
    Calculate accuracy for a single MCQ experiment.

    Args:
        result_file: Path to result JSON file

    Returns:
        Dictionary with accuracy metrics
    """
    print(f"\n{'─'*70}")
    print(f"Processing: {result_file.name}")
    print(f"{'─'*70}")

    # Load results
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    total = len(results)
    correct = 0
    errors = 0

    # Count correct predictions
    for item in results:
        # Skip if error
        if 'error' in item:
            errors += 1
            continue

        gold_answer = item.get('gold_answer', '').upper()
        judge_decision = item.get('judge', {}).get('decision', '').upper()

        if not gold_answer or not judge_decision:
            errors += 1
            continue

        if judge_decision == gold_answer:
            correct += 1

    # Calculate accuracy
    valid_questions = total - errors
    accuracy = (correct / valid_questions * 100) if valid_questions > 0 else 0.0

    result = {
        'experiment': result_file.stem,
        'total_questions': total,
        'valid_questions': valid_questions,
        'errors': errors,
        'correct': correct,
        'incorrect': valid_questions - correct,
        'accuracy': round(accuracy, 2)
    }

    # Print summary
    print(f"  Total questions:  {total}")
    print(f"  Valid questions:  {valid_questions}")
    print(f"  Correct:          {correct}")
    print(f"  Incorrect:        {valid_questions - correct}")
    print(f"  Errors:           {errors}")
    print(f"  Accuracy:         {accuracy:.2f}%")

    return result


def compare_experiments(
    results_dir: str = "results",
    pattern: str = "*500*.json",
    output_dir: str = "evaluation_results"
) -> None:
    """
    Calculate accuracy for multiple experiments and generate comparison.

    Args:
        results_dir: Directory containing result files
        pattern: Glob pattern to match files (default: *500*.json for all 500-question experiments)
        output_dir: Directory to save evaluation results
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)

    print("\n" + "="*70)
    print("MCQ ACCURACY EVALUATION - BAR EXAM EXPERIMENTS")
    print("="*70)
    print(f"Results directory: {results_dir}")
    print(f"File pattern: {pattern}")
    print(f"Output directory: {output_dir}")
    print("="*70)

    if not results_path.exists():
        print(f"\nERROR: Results directory not found: {results_dir}")
        return

    # Find all result files matching pattern
    result_files = list(results_path.glob(pattern))

    # Exclude checkpoint files
    result_files = [f for f in result_files if not f.name.startswith('checkpoint_')]

    if not result_files:
        print(f"\nWarning: No result files found matching pattern: {pattern}")
        return

    print(f"\nFound {len(result_files)} experiment(s):")
    for f in result_files:
        print(f"  - {f.name}")

    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for result_file in result_files:
        try:
            accuracy_result = calculate_accuracy(result_file)
            all_results[result_file.stem] = accuracy_result
        except Exception as e:
            print(f"\nERROR processing {result_file.name}: {e}")
            continue

    # Generate comparison report
    if all_results:
        print("\n" + "="*70)
        print("COMPARATIVE REPORT")
        print("="*70)

        # Sort by accuracy (descending)
        sorted_results = sorted(
            all_results.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        print("\nRanking by Accuracy:")
        print("─"*70)
        for rank, (exp_name, metrics) in enumerate(sorted_results, 1):
            # Shorten name for display
            display_name = exp_name.replace('mad_', '').replace('bar_exam_qa_500', '').replace('_', ' ').strip()
            if not display_name:
                display_name = 'vanilla llama 8B'
            print(f"{rank}. {display_name}")
            print(f"   Accuracy: {metrics['accuracy']:.2f}% ({metrics['correct']}/{metrics['valid_questions']})")

        # Separate by mode
        vanilla_experiments = {k: v for k, v in all_results.items() if 'vanilla' in k.lower()}
        irac_experiments = {k: v for k, v in all_results.items() if 'irac' in k.lower()}

        if vanilla_experiments:
            print("\n\nMAD Vanilla Experiments:")
            print("─"*70)
            for exp_name, metrics in sorted(vanilla_experiments.items(), key=lambda x: x[1]['accuracy'], reverse=True):
                print(f"  {exp_name}:")
                print(f"    Accuracy: {metrics['accuracy']:.2f}%")
                print(f"    Correct:  {metrics['correct']}/{metrics['valid_questions']}")

        if irac_experiments:
            print("\n\nMAD IRAC Experiments:")
            print("─"*70)
            for exp_name, metrics in sorted(irac_experiments.items(), key=lambda x: x[1]['accuracy'], reverse=True):
                print(f"  {exp_name}:")
                print(f"    Accuracy: {metrics['accuracy']:.2f}%")
                print(f"    Correct:  {metrics['correct']}/{metrics['valid_questions']}")

        # Save results
        comparison_file = output_path / "mcq_accuracy_report.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n\nAccuracy report saved to: {comparison_file}")

    print("\n" + "="*70)
    print("Evaluation completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate accuracy for MAD MCQ experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all 500-question experiments (default)
  python -m src.evaluation.calculate_mcq_accuracy

  # Evaluate all MAD experiments
  python -m src.evaluation.calculate_mcq_accuracy --pattern "mad_*.json"

  # Evaluate only vanilla experiments
  python -m src.evaluation.calculate_mcq_accuracy --pattern "mad_vanilla_*.json"

  # Evaluate specific pattern
  python -m src.evaluation.calculate_mcq_accuracy --pattern "*irac*500*.json"
        """
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing result JSON files (default: results)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*500*.json",
        help="Glob pattern to match files (default: *500*.json for all 500-question experiments)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for evaluation results (default: evaluation_results)"
    )

    args = parser.parse_args()

    compare_experiments(
        results_dir=args.results_dir,
        pattern=args.pattern,
        output_dir=args.output_dir
    )
