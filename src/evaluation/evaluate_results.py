"""
Evaluate OAB experiment results using automatic metrics.
Processes result JSONs and generates comparative reports.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from src.evaluation.automatic_metrics import (
    evaluate_single_result,
    aggregate_metrics
)
from src.evaluation.llm_judge import evaluate_with_llm_judge
from src.utils.api_client_experimental import OpenRouterClient


def evaluate_experiment(
    result_file: Path,
    output_dir: Path = None,
    use_llm_judge: bool = False
) -> Dict:
    """
    Evaluate a single experiment result file.

    Args:
        result_file: Path to result JSON file
        output_dir: Optional directory to save detailed metrics
        use_llm_judge: Whether to use LLM-as-Judge evaluation (slower but more comprehensive)

    Returns:
        Aggregated metrics dictionary
    """
    print(f"\n{'─'*70}")
    print(f"Evaluating: {result_file.name}")
    print(f"{'─'*70}")

    # Load results
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Initialize LLM-as-Judge client if needed
    llm_client = None
    if use_llm_judge:
        print("  LLM-as-Judge: ENABLED (using GPT-4o-mini)")
        llm_client = OpenRouterClient(
            model="openai/gpt-4o-mini",
            temperature=0.1,
            max_tokens=500
        )
    else:
        print("  LLM-as-Judge: DISABLED (use --use-llm-judge to enable)")

    # Evaluate each question
    all_metrics = []
    errors = 0

    for result in tqdm(results, desc="Computing metrics"):
        # Skip error results
        if 'error' in result:
            errors += 1
            continue

        try:
            # Run automatic metrics
            metrics = evaluate_single_result(result)

            # Add LLM-as-Judge if enabled
            if use_llm_judge:
                # Extract data based on result type
                if 'judge' in result:
                    predicted_answer = result['judge'].get('final_answer', '')
                else:
                    predicted_answer = result.get('answer', '')

                ground_truth = result.get('ground_truth', {})
                reference_answer = ground_truth.get('reference_answer', '')
                question = result.get('question', '')

                llm_scores = evaluate_with_llm_judge(
                    prediction=predicted_answer,
                    reference=reference_answer,
                    question=question,
                    client=llm_client
                )
                metrics['llm_judge'] = llm_scores

            # Preserve metadata for per-category analysis
            metrics['question_id'] = result.get('question_id', '')
            metrics['category'] = result.get('category', '')

            all_metrics.append(metrics)
        except Exception as e:
            print(f"\nWarning: Error evaluating {result.get('question_id', 'unknown')}: {e}")
            errors += 1
            continue

    # Aggregate metrics
    if not all_metrics:
        print("ERROR: No valid results to evaluate!")
        return {}

    aggregated = aggregate_metrics(all_metrics)

    # Print summary
    print(f"\nResults Summary:")
    print(f"  Total questions: {len(results)}")
    print(f"  Successfully evaluated: {len(all_metrics)}")
    print(f"  Errors: {errors}")
    print(f"\nMetrics:")
    print(f"  Citation F1:    {aggregated['citation_f1']['f1']:.4f}")
    print(f"    - Precision:  {aggregated['citation_f1']['precision']:.4f}")
    print(f"    - Recall:     {aggregated['citation_f1']['recall']:.4f}")
    print(f"  BERTScore F1:   {aggregated['bertscore']['f1']:.4f}")
    print(f"    - Precision:  {aggregated['bertscore']['precision']:.4f}")
    print(f"    - Recall:     {aggregated['bertscore']['recall']:.4f}")

    # Print LLM-as-Judge summary if enabled
    if use_llm_judge and 'llm_judge' in aggregated:
        print(f"  LLM-as-Judge:   {aggregated['llm_judge']['normalized']:.4f}")
        print(f"    - Correctness: {aggregated['llm_judge']['correctness']:.2f}/4")
        print(f"    - Reasoning:   {aggregated['llm_judge']['reasoning']:.2f}/3")
        print(f"    - Citations:   {aggregated['llm_judge']['citations']:.2f}/4")

    # Save detailed metrics if output_dir provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save aggregated metrics
        metrics_file = output_dir / f"metrics_{result_file.stem}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump({
                'experiment': result_file.stem,
                'total_questions': len(results),
                'evaluated_questions': len(all_metrics),
                'errors': errors,
                'aggregated_metrics': aggregated,
                'per_question_metrics': all_metrics
            }, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed metrics saved to: {metrics_file}")

    return aggregated


def compare_experiments(
    results_dir: str = "results",
    pattern: str = "*oab*.json",
    output_dir: str = "evaluation_results",
    use_llm_judge: bool = False
):
    """
    Evaluate and compare all experiments.

    Args:
        results_dir: Directory containing result files
        pattern: Glob pattern to match files
        output_dir: Directory to save evaluation results
        use_llm_judge: Whether to use LLM-as-Judge evaluation
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)

    print("\n" + "="*70)
    print("AUTOMATIC EVALUATION - OAB EXPERIMENTS")
    print("="*70)
    print(f"Results directory: {results_dir}")
    print(f"File pattern: {pattern}")
    print(f"Output directory: {output_dir}")
    print("="*70)

    if not results_path.exists():
        print(f"\nERROR: Results directory not found: {results_dir}")
        return

    # Find all result files
    result_files = list(results_path.glob(pattern))

    # Exclude checkpoint files
    result_files = [f for f in result_files if not f.name.startswith('checkpoint_')]

    if not result_files:
        print(f"\nWarning: No result files found matching pattern: {pattern}")
        return

    print(f"\nFound {len(result_files)} experiment(s):")
    for f in result_files:
        print(f"  - {f.name}")

    # Evaluate each experiment
    all_results = {}

    for result_file in result_files:
        try:
            aggregated = evaluate_experiment(result_file, output_path, use_llm_judge=use_llm_judge)
            all_results[result_file.stem] = aggregated
        except Exception as e:
            print(f"\nERROR processing {result_file.name}: {e}")
            continue

    # Generate comparison report
    if all_results:
        print("\n" + "="*70)
        print("COMPARATIVE REPORT")
        print("="*70)

        # Sort by experiment type
        mad_experiments = {k: v for k, v in all_results.items() if 'mad' in k.lower()}
        baseline_experiments = {k: v for k, v in all_results.items() if 'baseline' in k.lower()}

        # Print MAD results
        if mad_experiments:
            print("\nMAD Experiments:")
            print("─"*70)
            for exp_name, metrics in sorted(mad_experiments.items()):
                print(f"\n  {exp_name}:")
                print(f"    Citation F1:  {metrics['citation_f1']['f1']:.4f}")
                print(f"    BERTScore:    {metrics['bertscore']['f1']:.4f}")
                if 'llm_judge' in metrics:
                    print(f"    LLM-as-Judge: {metrics['llm_judge']['normalized']:.4f}")

        # Print Baseline results
        if baseline_experiments:
            print("\nBaseline Experiments:")
            print("─"*70)
            for exp_name, metrics in sorted(baseline_experiments.items()):
                print(f"\n  {exp_name}:")
                print(f"    Citation F1:  {metrics['citation_f1']['f1']:.4f}")
                print(f"    BERTScore:    {metrics['bertscore']['f1']:.4f}")
                if 'llm_judge' in metrics:
                    print(f"    LLM-as-Judge: {metrics['llm_judge']['normalized']:.4f}")

        # Save comparison
        comparison_file = output_path / "comparison_report.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\nComparison report saved to: {comparison_file}")

    print("\n" + "="*70)
    print("Evaluation completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate OAB experiment results with automatic metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all OAB experiments
  python -m src.evaluation.evaluate_results

  # Evaluate specific pattern
  python -m src.evaluation.evaluate_results --pattern "mad_oab_*.json"

  # Custom directories
  python -m src.evaluation.evaluate_results --results-dir my_results/ --output-dir my_eval/
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
        default="*oab*.json",
        help="Glob pattern to match files (default: *oab*.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for evaluation results (default: evaluation_results)"
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Enable LLM-as-Judge evaluation using GPT-4o-mini (slower but more comprehensive)"
    )

    args = parser.parse_args()

    compare_experiments(
        results_dir=args.results_dir,
        pattern=args.pattern,
        output_dir=args.output_dir,
        use_llm_judge=args.use_llm_judge
    )
