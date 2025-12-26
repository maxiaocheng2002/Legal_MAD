"""
Per-category analysis for OAB experiment evaluations.
Groups metrics by legal area (Direito Administrativo, Civil, Penal, etc.)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def aggregate_metrics_list(metrics_list: List[Dict]) -> Dict:
    """
    Aggregate a list of metrics (same logic as evaluate_results.py).

    Args:
        metrics_list: List of metric dictionaries

    Returns:
        Aggregated metrics (mean across all questions)
    """
    if not metrics_list:
        return {}

    # Check if LLM-as-Judge is present
    has_llm_judge = 'llm_judge' in metrics_list[0]

    # Initialize aggregators
    aggregated = {
        'citation_f1': {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        },
        'bertscore': {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    }

    if has_llm_judge:
        aggregated['llm_judge'] = {
            'correctness': 0.0,
            'reasoning': 0.0,
            'citations': 0.0,
            'total': 0.0,
            'normalized': 0.0
        }

    # Sum all metrics
    n = len(metrics_list)
    for metrics in metrics_list:
        for metric_type in ['citation_f1', 'bertscore']:
            for score_type in ['precision', 'recall', 'f1']:
                aggregated[metric_type][score_type] += metrics[metric_type][score_type]

        if has_llm_judge and 'llm_judge' in metrics:
            for score_type in ['correctness', 'reasoning', 'citations', 'total', 'normalized']:
                aggregated['llm_judge'][score_type] += metrics['llm_judge'][score_type]

    # Calculate mean
    for metric_type in ['citation_f1', 'bertscore']:
        for score_type in ['precision', 'recall', 'f1']:
            aggregated[metric_type][score_type] = round(
                aggregated[metric_type][score_type] / n, 4
            )

    if has_llm_judge:
        for score_type in ['correctness', 'reasoning', 'citations', 'total', 'normalized']:
            aggregated['llm_judge'][score_type] = round(
                aggregated['llm_judge'][score_type] / n, 4
            )

    return aggregated


def analyze_by_category(metrics_file: Path, output_dir: Path = None) -> Dict:
    """
    Analyze evaluation metrics grouped by legal category.

    Args:
        metrics_file: Path to metrics JSON file (from evaluate_results.py)
        output_dir: Optional directory to save category analysis

    Returns:
        Dictionary with per-category metrics
    """
    print(f"\n{'='*70}")
    print(f"CATEGORY ANALYSIS: {metrics_file.name}")
    print(f"{'='*70}")

    # Load metrics
    with open(metrics_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    per_question_metrics = data.get('per_question_metrics', [])

    if not per_question_metrics:
        print("ERROR: No per_question_metrics found!")
        return {}

    # Group by category
    by_category = defaultdict(list)
    no_category_count = 0

    for question_metrics in per_question_metrics:
        category = question_metrics.get('category', '')

        if not category:
            no_category_count += 1
            category = 'Unknown'

        by_category[category].append(question_metrics)

    if no_category_count > 0:
        print(f"\nWARNING: {no_category_count} questions without category information")
        print("  → Re-run evaluation with updated pipeline to fix this")

    # Aggregate metrics per category
    category_results = {}

    print(f"\nFound {len(by_category)} categories:")
    print("─"*70)

    for category in sorted(by_category.keys()):
        metrics_list = by_category[category]
        count = len(metrics_list)

        # Aggregate metrics for this category
        aggregated = aggregate_metrics_list(metrics_list)

        category_results[category] = {
            'count': count,
            'aggregated_metrics': aggregated
        }

        # Print summary
        print(f"\n{category}: {count} questions")
        print(f"  Citation F1:    {aggregated['citation_f1']['f1']:.4f}")
        print(f"  BERTScore F1:   {aggregated['bertscore']['f1']:.4f}")
        if 'llm_judge' in aggregated:
            print(f"  LLM-as-Judge:   {aggregated['llm_judge']['normalized']:.4f}")

    # Save results if output_dir provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"category_analysis_{metrics_file.stem}.json"

        # Prepare full output
        full_output = {
            'source_file': metrics_file.name,
            'total_questions': sum(cat['count'] for cat in category_results.values()),
            'total_categories': len(category_results),
            'by_category': category_results,
            'overall_metrics': data.get('aggregated_metrics', {})
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_output, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*70}")
        print(f"Category analysis saved to: {output_file}")
        print(f"{'='*70}\n")

    return category_results


def compare_categories(
    results_dir: str = "evaluation_results",
    pattern: str = "metrics_mad_oab_*.json",
    output_dir: str = "category_analysis"
):
    """
    Run category analysis for multiple experiments.

    Args:
        results_dir: Directory containing metrics files
        pattern: Glob pattern to match files
        output_dir: Directory to save category analyses
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)

    print("\n" + "="*70)
    print("CATEGORY-LEVEL ANALYSIS - OAB EXPERIMENTS")
    print("="*70)
    print(f"Metrics directory: {results_dir}")
    print(f"File pattern: {pattern}")
    print(f"Output directory: {output_dir}")
    print("="*70)

    if not results_path.exists():
        print(f"\nERROR: Directory not found: {results_dir}")
        return

    # Find all metrics files
    metrics_files = list(results_path.glob(pattern))

    if not metrics_files:
        print(f"\nERROR: No files found matching pattern: {pattern}")
        return

    print(f"\nFound {len(metrics_files)} file(s):")
    for f in metrics_files:
        print(f"  - {f.name}")

    # Analyze each file
    all_analyses = {}

    for metrics_file in metrics_files:
        try:
            category_results = analyze_by_category(metrics_file, output_path)
            all_analyses[metrics_file.stem] = category_results
        except Exception as e:
            print(f"\nERROR processing {metrics_file.name}: {e}")
            continue

    # Generate comparative summary
    if all_analyses:
        print("\n" + "="*70)
        print("SUMMARY: All Experiments")
        print("="*70)

        for exp_name, categories in sorted(all_analyses.items()):
            print(f"\n{exp_name}:")
            print(f"  Categories: {len(categories)}")
            print(f"  Total questions: {sum(cat['count'] for cat in categories.values())}")

    print("\n" + "="*70)
    print("Category analysis completed!")
    print(f"Results saved to: {output_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze OAB evaluation metrics by legal category",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single metrics file
  python -m src.analysis.category_metrics \\
      --file evaluation_results/metrics_mad_oab_irac_200_llama-3.3-70b-versatile.json

  # Analyze all OAB metrics
  python -m src.analysis.category_metrics --pattern "metrics_mad_oab_*.json"

  # Custom directories
  python -m src.analysis.category_metrics \\
      --results-dir evaluation_results \\
      --pattern "metrics_*irac*.json" \\
      --output-dir category_results
        """
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--file",
        type=str,
        help="Single metrics file to analyze"
    )
    group.add_argument(
        "--pattern",
        type=str,
        default="metrics_mad_oab_*.json",
        help="Glob pattern to match multiple files (default: metrics_mad_oab_*.json)"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="evaluation_results",
        help="Directory containing metrics files (default: evaluation_results)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="category_analysis",
        help="Output directory for category analyses (default: category_analysis)"
    )

    args = parser.parse_args()

    if args.file:
        # Analyze single file
        metrics_file = Path(args.file)
        output_dir = Path(args.output_dir)

        if not metrics_file.exists():
            print(f"ERROR: File not found: {args.file}")
        else:
            analyze_by_category(metrics_file, output_dir)
    else:
        # Analyze multiple files
        compare_categories(
            results_dir=args.results_dir,
            pattern=args.pattern,
            output_dir=args.output_dir
        )
