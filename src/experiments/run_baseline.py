"""
Run baseline experiments on legal QA datasets.
B1: Single-Agent (Zero-shot) - Arihant's implementation
"""

import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from src.utils.api_client import GroqClient
from src.utils.data_loader import load_bar_exam_qa
from src.baselines.single_agent import SingleAgentBaseline


def run_baseline_mcq(
    question_data: Dict,
    baseline: SingleAgentBaseline
) -> Dict:
    """
    Run single-agent baseline on a single MCQ.

    Args:
        question_data: Question dictionary from dataset
        baseline: SingleAgentBaseline instance

    Returns:
        Complete result dictionary with answer and metadata
    """
    # Get baseline answer
    response = baseline.answer_mcq(
        question=question_data['question'],
        prompt_context=question_data['prompt'],
        choices=question_data['choices']
    )

    # Compile result
    result = {
        'question_id': question_data['id'],
        'question': question_data['question'],
        'prompt': question_data['prompt'],
        'choices': question_data['choices'],
        'gold_answer': question_data['answer'],
        'gold_passage': question_data.get('gold_passage', ''),
        'baseline': {
            'answer': response['answer'],
            'reasoning': response.get('reasoning', ''),
            'citations': response.get('citations', []),
            'correct': response['answer'] == question_data['answer']
        }
    }

    return result


def run_baseline_open_ended(
    question_data: Dict,
    baseline: SingleAgentBaseline
) -> Dict:
    """
    Run single-agent baseline on open-ended question.

    Args:
        question_data: Question dictionary
        baseline: SingleAgentBaseline instance

    Returns:
        Complete result dictionary
    """
    # Get baseline answer
    response = baseline.answer_open_ended(
        question=question_data['question'],
        prompt_context=question_data.get('prompt', '')
    )

    # Compile result
    result = {
        'question_id': question_data['id'],
        'question': question_data['question'],
        'prompt': question_data.get('prompt', ''),
        'baseline': {
            'answer': response['answer'],
            'irac': response.get('irac', {}),
            'citations': response.get('citations', [])
        }
    }

    return result


def run_experiments(
    dataset_name: str = "bar_exam_qa",
    sample_size: int = 50,
    output_dir: str = "results",
    question_type: str = "mcq"
):
    """
    Run baseline experiments on dataset.

    Args:
        dataset_name: Name of dataset to use
        sample_size: Number of questions to process
        output_dir: Directory to save results
        question_type: "mcq" or "open_ended"
    """
    print(f"Running Single-Agent Baseline (B1) on {dataset_name}")
    print(f"Sample size: {sample_size}")
    print(f"Question type: {question_type}")

    # Load data
    if dataset_name == "bar_exam_qa":
        questions = load_bar_exam_qa(sample_size=sample_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Initialize baseline
    baseline = SingleAgentBaseline()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run experiments
    results = []
    correct_count = 0

    for question in tqdm(questions, desc="Processing questions"):
        try:
            if question_type == "mcq":
                result = run_baseline_mcq(question, baseline)
                if result['baseline']['correct']:
                    correct_count += 1

                # Print progress
                print(f"\nQuestion {result['question_id']}: "
                      f"Answer: {result['baseline']['answer']}, "
                      f"Gold: {result['gold_answer']}, "
                      f"Correct: {result['baseline']['correct']}")

            else:  # open_ended
                result = run_baseline_open_ended(question, baseline)
                print(f"\nQuestion {result['question_id']}: Processed")

            results.append(result)

        except Exception as e:
            print(f"Error processing question {question['id']}: {e}")
            continue

    # Save results
    output_file = output_path / f"baseline_{dataset_name}_{sample_size}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    if question_type == "mcq":
        accuracy = correct_count / len(results) if results else 0
        print(f"\n=== Results ===")
        print(f"Total questions: {len(results)}")
        print(f"Correct: {correct_count}")
        print(f"Accuracy: {accuracy:.2%}")
    else:
        print(f"\n=== Results ===")
        print(f"Total questions processed: {len(results)}")

    print(f"Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    # Run full experiment on all questions
    run_experiments(
        dataset_name="bar_exam_qa",
        sample_size=None,  # None = all questions
        output_dir="results",
        question_type="mcq"
    )
