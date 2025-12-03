"""
Run MAD experiments on legal QA datasets.
"""

import json
import random
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from src.utils.api_client import GroqClient
from src.utils.data_loader import load_bar_exam_qa
from src.agents.debater import Debater
from src.agents.judge import Judge


def assign_positions(choices: List[str]) -> tuple:
    """
    Assign positions to debaters.

    Strategy:
    - Debater 1: Chooses freely (will be determined by model preference in opening)
    - Debater 2: Must choose different position from Debater 1

    For now, we randomly assign to ensure adversarial setup.

    Args:
        choices: List of 4 answer choices

    Returns:
        Tuple of (position_x, position_y) e.g., ('A', 'C')
    """
    available_positions = ['A', 'B', 'C', 'D']
    positions = random.sample(available_positions, 2)
    return positions[0], positions[1]


def run_mad_mcq(
    question_data: Dict,
    client: GroqClient
) -> Dict:
    """
    Run MAD debate on a single MCQ with IRAC structure and token optimization.

    Features:
    - IRAC-structured arguments (Issue, Rule, Application, Conclusion)
    - Token-efficient format using summaries instead of full arguments
    - Reduced token limits: 350 (opening), 300 (rebuttal), 300 (judge)
    - Expected ~28% token reduction vs. vanilla version

    Args:
        question_data: Question dictionary from dataset
        client: Groq API client

    Returns:
        Complete result dictionary with IRAC-structured debate logs and decision
    """
    # Initialize agents
    debater_x = Debater(client, name="Debater_X")
    debater_y = Debater(client, name="Debater_Y")
    judge = Judge(client)

    # Assign positions
    pos_x, pos_y = assign_positions(question_data['choices'])

    # Round 1: Opening arguments (simultaneous)
    opening_x = debater_x.generate_opening(
        question=question_data['question'],
        prompt_context=question_data['prompt'],
        choices=question_data['choices'],
        position=pos_x
    )

    opening_y = debater_y.generate_opening(
        question=question_data['question'],
        prompt_context=question_data['prompt'],
        choices=question_data['choices'],
        position=pos_y
    )

    # Round 2: Rebuttals (sequential)
    rebuttal_x = debater_x.generate_rebuttal(
        question=question_data['question'],
        prompt_context=question_data['prompt'],
        opponent_opening=opening_y
    )

    rebuttal_y = debater_y.generate_rebuttal(
        question=question_data['question'],
        prompt_context=question_data['prompt'],
        opponent_opening=opening_x
    )

    # Judge decision
    debate_history = {
        'debater_x': {
            'opening': opening_x,
            'rebuttal': rebuttal_x
        },
        'debater_y': {
            'opening': opening_y,
            'rebuttal': rebuttal_y
        }
    }

    decision = judge.make_decision(
        question=question_data['question'],
        prompt_context=question_data['prompt'],
        choices=question_data['choices'],
        debate_history=debate_history
    )

    # Compile result with enhanced IRAC structure
    result = {
        'question_id': question_data['id'],
        'question': question_data['question'],
        'prompt': question_data['prompt'],
        'choices': question_data['choices'],
        'gold_answer': question_data['answer'],
        'gold_passage': question_data.get('gold_passage', ''),
        'debate': {
            'round_1': {
                'debater_x': {
                    'position': opening_x['position'],
                    'irac': opening_x.get('irac', {}),
                    'key_citations': opening_x.get('key_citations', []),
                    'summary': opening_x.get('argument_summary', '')
                },
                'debater_y': {
                    'position': opening_y['position'],
                    'irac': opening_y.get('irac', {}),
                    'key_citations': opening_y.get('key_citations', []),
                    'summary': opening_y.get('argument_summary', '')
                }
            },
            'round_2': {
                'debater_x': {
                    'rebuttal_irac': rebuttal_x.get('rebuttal_irac', {}),
                    'counter_argument': rebuttal_x.get('counter_argument', ''),
                    'key_citations': rebuttal_x.get('key_citations', []),
                    'summary': rebuttal_x.get('rebuttal_summary', '')
                },
                'debater_y': {
                    'rebuttal_irac': rebuttal_y.get('rebuttal_irac', {}),
                    'counter_argument': rebuttal_y.get('counter_argument', ''),
                    'key_citations': rebuttal_y.get('key_citations', []),
                    'summary': rebuttal_y.get('rebuttal_summary', '')
                }
            }
        },
        'judge': {
            'decision': decision.get('decision', ''),
            'rationale': decision.get('rationale', ''),
            'irac_analysis': decision.get('irac_analysis', {}),
            'key_factors': decision.get('key_factors', []),
            'correct': decision.get('decision', '') == question_data['answer']
        }
    }

    return result


def run_experiments(
    dataset_name: str = "bar_exam_qa",
    sample_size: int = 50,
    output_dir: str = "results"
):
    """
    Run MAD experiments on dataset.

    Args:
        dataset_name: Name of dataset to use
        sample_size: Number of questions to process
        output_dir: Directory to save results
    """
    print(f"Running MAD experiments on {dataset_name}")
    print(f"Sample size: {sample_size}")

    # Load data
    if dataset_name == "bar_exam_qa":
        questions = load_bar_exam_qa(sample_size=sample_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Initialize client with reduced token limits (IRAC structure is more concise)
    client = GroqClient(max_tokens=350)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run experiments
    results = []
    correct_count = 0

    for question in tqdm(questions, desc="Processing questions"):
        try:
            result = run_mad_mcq(question, client)
            results.append(result)

            if result['judge']['correct']:
                correct_count += 1

            # Print progress
            print(f"\nQuestion {result['question_id']}: "
                  f"Judge decided {result['judge']['decision']}, "
                  f"Gold: {result['gold_answer']}, "
                  f"Correct: {result['judge']['correct']}")

        except Exception as e:
            print(f"Error processing question {question['id']}: {e}")
            continue

    # Save results
    output_file = output_path / f"mad_{dataset_name}_{sample_size}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    accuracy = correct_count / len(results) if results else 0
    print(f"\n=== Results ===")
    print(f"Total questions: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    # Run on small sample for testing
    run_experiments(
        dataset_name="bar_exam_qa",
        sample_size=5,
        output_dir="results"
    )
