"""
Run MAD experiments on legal QA datasets.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from src.utils.data_loader import load_bar_exam_qa
from src.agents.debater_experimental import Debater
from src.agents.judge_experimental import Judge


def create_client(provider: str = "groq", model: str = None, max_tokens: int = 1500, max_retries: int = 10, retry_delay: float = 2.0):
    """
    Create API client based on provider.

    Args:
        provider: "groq" or "openrouter"
        model: Model name (uses defaults if None)
        max_tokens: Maximum tokens to generate
        max_retries: Maximum retry attempts
        retry_delay: Initial retry delay

    Returns:
        GroqClient or OpenRouterClient instance
    """
    if provider == "groq":
        from src.utils.api_client_experimental import GroqClient
        default_model = "llama-3.1-8b-instant"
        return GroqClient(
            model=model or default_model,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    elif provider == "openrouter":
        from src.utils.api_client_experimental import OpenRouterClient
        default_model = "meta-llama/llama-3.3-70b-instruct:free"
        return OpenRouterClient(
            model=model or default_model,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_available_positions_for_debater_y(position_x: str) -> List[str]:
    """
    Get available positions for Debater Y (all except X's choice).

    Args:
        position_x: Position chosen by Debater X

    Returns:
        List of available positions for Debater Y
    """
    all_positions = ['A', 'B', 'C', 'D']
    return [p for p in all_positions if p != position_x]


def run_mad_mcq(
    question_data: Dict,
    client
) -> Dict:
    """
    Run MAD debate on a single MCQ.

    Position assignment strategy:
    - Debater X: Generates opening argument without pre-assigned position,
                 naturally choosing the position they find most defensible
    - Debater Y: Gets a randomly selected position from remaining choices
                 (any except X's choice)

    Args:
        question_data: Question dictionary from dataset
        client: Groq API client

    Returns:
        Complete result dictionary with debate logs and decision
    """
    # Initialize agents
    debater_x = Debater(client, name="Debater_X")
    debater_y = Debater(client, name="Debater_Y")
    judge = Judge(client)

    # Round 1: Debater X generates opening (chooses position freely)
    opening_x = debater_x.generate_opening(
        question=question_data['question'],
        prompt_context=question_data['prompt'],
        choices=question_data['choices'],
        position=None  # Let debater choose
    )

    # Extract position chosen by Debater X
    pos_x = opening_x.get('position')
    if not pos_x or pos_x not in ['A', 'B', 'C', 'D']:
        raise ValueError(f"Debater X returned invalid position: {pos_x}")

    # Assign Debater Y a different position
    available_positions = get_available_positions_for_debater_y(pos_x)
    pos_y = random.choice(available_positions)

    # Round 1: Debater Y generates opening with assigned position
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

    # Compile result
    result = {
        'question_id': question_data['id'],
        'question': question_data['question'],
        'prompt': question_data['prompt'],
        'choices': question_data['choices'],
        'gold_answer': question_data['answer'],
        'gold_passage': question_data.get('gold_passage', ''),
        'position_assignment': {
            'debater_x_position': pos_x,
            'debater_y_position': pos_y,
            'strategy': 'X chooses freely, Y assigned different position'
        },
        'debate': {
            'round_1': {
                'debater_x': opening_x,
                'debater_y': opening_y
            },
            'round_2': {
                'debater_x': rebuttal_x,
                'debater_y': rebuttal_y
            }
        },
        'judge': {
            'rationale': decision.get('rationale', ''),
            'winner': decision.get('winner', 'unknown'),
            'decision': decision['decision'],
            'synthesis': decision.get('synthesis', ''),
            'correct': decision['decision'] == question_data['answer']
        }
    }

    return result


def run_mad_irac_mcq(
    question_data: Dict,
    client
) -> Dict:
    """
    Run MAD debate with IRAC structure on a single MCQ.

    Position assignment strategy (same as vanilla MAD):
    - Debater X: Generates IRAC opening argument without pre-assigned position
    - Debater Y: Gets a randomly selected position from remaining choices

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

    # Round 1: Debater X generates IRAC opening (chooses position freely)
    opening_x = debater_x.generate_opening_irac(
        question=question_data['question'],
        prompt_context=question_data['prompt'],
        choices=question_data['choices'],
        position=None  # Let debater choose
    )

    # Extract position chosen by Debater X
    pos_x = opening_x.get('position')
    if not pos_x or pos_x not in ['A', 'B', 'C', 'D']:
        raise ValueError(f"Debater X returned invalid position: {pos_x}")

    # Assign Debater Y a different position
    available_positions = get_available_positions_for_debater_y(pos_x)
    pos_y = random.choice(available_positions)

    # Round 1: Debater Y generates IRAC opening with assigned position
    opening_y = debater_y.generate_opening_irac(
        question=question_data['question'],
        prompt_context=question_data['prompt'],
        choices=question_data['choices'],
        position=pos_y
    )

    # Round 2: IRAC Rebuttals (sequential)
    rebuttal_x = debater_x.generate_rebuttal_irac(
        question=question_data['question'],
        prompt_context=question_data['prompt'],
        opponent_opening=opening_y
    )

    rebuttal_y = debater_y.generate_rebuttal_irac(
        question=question_data['question'],
        prompt_context=question_data['prompt'],
        opponent_opening=opening_x
    )

    # Judge decision with IRAC synthesis
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

    decision = judge.make_decision_irac(
        question=question_data['question'],
        prompt_context=question_data['prompt'],
        choices=question_data['choices'],
        debate_history=debate_history
    )

    # Compile result
    result = {
        'question_id': question_data['id'],
        'question': question_data['question'],
        'prompt': question_data['prompt'],
        'choices': question_data['choices'],
        'gold_answer': question_data['answer'],
        'gold_passage': question_data.get('gold_passage', ''),
        'position_assignment': {
            'debater_x_position': pos_x,
            'debater_y_position': pos_y,
            'strategy': 'X chooses freely, Y assigned different position (IRAC)'
        },
        'debate': {
            'round_1': {
                'debater_x': opening_x,
                'debater_y': opening_y
            },
            'round_2': {
                'debater_x': rebuttal_x,
                'debater_y': rebuttal_y
            }
        },
        'judge': {
            'rationale': decision.get('rationale', ''),
            'winner': decision.get('winner', 'unknown'),
            'decision': decision['decision'],
            'synthesis': decision.get('synthesis', {}),  # IRAC dict structure
            'correct': decision['decision'] == question_data['answer']
        }
    }

    return result


def run_mad_irac_hybrid_mcq(
    question_data: Dict,
    client
) -> Dict:
    """
    Run MAD debate with IRAC hybrid structure on a single MCQ.

    Hybrid configuration:
    - Opening: IRAC structured
    - Rebuttal: Vanilla (free-form)
    - Judge: Vanilla synthesis (string, not IRAC dict)

    Position assignment strategy (same as vanilla MAD):
    - Debater X: Generates IRAC opening argument without pre-assigned position
    - Debater Y: Gets a randomly selected position from remaining choices

    Args:
        question_data: Question dictionary from dataset
        client: Groq API client

    Returns:
        Complete result dictionary with hybrid debate logs and decision
    """
    # Initialize agents
    debater_x = Debater(client, name="Debater_X")
    debater_y = Debater(client, name="Debater_Y")
    judge = Judge(client)

    # Round 1: Debater X generates IRAC opening (chooses position freely)
    opening_x = debater_x.generate_opening_irac(
        question=question_data['question'],
        prompt_context=question_data['prompt'],
        choices=question_data['choices'],
        position=None  # Let debater choose
    )

    # Extract position chosen by Debater X
    pos_x = opening_x.get('position')
    if not pos_x or pos_x not in ['A', 'B', 'C', 'D']:
        raise ValueError(f"Debater X returned invalid position: {pos_x}")

    # Assign Debater Y a different position
    available_positions = get_available_positions_for_debater_y(pos_x)
    pos_y = random.choice(available_positions)

    # Round 1: Debater Y generates IRAC opening with assigned position
    opening_y = debater_y.generate_opening_irac(
        question=question_data['question'],
        prompt_context=question_data['prompt'],
        choices=question_data['choices'],
        position=pos_y
    )

    # Round 2: Vanilla Rebuttals (sequential)
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

    # Judge decision with hybrid prompt (IRAC openings + vanilla rebuttals/synthesis)
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

    decision = judge.make_decision_hybrid(
        question=question_data['question'],
        prompt_context=question_data['prompt'],
        choices=question_data['choices'],
        debate_history=debate_history
    )

    # Compile result
    result = {
        'question_id': question_data['id'],
        'question': question_data['question'],
        'prompt': question_data['prompt'],
        'choices': question_data['choices'],
        'gold_answer': question_data['answer'],
        'gold_passage': question_data.get('gold_passage', ''),
        'position_assignment': {
            'debater_x_position': pos_x,
            'debater_y_position': pos_y,
            'strategy': 'X chooses freely, Y assigned different position (IRAC Hybrid)'
        },
        'debate': {
            'round_1': {
                'debater_x': opening_x,
                'debater_y': opening_y
            },
            'round_2': {
                'debater_x': rebuttal_x,
                'debater_y': rebuttal_y
            }
        },
        'judge': {
            'rationale': decision.get('rationale', ''),
            'winner': decision.get('winner', 'unknown'),
            'decision': decision['decision'],
            'synthesis': decision.get('synthesis', ''),  # Vanilla string synthesis
            'correct': decision['decision'] == question_data['answer']
        }
    }

    return result


def run_experiments(
    dataset_name: str = "bar_exam_qa",
    sample_size: int = 50,
    output_dir: str = "results",
    provider: str = "groq",
    model: str = None
):
    """
    Run MAD experiments on dataset.

    Args:
        dataset_name: Name of dataset to use
        sample_size: Number of questions to process
        output_dir: Directory to save results
        provider: API provider ("groq" or "openrouter")
        model: Model name (optional, uses provider default)
    """
    start_time = time.time()

    # Get model name for display
    model_display = model if model else f"{provider}_default"

    print(f"=== MAD Experiment Started ===")
    print(f"Dataset: {dataset_name}")
    print(f"Sample size: {sample_size}")
    print(f"Provider: {provider}")
    print(f"Model: {model_display}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)

    # Load data
    if dataset_name == "bar_exam_qa":
        questions = load_bar_exam_qa(sample_size=sample_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Initialize client with retry logic
    client = create_client(provider=provider, model=model, max_tokens=1500, max_retries=10, retry_delay=2.0)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Sanitize model name for filename
    model_suffix = (model or f"{provider}_default").replace("/", "_").replace(":", "_")

    # Checkpoint: Load if exists
    checkpoint_file = output_path / f"checkpoint_mad_{dataset_name}_{sample_size}_{model_suffix}.json"
    if checkpoint_file.exists():
        print(f"Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            results = checkpoint['results']
            correct_count = checkpoint['correct_count']
            processed_ids = {r['question_id'] for r in results}
            questions = [q for q in questions if q['id'] not in processed_ids]
        print(f"Resuming from question {len(results) + 1}")
    else:
        results = []
        correct_count = 0

    question_times = []

    for idx, question in enumerate(tqdm(questions, desc="Processing questions"), 1):
        question_start = time.time()
        try:
            result = run_mad_mcq(question, client)
            results.append(result)

            if result['judge']['correct']:
                correct_count += 1

            question_time = time.time() - question_start
            question_times.append(question_time)

            # Print progress with timing
            current_accuracy = (correct_count / len(results)) * 100
            avg_time = sum(question_times) / len(question_times)
            remaining = sample_size - idx
            eta_seconds = remaining * avg_time
            eta_minutes = eta_seconds / 60

            print(f"\n[{idx}/{sample_size}] Question {result['question_id']} | "
                  f"Time: {question_time:.1f}s | "
                  f"Decision: {result['judge']['decision']} | "
                  f"Gold: {result['gold_answer']} | "
                  f"{'✓' if result['judge']['correct'] else '✗'} | "
                  f"Accuracy: {current_accuracy:.1f}% | "
                  f"ETA: {eta_minutes:.1f}min")

            # Checkpoint: Save every 50 questions
            if len(results) % 50 == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({'results': results, 'correct_count': correct_count}, f)
                print(f"[CHECKPOINT] Saved at {len(results)} questions")

        except Exception as e:
            print(f"\n[ERROR] Question {question['id']}: {e}")
            continue

    # Calculate total time
    total_time = time.time() - start_time
    avg_time_per_question = total_time / len(results) if results else 0

    # Checkpoint: Delete on successful completion
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    # Save results
    output_file = output_path / f"mad_vanilla_{dataset_name}_{sample_size}_{model_suffix}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    accuracy = correct_count / len(results) if results else 0
    print(f"\n{'=' * 40}")
    print(f"=== Experiment Complete ===")
    print(f"Total questions: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Avg time per question: {avg_time_per_question:.1f}s")
    print(f"Results saved to: {output_file}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)


def run_experiments_irac(
    dataset_name: str = "bar_exam_qa",
    sample_size: int = 50,
    output_dir: str = "results"
):
    """
    Run MAD+IRAC experiments on dataset.

    Args:
        dataset_name: Name of dataset to use
        sample_size: Number of questions to process
        output_dir: Directory to save results
    """
    start_time = time.time()

    print(f"=== MAD+IRAC Experiment Started ===")
    print(f"Dataset: {dataset_name}")
    print(f"Sample size: {sample_size}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)

    # Load data
    if dataset_name == "bar_exam_qa":
        questions = load_bar_exam_qa(sample_size=sample_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Initialize client with retry logic
    client = GroqClient(max_tokens=1000, max_retries=5, retry_delay=2.0)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run experiments
    results = []
    correct_count = 0
    question_times = []

    for idx, question in enumerate(tqdm(questions, desc="Processing questions (IRAC)"), 1):
        question_start = time.time()
        try:
            result = run_mad_irac_mcq(question, client)
            results.append(result)

            if result['judge']['correct']:
                correct_count += 1

            question_time = time.time() - question_start
            question_times.append(question_time)

            # Print progress with timing
            current_accuracy = (correct_count / len(results)) * 100
            avg_time = sum(question_times) / len(question_times)
            remaining = sample_size - idx
            eta_seconds = remaining * avg_time
            eta_minutes = eta_seconds / 60

            print(f"\n[{idx}/{sample_size}] Question {result['question_id']} | "
                  f"Time: {question_time:.1f}s | "
                  f"Decision: {result['judge']['decision']} | "
                  f"Gold: {result['gold_answer']} | "
                  f"{'✓' if result['judge']['correct'] else '✗'} | "
                  f"Accuracy: {current_accuracy:.1f}% | "
                  f"ETA: {eta_minutes:.1f}min")

        except Exception as e:
            print(f"\n[ERROR] Question {question['id']}: {e}")
            continue

    # Calculate total time
    total_time = time.time() - start_time
    avg_time_per_question = total_time / len(results) if results else 0

    # Save results
    output_file = output_path / f"mad_irac_{dataset_name}_{sample_size}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    accuracy = correct_count / len(results) if results else 0
    print(f"\n{'=' * 40}")
    print(f"=== MAD+IRAC Experiment Complete ===")
    print(f"Total questions: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Avg time per question: {avg_time_per_question:.1f}s")
    print(f"Results saved to: {output_file}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)


def run_experiments_irac_hybrid(
    dataset_name: str = "bar_exam_qa",
    sample_size: int = 50,
    output_dir: str = "results",
    provider: str = "groq",
    model: str = None
):
    """
    Run MAD+IRAC Hybrid experiments on dataset.

    Hybrid configuration:
    - Openings: IRAC structured
    - Rebuttals: Vanilla
    - Judge: Vanilla synthesis

    Args:
        dataset_name: Name of dataset to use
        sample_size: Number of questions to process
        output_dir: Directory to save results
        provider: API provider ("groq" or "openrouter")
        model: Model name (optional, uses provider default)
    """
    start_time = time.time()

    # Get model name for display
    model_display = model if model else f"{provider}_default"

    print(f"=== MAD+IRAC Hybrid Experiment Started ===")
    print(f"Dataset: {dataset_name}")
    print(f"Sample size: {sample_size}")
    print(f"Provider: {provider}")
    print(f"Model: {model_display}")
    print(f"Configuration: IRAC openings + Vanilla rebuttals/judge")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)

    # Load data
    if dataset_name == "bar_exam_qa":
        questions = load_bar_exam_qa(sample_size=sample_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Initialize client with retry logic
    client = create_client(provider=provider, model=model, max_tokens=2000, max_retries=10, retry_delay=2.0)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Sanitize model name for filename
    model_suffix = (model or f"{provider}_default").replace("/", "_").replace(":", "_")

    # Checkpoint: Load if exists
    checkpoint_file = output_path / f"checkpoint_mad_irac_hybrid_{dataset_name}_{sample_size}_{model_suffix}.json"
    if checkpoint_file.exists():
        print(f"Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            results = checkpoint['results']
            correct_count = checkpoint['correct_count']
            processed_ids = {r['question_id'] for r in results}
            questions = [q for q in questions if q['id'] not in processed_ids]
        print(f"Resuming from question {len(results) + 1}")
    else:
        results = []
        correct_count = 0

    question_times = []

    for idx, question in enumerate(tqdm(questions, desc="Processing questions (IRAC Hybrid)"), 1):
        question_start = time.time()
        try:
            result = run_mad_irac_hybrid_mcq(question, client)
            results.append(result)

            if result['judge']['correct']:
                correct_count += 1

            question_time = time.time() - question_start
            question_times.append(question_time)

            # Print progress with timing
            current_accuracy = (correct_count / len(results)) * 100
            avg_time = sum(question_times) / len(question_times)
            remaining = sample_size - idx
            eta_seconds = remaining * avg_time
            eta_minutes = eta_seconds / 60

            print(f"\n[{idx}/{sample_size}] Question {result['question_id']} | "
                  f"Time: {question_time:.1f}s | "
                  f"Decision: {result['judge']['decision']} | "
                  f"Gold: {result['gold_answer']} | "
                  f"{'✓' if result['judge']['correct'] else '✗'} | "
                  f"Accuracy: {current_accuracy:.1f}% | "
                  f"ETA: {eta_minutes:.1f}min")

            # Checkpoint: Save every 50 questions
            if len(results) % 50 == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({'results': results, 'correct_count': correct_count}, f)
                print(f"[CHECKPOINT] Saved at {len(results)} questions")

        except Exception as e:
            print(f"\n[ERROR] Question {question['id']}: {e}")
            continue

    # Calculate total time
    total_time = time.time() - start_time
    avg_time_per_question = total_time / len(results) if results else 0

    # Checkpoint: Delete on successful completion
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    # Save results
    output_file = output_path / f"mad_irac_hybrid_{dataset_name}_{sample_size}_{model_suffix}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    accuracy = correct_count / len(results) if results else 0
    print(f"\n{'=' * 40}")
    print(f"=== MAD+IRAC Hybrid Experiment Complete ===")
    print(f"Total questions: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Avg time per question: {avg_time_per_question:.1f}s")
    print(f"Results saved to: {output_file}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MAD experiments with different providers and models")
    parser.add_argument("--provider", type=str, choices=["groq", "openrouter"], default="groq", help="API provider (default: groq)")
    parser.add_argument("--model", type=str, default=None, help="Model name (optional)")
    parser.add_argument("--experiment", type=str, choices=["vanilla", "irac_hybrid"], default="irac_hybrid", help="Experiment type (default: irac_hybrid)")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of questions (default: 5)")
    parser.add_argument("--dataset", type=str, default="bar_exam_qa", help="Dataset name (default: bar_exam_qa)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory (default: results)")
    args = parser.parse_args()

    if args.experiment == "vanilla":
        run_experiments(
            dataset_name=args.dataset,
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            provider=args.provider,
            model=args.model
        )
    elif args.experiment == "irac_hybrid":
        run_experiments_irac_hybrid(
            dataset_name=args.dataset,
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            provider=args.provider,
            model=args.model
        )
