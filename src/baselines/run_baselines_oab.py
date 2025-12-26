"""
Run baseline experiments on OAB open-ended questions.
Supports: single-agent, cot, self-consistency.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from collections import Counter

from src.utils.data_loader import load_oab_with_guidelines
from src.baselines.prompts_oab_baselines import (
    get_single_agent_prompt_oab,
    get_cot_prompt_oab,
    get_self_consistency_prompt_oab
)


def create_client(provider: str = "openrouter", model: str = None, max_tokens: int = 2000, max_retries: int = 10, retry_delay: float = 2.0):
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


def run_single_agent_oab(question_data: Dict, client) -> Dict:
    """
    Run single-agent baseline on OAB question.

    Args:
        question_data: Question dictionary
        client: API client

    Returns:
        Result dictionary
    """
    question = question_data['statement']
    category = question_data['category']

    # Generate prompt
    prompt = get_single_agent_prompt_oab(question, category)

    # Get response
    response = client.generate_json(prompt, max_tokens=1500)

    # Compile result
    result = {
        'question_id': question_data['question_id'],
        'question': question,
        'category': category,
        'answer': response.get('answer', ''),
        'key_citations': response.get('key_citations', []),
        'ground_truth': question_data.get('ground_truth', {
            'reference_answer': '',
            'key_citations_expected': []
        })
    }

    return result


def run_cot_oab(question_data: Dict, client) -> Dict:
    """
    Run Chain-of-Thought baseline on OAB question.

    Args:
        question_data: Question dictionary
        client: API client

    Returns:
        Result dictionary
    """
    question = question_data['statement']
    category = question_data['category']

    # Generate prompt
    prompt = get_cot_prompt_oab(question, category)

    # Get response
    response = client.generate_json(prompt, max_tokens=2000)

    # Compile result
    result = {
        'question_id': question_data['question_id'],
        'question': question,
        'category': category,
        'reasoning': response.get('reasoning', ''),
        'answer': response.get('answer', ''),
        'key_citations': response.get('key_citations', []),
        'ground_truth': question_data.get('ground_truth', {
            'reference_answer': '',
            'key_citations_expected': []
        })
    }

    return result


def run_self_consistency_oab(question_data: Dict, client, num_samples: int = 5) -> Dict:
    """
    Run Self-Consistency baseline on OAB question.

    Generates N responses and selects most common answer.

    Args:
        question_data: Question dictionary
        client: API client
        num_samples: Number of samples to generate

    Returns:
        Result dictionary
    """
    question = question_data['statement']
    category = question_data['category']

    # Generate N responses
    responses = []
    for i in range(num_samples):
        prompt = get_self_consistency_prompt_oab(question, category)
        response = client.generate_json(prompt, max_tokens=2000)
        responses.append(response)

    # Extract answers
    answers = [r.get('answer', '') for r in responses]

    # Simple voting: choose most common answer (by first 100 chars to handle minor variations)
    answer_keys = [ans[:100] for ans in answers]
    vote_counter = Counter(answer_keys)
    most_common_key = vote_counter.most_common(1)[0][0]

    # Find full answer matching the most common key
    final_answer = ''
    for ans in answers:
        if ans[:100] == most_common_key:
            final_answer = ans
            break

    # Aggregate citations
    all_citations = []
    for r in responses:
        all_citations.extend(r.get('key_citations', []))
    unique_citations = list(set(all_citations))

    # Compile result
    result = {
        'question_id': question_data['question_id'],
        'question': question,
        'category': category,
        'num_samples': num_samples,
        'all_responses': responses,  # Store all for analysis
        'answer': final_answer,
        'key_citations': unique_citations,
        'ground_truth': question_data.get('ground_truth', {
            'reference_answer': '',
            'key_citations_expected': []
        })
    }

    return result


def run_experiments_baselines_oab(
    baseline: str = "single",
    sample_size: int = 5,
    output_dir: str = "results",
    provider: str = "openrouter",
    model: str = None,
    num_samples: int = 5  # For self-consistency
):
    """
    Run baseline experiments on OAB open-ended questions.

    Args:
        baseline: Baseline type ("single", "cot", "sc")
        sample_size: Number of questions to process
        output_dir: Directory to save results
        provider: API provider ("groq" or "openrouter")
        model: Model name (optional)
        num_samples: Number of samples for self-consistency
    """
    start_time = time.time()

    # Get model name for display
    model_display = model if model else f"{provider}_default"

    print(f"=== Baseline OAB Experiment Started ===")
    print(f"Dataset: OAB-Bench (open-ended)")
    print(f"Baseline: {baseline.upper()}")
    print(f"Sample size: {sample_size}")
    print(f"Provider: {provider}")
    print(f"Model: {model_display}")
    if baseline == "sc":
        print(f"SC num_samples: {num_samples}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)

    # Load data
    print(f"Loading OAB-Bench questions with ground truth...")
    questions = load_oab_with_guidelines(sample_size=sample_size)

    # Initialize client
    client = create_client(provider=provider, model=model, max_tokens=2000, max_retries=10, retry_delay=2.0)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Sanitize model name for filename
    model_suffix = (model or f"{provider}_default").replace("/", "_").replace(":", "_")

    # Checkpoint file
    checkpoint_file = output_path / f"checkpoint_baseline_oab_{baseline}_{sample_size}_{model_suffix}.json"
    if checkpoint_file.exists():
        print(f"Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            results = checkpoint['results']
            processed_ids = {r['question_id'] for r in results}
            questions = [q for q in questions if q['question_id'] not in processed_ids]
        print(f"Resuming from question {len(results) + 1}")
    else:
        results = []

    question_times = []

    # Select baseline function
    if baseline == "single":
        run_func = run_single_agent_oab
    elif baseline == "cot":
        run_func = run_cot_oab
    elif baseline == "sc":
        run_func = lambda q, c: run_self_consistency_oab(q, c, num_samples=num_samples)
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    # Run experiments
    for idx, question in enumerate(tqdm(questions, desc=f"Processing {baseline.upper()} baseline"), 1):
        question_start = time.time()
        try:
            result = run_func(question, client)
            results.append(result)

            question_time = time.time() - question_start
            question_times.append(question_time)

            # Print progress
            avg_time = sum(question_times) / len(question_times)
            remaining = sample_size - (len(results) - len([r for r in results if 'error' in r]))
            eta_seconds = remaining * avg_time
            eta_minutes = eta_seconds / 60

            answer_length = len(result.get('answer', ''))
            print(f"\n[{idx}/{sample_size}] Question {result['question_id']} | "
                  f"Category: {result['category']} | "
                  f"Time: {question_time:.1f}s | "
                  f"Answer length: {answer_length} chars | "
                  f"ETA: {eta_minutes:.1f}min")

            # Checkpoint every 10 questions
            if len(results) % 10 == 0:
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump({'results': results}, f, ensure_ascii=False)
                print(f"[CHECKPOINT] Saved at {len(results)} questions")

        except Exception as e:
            print(f"\n[ERROR] Question {question['question_id']}: {e}")
            results.append({
                'question_id': question['question_id'],
                'error': str(e)
            })
            continue

    # Calculate total time
    total_time = time.time() - start_time
    avg_time_per_question = total_time / len(results) if results else 0

    # Delete checkpoint on success
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    # Save results
    output_file = output_path / f"baseline_oab_{baseline}_{sample_size}_{model_suffix}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    successful = len([r for r in results if 'error' not in r])
    errors = len([r for r in results if 'error' in r])
    print(f"\n{'=' * 40}")
    print(f"=== Baseline {baseline.upper()} OAB Experiment Complete ===")
    print(f"Total questions: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Errors: {errors}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Avg time per question: {avg_time_per_question:.1f}s")
    print(f"Results saved to: {output_file}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baselines on OAB open-ended questions")
    parser.add_argument("--baseline", type=str, choices=["single", "cot", "sc"], required=True,
                        help="Baseline type: 'single' (direct), 'cot' (chain-of-thought), 'sc' (self-consistency)")
    parser.add_argument("--provider", type=str, choices=["groq", "openrouter"], default="openrouter",
                        help="API provider (default: openrouter)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (optional)")
    parser.add_argument("--sample-size", type=int, default=5,
                        help="Number of questions (default: 5, max: 105)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of samples for self-consistency (default: 5)")
    args = parser.parse_args()

    run_experiments_baselines_oab(
        baseline=args.baseline,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
        provider=args.provider,
        model=args.model,
        num_samples=args.num_samples
    )
