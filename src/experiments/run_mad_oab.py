"""
Run Multi-Agent Debate experiments on OAB open-ended questions.
Supports both IRAC and Vanilla modes.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

from src.utils.data_loader import load_oab_with_guidelines
from src.agents.debater_experimental import Debater
from src.agents.judge_experimental import Judge
from bert_score import BERTScorer


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


def create_bert_scorer(model_type: Optional[str] = None) -> Optional[BERTScorer]:
    """
    Initialize a BERTScorer for Portuguese answers.

    Args:
        model_type: Optional HF model name. If None, uses language-specific default.

    Returns:
        BERTScorer instance or None if initialization fails.
    """
    try:
        if model_type:
            return BERTScorer(model_type=model_type, rescale_with_baseline=True)
        # Default Portuguese scorer from BERTScore
        return BERTScorer(lang="pt", rescale_with_baseline=True)
    except Exception as exc:
        print(f"[WARNING] Could not initialize BERTScorer ({exc}). "
              f"BERT-based accuracy will be skipped.")
        return None


def compute_bert_accuracy(
    prediction: str,
    reference: str,
    scorer: Optional[BERTScorer]
) -> Optional[float]:
    """
    Compute BERTScore F1 between prediction and reference to approximate accuracy.

    Returns:
        Float score in [0, 1] or None if scorer/reference unavailable.
    """
    if scorer is None:
        return None

    prediction = (prediction or "").strip()
    reference = (reference or "").strip()
    if not prediction or not reference:
        return None

    try:
        _, _, f1 = scorer.score([prediction], [reference])
        return float(f1.mean().item())
    except Exception as exc:
        print(f"[WARNING] Failed to compute BERTScore: {exc}")
        return None


def run_mad_oab(
    question_data: Dict,
    client,
    mode: str = "irac"
) -> Dict:
    """
    Run MAD debate on single OAB open-ended question.

    Debate structure:
    - Debater X: Opening argument (neutral)
    - Debater Y: Adversarial opening argument
    - Debater X: Rebuttal
    - Debater Y: Rebuttal
    - Judge: Final synthesis

    Args:
        question_data: Question dictionary from OAB dataset
        client: API client
        mode: "irac" or "vanilla"

    Returns:
        Complete result dictionary with debate logs and synthesis
    """
    # Initialize agents
    debater_x = Debater(client, name="Debater_X")
    debater_y = Debater(client, name="Debater_Y")
    judge = Judge(client)

    # Extract question info
    question = question_data['statement']
    category = question_data['category']

    # Select methods based on mode
    if mode == "irac":
        # Round 1: Debater X opening (neutral, IRAC)
        opening_x = debater_x.generate_opening_oab(
            question=question,
            category=category
        )

        # Round 1: Debater Y opening (adversarial - sees X's position, IRAC)
        opening_y = debater_y.generate_opening_oab_adversarial(
            question=question,
            category=category,
            opponent_opening=opening_x
        )

        # Round 2: Rebuttals (IRAC)
        rebuttal_x = debater_x.generate_rebuttal_oab(
            question=question,
            category=category,
            opponent_opening=opening_y
        )

        rebuttal_y = debater_y.generate_rebuttal_oab(
            question=question,
            category=category,
            opponent_opening=opening_x
        )

        # Judge synthesis (IRAC)
        synthesis = judge.synthesize_answer_oab(
            question=question,
            category=category,
            debater_x_rebuttal=rebuttal_x,
            debater_y_rebuttal=rebuttal_y
        )

    elif mode == "vanilla":
        # Round 1: Debater X opening (neutral, vanilla)
        opening_x = debater_x.generate_opening_oab_vanilla(
            question=question,
            category=category
        )

        # Round 1: Debater Y opening (adversarial - sees X's position, vanilla)
        opening_y = debater_y.generate_opening_oab_adversarial_vanilla(
            question=question,
            category=category,
            opponent_opening=opening_x
        )

        # Round 2: Rebuttals (vanilla)
        rebuttal_x = debater_x.generate_rebuttal_oab_vanilla(
            question=question,
            category=category,
            opponent_opening=opening_y
        )

        rebuttal_y = debater_y.generate_rebuttal_oab_vanilla(
            question=question,
            category=category,
            opponent_opening=opening_x
        )

        # Judge synthesis (vanilla)
        synthesis = judge.synthesize_answer_oab_vanilla(
            question=question,
            category=category,
            debater_x_rebuttal=rebuttal_x,
            debater_y_rebuttal=rebuttal_y
        )

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'irac' or 'vanilla'")

    # Compile result
    result = {
        'question_id': question_data['question_id'],
        'question': question,
        'category': category,
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
            'final_answer': synthesis.get('final_answer', ''),
            'rationale': synthesis.get('rationale', ''),
            'sources_used': synthesis.get('sources_used', {}),
            'key_citations': synthesis.get('key_citations', [])
        },
        'ground_truth': question_data.get('ground_truth', {
            'reference_answer': '',
            'key_citations_expected': []
        })
    }

    return result


def run_experiments_oab(
    sample_size: int = 5,
    output_dir: str = "results",
    provider: str = "openrouter",
    model: str = None,
    mode: str = "irac",
    bert_model: Optional[str] = None,
    accuracy_metric: str = "bertscore",
    accuracy_threshold: Optional[float] = None
):
    """
    Run MAD experiments on OAB open-ended questions.

    Args:
        sample_size: Number of questions to process (max 105)
        output_dir: Directory to save results
        provider: API provider ("groq" or "openrouter")
        model: Model name (optional, uses provider default)
        mode: "irac" or "vanilla"
    """
    start_time = time.time()

    # Get model name for display
    model_display = model if model else f"{provider}_default"

    print(f"=== MAD OAB Experiment Started ===")
    print(f"Dataset: OAB-Bench (open-ended)")
    print(f"Mode: MAD {mode.upper()}")
    print(f"Sample size: {sample_size}")
    print(f"Provider: {provider}")
    print(f"Model: {model_display}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)

    # Validate accuracy settings
    accuracy_metric = accuracy_metric.lower()
    if accuracy_metric != "bertscore":
        raise ValueError(f"Unsupported accuracy metric '{accuracy_metric}'. Only 'bertscore' is available.")
    if accuracy_threshold is not None and not (0.0 <= accuracy_threshold <= 1.0):
        raise ValueError("Accuracy threshold must be between 0 and 1.")

    # Load data with ground truth
    print(f"Loading OAB-Bench questions with ground truth...")
    questions = load_oab_with_guidelines(sample_size=sample_size)

    # Initialize client with retry logic
    client = create_client(provider=provider, model=model, max_tokens=2000, max_retries=10, retry_delay=2.0)

    # Initialize BERT scorer once for reuse
    bert_scorer = create_bert_scorer(model_type=bert_model)
    if bert_scorer:
        print(f"BERT accuracy: using {'custom model '+bert_model if bert_model else 'default Portuguese baseline'}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Sanitize model name for filename
    model_suffix = (model or f"{provider}_default").replace("/", "_").replace(":", "_")

    # Checkpoint: Load if exists
    checkpoint_file = output_path / f"checkpoint_mad_oab_{mode}_{sample_size}_{model_suffix}.json"
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
    bert_scores = []
    accuracy_hits = []

    for idx, question in enumerate(tqdm(questions, desc="Processing OAB questions"), 1):
        question_start = time.time()
        try:
            result = run_mad_oab(question, client, mode=mode)
            results.append(result)

            # Compute BERTScore-based accuracy
            reference_answer = result['ground_truth'].get('reference_answer', '')
            bert_score = compute_bert_accuracy(
                prediction=result['judge']['final_answer'],
                reference=reference_answer,
                scorer=bert_scorer
            )
            accuracy_pass = None
            if bert_score is not None:
                bert_scores.append(bert_score)
                if accuracy_threshold is not None:
                    accuracy_pass = bert_score >= accuracy_threshold
                    accuracy_hits.append(1 if accuracy_pass else 0)

            result['metrics'] = {
                'bert_score_f1': bert_score,
                'accuracy_pass': accuracy_pass
            }

            question_time = time.time() - question_start
            question_times.append(question_time)

            # Print progress with timing
            avg_time = sum(question_times) / len(question_times)
            remaining = sample_size - idx
            eta_seconds = remaining * avg_time
            eta_minutes = eta_seconds / 60

            answer_length = len(result['judge']['final_answer'])
            progress_msg = (
                f"\n[{idx}/{sample_size}] Question {result['question_id']} | "
                f"Category: {result['category']} | "
                f"Time: {question_time:.1f}s | "
                f"Answer length: {answer_length} chars | "
                f"ETA: {eta_minutes:.1f}min"
            )
            if bert_score is not None:
                progress_msg += f" | BERT-F1: {bert_score:.3f}"
                if accuracy_pass is not None:
                    status = "PASS" if accuracy_pass else "FAIL"
                    progress_msg += f" @ {accuracy_threshold:.2f}: {status}"
            print(progress_msg)

            # Checkpoint: Save every 10 questions
            if len(results) % 10 == 0:
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump({'results': results}, f, ensure_ascii=False)
                print(f"[CHECKPOINT] Saved at {len(results)} questions")

        except Exception as e:
            print(f"\n[ERROR] Question {question['question_id']}: {e}")
            # Save error result
            results.append({
                'question_id': question['question_id'],
                'error': str(e)
            })
            continue

    # Calculate total time
    total_time = time.time() - start_time
    avg_time_per_question = total_time / len(results) if results else 0

    # Checkpoint: Delete on successful completion
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    # Save results
    output_file = output_path / f"mad_oab_{mode}_{sample_size}_{model_suffix}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    successful = len([r for r in results if 'error' not in r])
    errors = len([r for r in results if 'error' in r])
    avg_bert_score = sum(bert_scores) / len(bert_scores) if bert_scores else None
    accuracy_rate = (sum(accuracy_hits) / len(accuracy_hits)) if accuracy_hits else None

    print(f"\n{'=' * 40}")
    print(f"=== MAD OAB {mode.upper()} Experiment Complete ===")
    print(f"Total questions: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Errors: {errors}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Avg time per question: {avg_time_per_question:.1f}s")
    if avg_bert_score is not None:
        print(f"BERTScore accuracy (avg F1): {avg_bert_score:.4f} over {len(bert_scores)} evaluated answers")
    else:
        print("BERTScore accuracy not computed (missing scorer or references)")
    if accuracy_rate is not None:
        print(f"Accuracy@{accuracy_threshold:.2f}: {accuracy_rate*100:.1f}% ({len(accuracy_hits)} evaluated)")
    print(f"Results saved to: {output_file}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MAD on OAB open-ended questions")
    parser.add_argument("--provider", type=str, choices=["groq", "openrouter"], default="openrouter",
                        help="API provider (default: openrouter)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (optional)")
    parser.add_argument("--sample-size", type=int, default=5,
                        help="Number of questions (default: 5, max: 105)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--mode", type=str, choices=["irac", "vanilla"], default="irac",
                        help="MAD mode: 'irac' (structured) or 'vanilla' (simple) (default: irac)")
    parser.add_argument("--bert-model", type=str, default=None,
                        help="Optional HuggingFace model for BERTScore accuracy (default: multilingual Portuguese baseline)")
    parser.add_argument("--accuracy-metric", type=str, default="bertscore",
                        choices=["bertscore"],
                        help="Accuracy metric for open-ended evaluation (default: bertscore)")
    parser.add_argument("--accuracy-threshold", type=float, default=None,
                        help="Optional threshold (0-1) to convert metric to pass/fail accuracy")
    args = parser.parse_args()

    run_experiments_oab(
        sample_size=args.sample_size,
        output_dir=args.output_dir,
        provider=args.provider,
        model=args.model,
        mode=args.mode,
        bert_model=args.bert_model,
        accuracy_metric=args.accuracy_metric,
        accuracy_threshold=args.accuracy_threshold
    )
