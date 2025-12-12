"""
Data loader for legal QA datasets.
"""

from typing import List, Dict, Optional
import pandas as pd
from huggingface_hub import hf_hub_download


def load_bar_exam_qa(sample_size: Optional[int] = None, split: str = "train") -> List[Dict]:
    """
    Load Bar Exam QA dataset from HuggingFace (reglab/barexam_qa).

    Downloads CSV files directly from the repository to avoid the deprecated
    loading script system.

    Args:
        sample_size: Number of questions to sample (None = all)
        split: Dataset split to load ('train', 'validation', or 'test')

    Returns:
        List of question dictionaries with structure:
        {
            'id': str,
            'prompt': str,
            'question': str,
            'choices': List[str],
            'answer': str,
            'gold_passage': str,
            'gold_idx': str
        }
    """
    print(f"Loading Bar Exam QA dataset ({split} split)...")

    # Download the CSV file from HuggingFace Hub
    file_path = hf_hub_download(
        repo_id="reglab/barexam_qa",
        filename=f"data/qa/{split}.csv",
        repo_type="dataset"
    )

    # Load CSV with pandas
    df = pd.read_csv(file_path)

    questions = []
    for idx, row in df.iterrows():
        if sample_size and len(questions) >= sample_size:
            break

        question_dict = {
            'id': str(row.get('idx', idx)),
            'prompt': str(row.get('prompt', '')),
            'question': str(row['question']),
            'choices': [
                str(row['choice_a']),
                str(row['choice_b']),
                str(row['choice_c']),
                str(row['choice_d'])
            ],
            'answer': str(row['answer']),
            'gold_passage': str(row.get('gold_passage', '')),
            'gold_idx': str(row.get('gold_idx', ''))
        }

        questions.append(question_dict)

    print(f"Loaded {len(questions)} questions from Bar Exam QA ({split} split)")
    return questions


def load_oab_open_ended(sample_size: Optional[int] = None) -> List[Dict]:
    """
    Load OAB open-ended questions from HuggingFace (maritaca-ai/oab-bench).

    Filters to include only dissertative questions (QUESTÃO), excluding
    procedural pieces (PEÇA PRÁTICO-PROFISSIONAL).

    Explodes multi-turn questions into separate questions by combining
    the general statement (enunciado) with each sub-question (turn).

    Args:
        sample_size: Number of questions to sample (None = all)

    Returns:
        List of question dictionaries with structure:
        {
            'question_id': str (includes turn index, e.g., 'original_id_turn_0'),
            'category': str (formatted as "Direito X"),
            'statement': str (combined enunciado + sub-question),
            'turn_index': int,
            'values': List[float],
            'system': str
        }
    """
    from datasets import load_dataset

    print(f"Loading OAB-Bench dataset (dissertative questions only)...")

    # Load questions subset from HuggingFace
    dataset = load_dataset("maritaca-ai/oab-bench", "questions", split="train")

    questions = []
    for item in dataset:
        # Skip procedural pieces (PEÇA PRÁTICO-PROFISSIONAL)
        if 'peca_praticoprofissional' in item['question_id']:
            continue

        # Format category: "39_direito_administrativo" -> "Direito Administrativo"
        category_raw = item['category']
        # Remove exam number prefix (e.g., "39_")
        category_parts = category_raw.split('_', 1)
        if len(category_parts) > 1:
            category_clean = category_parts[1].replace('_', ' ').title()
        else:
            category_clean = category_raw.replace('_', ' ').title()

        # Explode each turn into a separate question
        for turn_idx, turn_text in enumerate(item['turns']):
            if not turn_text.strip():  # Skip empty turns
                continue

            # Combine statement + turn for complete question
            full_question = f"{item['statement']}\n\n{turn_text}"

            question_dict = {
                'question_id': f"{item['question_id']}_turn_{turn_idx}",
                'category': category_clean,
                'statement': full_question,
                'turn_index': turn_idx,
                'values': item.get('values', []),
                'system': item.get('system', '')
            }

            questions.append(question_dict)

            if sample_size and len(questions) >= sample_size:
                break

        if sample_size and len(questions) >= sample_size:
            break

    print(f"Loaded {len(questions)} dissertative questions from OAB-Bench (exploded from turns)")
    return questions


def load_oab_guidelines(sample_size: Optional[int] = None) -> Dict[str, Dict]:
    """
    Load OAB guidelines (ground truth) from HuggingFace (maritaca-ai/oab-bench).

    Filters to include only dissertative questions, excluding procedural pieces.
    Explodes turns to match the questions structure.

    Args:
        sample_size: Number of guidelines to sample (None = all)

    Returns:
        Dictionary mapping question_id to guideline data:
        {
            'question_id_turn_0': {
                'reference_answer': str (ground truth reference answer from OAB espelho),
                'key_citations_expected': List[str] (for future citation evaluation)
            }
        }
    """
    from datasets import load_dataset

    print(f"Loading OAB-Bench guidelines (ground truth)...")

    # Load guidelines subset from HuggingFace
    dataset = load_dataset("maritaca-ai/oab-bench", "guidelines", split="train")

    guidelines_dict = {}
    count = 0

    for item in dataset:
        # Skip procedural pieces
        if 'peca_praticoprofissional' in item['question_id']:
            continue

        # Extract turns from choices (guidelines structure)
        choices = item.get('choices', [])
        if not choices:
            continue

        # Get turns from first choice (contains reference answers)
        turns = choices[0].get('turns', [])

        # Explode each turn into a separate guideline entry
        for turn_idx, turn_text in enumerate(turns):
            if not turn_text.strip():
                continue

            # Create question_id matching the questions format
            guideline_id = f"{item['question_id']}_turn_{turn_idx}"

            # Extract reference answer (ignore rubric table)
            # Format: "Answer text...\n\nDISTRIBUIÇÃO DOS PONTOS\n\n| ITEM | PONTUAÇÃO |"
            # Split at the markdown table start (| ITEM)
            if '| ITEM' in turn_text:
                reference_answer = turn_text.split('| ITEM', 1)[0].strip()
            else:
                reference_answer = turn_text.strip()

            # Clean up: remove trailing "DISTRIBUIÇÃO DOS PONTOS" if present
            if reference_answer.endswith('DISTRIBUIÇÃO DOS PONTOS'):
                reference_answer = reference_answer.replace('DISTRIBUIÇÃO DOS PONTOS', '').strip()

            guidelines_dict[guideline_id] = {
                'reference_answer': reference_answer,
                'key_citations_expected': []  # TODO: parse citations (Art. X Lei Y, Súmula Z) from reference_answer
            }

            count += 1
            if sample_size and count >= sample_size:
                break

        if sample_size and count >= sample_size:
            break

    print(f"Loaded {len(guidelines_dict)} guidelines from OAB-Bench")
    return guidelines_dict


def load_oab_with_guidelines(sample_size: Optional[int] = None) -> List[Dict]:
    """
    Load OAB questions WITH ground truth guidelines.

    Combines questions and guidelines into unified dataset.

    Args:
        sample_size: Number of questions to load (None = all)

    Returns:
        List of question dictionaries with ground_truth field:
        {
            'question_id': str,
            'category': str,
            'statement': str,
            'turn_index': int,
            'values': List[float],
            'system': str,
            'ground_truth': {
                'reference_answer': str (ground truth from OAB espelho),
                'key_citations_expected': List[str] (for future citation evaluation)
            }
        }
    """
    print(f"Loading OAB-Bench with ground truth...")

    # Load questions
    questions = load_oab_open_ended(sample_size=sample_size)

    # Load guidelines (load all to ensure we have matches)
    guidelines = load_oab_guidelines(sample_size=None)

    # Combine: add ground_truth to each question
    for question in questions:
        question_id = question['question_id']

        if question_id in guidelines:
            question['ground_truth'] = guidelines[question_id]
        else:
            # Question without guideline (shouldn't happen, but handle gracefully)
            question['ground_truth'] = {
                'reference_answer': '',
                'key_citations_expected': []
            }
            print(f"Warning: No guideline found for {question_id}")

    print(f"Combined {len(questions)} questions with ground truth")
    return questions


if __name__ == "__main__":
    # Test data loading
    questions = load_bar_exam_qa(sample_size=5)

    print("\nSample question:")
    q = questions[0]
    print(f"ID: {q['id']}")
    print(f"Prompt: {q['prompt'][:100] if q['prompt'] else 'None'}...")
    print(f"Question: {q['question'][:100]}...")
    print(f"Choices: A) {q['choices'][0][:50]}...")
    print(f"Answer: {q['answer']}")
    print(f"Gold passage length: {len(q['gold_passage'])} chars")
