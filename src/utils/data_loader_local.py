"""
Local Bar Exam QA loader (HuggingFace Arrow dataset)
"""

from typing import List, Dict, Optional
from pathlib import Path
from datasets import load_from_disk


def load_bar_exam_qa_local(
    sample_size: Optional[int] = None,
    split: str = "train"
) -> List[Dict]:
    """
    Load Bar Exam QA dataset from:
        src/datasets/barexam_qa_dataset/
    """

    print(f"[LOCAL] Loading Bar Exam QA dataset — split = {split}")

    # PROJECT ROOT = Legal_MAD/
    project_root = Path(__file__).resolve().parent.parent.parent

    # CORRECT DATASET LOCATION:
    dataset_path = project_root / "src" / "datasets" / "barexam_qa_dataset"

    if not dataset_path.exists():
        raise FileNotFoundError(f"[ERROR] Dataset not found at: {dataset_path}")

    print(f"[LOCAL] Dataset path: {dataset_path}")

    # Load HF dataset from disk
    ds = load_from_disk(str(dataset_path))[split]
    print(f"[LOCAL] Loaded {len(ds)} items from split '{split}'")

    questions = []
    for i, row in enumerate(ds):
        if sample_size and len(questions) >= sample_size:
            break

        q = {
            "id": str(row.get("idx", i)),
            "prompt": row.get("prompt", ""),
            "question": row["question"],
            "choices": [
                row["choice_a"],
                row["choice_b"],
                row["choice_c"],
                row["choice_d"],
            ],
            "answer": row["answer"].strip(),
            "gold_passage": row.get("gold_passage", ""),
            "gold_idx": row.get("gold_idx", ""),
        }

        questions.append(q)

    print(f"[LOCAL] Returning {len(questions)} formatted questions")
    return questions


if __name__ == "__main__":
    sample = load_bar_exam_qa_local(sample_size=3, split="train")
    print(sample[0])