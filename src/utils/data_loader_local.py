"""
SAFE Local Bar Exam QA loader (HuggingFace Arrow dataset)
Prevents silent empty output & guarantees SC baseline works.
"""

from typing import List, Dict, Optional
from pathlib import Path
from datasets import load_from_disk


def load_bar_exam_qa_local(
    sample_size: Optional[int] = None,
    split: str = "train"
) -> List[Dict]:
    print(f"[LOCAL] Loading Bar Exam QA dataset — split = {split}")

    # PROJECT ROOT = Legal_MAD/
    project_root = Path(__file__).resolve().parent.parent.parent

    dataset_path = project_root / "src" / "datasets" / "barexam_qa_dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(f"[ERROR] Dataset not found at: {dataset_path}")

    print(f"[LOCAL] Dataset path: {dataset_path}")

    # Load HF dataset
    ds = load_from_disk(str(dataset_path))[split]
    print(f"[LOCAL] Loaded {len(ds)} items from split '{split}'")

    questions = []

    # -----------------------------
    # SAFE ITERATION (fix empty bug)
    # -----------------------------
    for i, row in enumerate(ds):

        # Stop if enough samples collected
        if sample_size is not None and len(questions) >= sample_size:
            break

        # Skip broken rows with explanation
        try:
            question_text = row.get("question")
            a = row.get("choice_a")
            b = row.get("choice_b")
            c = row.get("choice_c")
            d = row.get("choice_d")
            ans = row.get("answer")

            # If ANY mandatory field missing → skip
            if not question_text or not a or not b or not c or not d or not ans:
                print(f"[SKIP] Missing essential field at row {i}")
                continue

            q = {
                "id": str(row.get("idx", i)),
                "prompt": row.get("prompt", ""),
                "question": question_text,
                "choices": [a, b, c, d],
                "answer": ans.strip().upper(),
                "gold_passage": row.get("gold_passage", ""),
                "gold_idx": row.get("gold_idx", ""),
            }

            questions.append(q)

        except Exception as e:
            print(f"[ERROR] Failed to parse row {i}: {e}")
            continue

    print(f"[LOCAL] Returning {len(questions)} formatted questions")
    return questions


if __name__ == "__main__":
    sample = load_bar_exam_qa_local(sample_size=3, split="train")
    print(sample[0] if sample else "EMPTY")
