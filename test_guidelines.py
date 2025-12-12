"""
Script to explore OAB-Bench guidelines dataset structure.

This script loads a small sample from the 'guidelines' subset to help
understand the structure of reference answers and scoring rubrics.
"""

from datasets import load_dataset

def explore_guidelines(num_samples=3):
    """
    Load and display samples from OAB-Bench guidelines subset.

    Args:
        num_samples: Number of samples to display
    """
    print("=" * 80)
    print("EXPLORING OAB-BENCH GUIDELINES DATASET")
    print("=" * 80)

    # Load guidelines subset
    print("\nLoading guidelines subset...")
    dataset = load_dataset("maritaca-ai/oab-bench", "guidelines", split="train")
    print(f"Total items in guidelines: {len(dataset)}")

    # Filter out procedural pieces (optional - for cleaner view)
    filtered_items = [
        item for item in dataset
        if 'peca_praticoprofissional' not in item['question_id']
    ]
    print(f"Dissertative questions only: {len(filtered_items)}")

    print("\n" + "=" * 80)
    print(f"DISPLAYING {num_samples} SAMPLE GUIDELINES")
    print("=" * 80)

    for idx, item in enumerate(filtered_items[:num_samples], 1):
        print(f"\n{'='*80}")
        print(f"SAMPLE {idx}")
        print(f"{'='*80}")

        print(f"\n[QUESTION ID]: {item['question_id']}")
        print(f"[ANSWER ID]: {item.get('answer_id', 'N/A')}")
        print(f"[MODEL ID]: {item.get('model_id', 'N/A')}")

        # Display top-level fields
        print(f"\n[TOP-LEVEL FIELDS]:")
        for key in item.keys():
            if key not in ['turns', 'choices']:
                value = item[key]
                if isinstance(value, str) and len(value) > 100:
                    print(f"  - {key}: {value[:100]}... (truncated)")
                else:
                    print(f"  - {key}: {value}")

        # Display turns (contains reference answers)
        print(f"\n[TURNS] (contains reference answers):")
        turns = item.get('turns', [])
        print(f"  Number of turns: {len(turns)}")

        for turn_idx, turn in enumerate(turns):
            print(f"\n  --- TURN {turn_idx} ---")
            if isinstance(turn, str):
                # Display first 500 chars to see reference answer + rubric start
                print(f"  {turn[:800]}")
                if len(turn) > 800:
                    print(f"  ... (total length: {len(turn)} chars)")
            else:
                print(f"  Type: {type(turn)}")
                print(f"  Content: {turn}")

        # Display choices (contains scoring info)
        print(f"\n[CHOICES] (contains scoring metadata):")
        choices = item.get('choices', [])
        print(f"  Number of choices: {len(choices)}")

        for choice_idx, choice in enumerate(choices[:2]):  # Show first 2
            print(f"\n  --- CHOICE {choice_idx} ---")
            if isinstance(choice, dict):
                for key, value in choice.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"    {key}: {value[:100]}... (truncated)")
                    else:
                        print(f"    {key}: {value}")
            else:
                print(f"  Type: {type(choice)}, Value: {choice}")

        print("\n" + "-" * 80)

    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)
    print("\nKey observations:")
    print("1. Check if 'turns' contains reference answers + rubrics")
    print("2. Verify question_id format matches the 'questions' subset")
    print("3. Note the structure of scoring rubrics (if present)")
    print("4. Confirm we can parse and extract reference answers")


if __name__ == "__main__":
    # Explore 3 samples
    explore_guidelines(num_samples=3)

    print("\n\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Review the output above")
    print("2. Confirm the structure is clear")
    print("3. Ready to implement load_oab_guidelines() function")
    print("=" * 80)
