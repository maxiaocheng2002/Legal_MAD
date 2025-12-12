def create_basic_cot_prompt(question):
    choices = question["choices"]
    choices_str = (
        f"(A) {choices[0]}\n"
        f"(B) {choices[1]}\n"
        f"(C) {choices[2]}\n"
        f"(D) {choices[3]}"
    )

    # minimal instruction
    system_instruction = (
        "Return only: Final Answer: X (A/B/C/D)."
    )
    prompt = (
        f"{question['question']}\n\n"
        f"{choices_str}\n\n"
        "Final Answer: "
    )

    return system_instruction, prompt
