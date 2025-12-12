def create_cot_prompt(question):
    choices = question["choices"]
    choices_str = (
        f"(A) {choices[0]}\n"
        f"(B) {choices[1]}\n"
        f"(C) {choices[2]}\n"
        f"(D) {choices[3]}"
    )

    # IRAC을 "쓸 수 있다" 정도로만 언급 (강제 X)
    system_instruction = (
        "Use IRAC internally (Issue, Rule, Application, Conclusion) to decide the answer. "
        "Do NOT write the IRAC steps. "
        "Return only the final answer in the format 'Final Answer: X' "
        "where X is A, B, C, or D."
    )


    prompt = (
        f"{question['question']}\n\n"
        f"{choices_str}\n\n"
        "Final Answer: "
    )

    return system_instruction, prompt
