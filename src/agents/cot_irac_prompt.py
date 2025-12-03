def create_cot_prompt(question):
    """Creates the Chain-of-Thought prompt text based on the IRAC framework."""
    
    question_text = f"**Question**: {question['question']}\n"
    
    # FIX: Use the 'choices' list from the question dictionary instead of individual 'choice_a', 'choice_b', etc.
    choices_list = question['choices']
    choices = (
        f"(A) {choices_list[0]}\n"
        f"(B) {choices_list[1]}\n"
        f"(C) {choices_list[2]}\n"
        f"(D) {choices_list[3]}"
    )
    
    # System Instruction (The core CoT/IRAC instruction)
    system_instruction = (
        "You are a legal expert with the highest logic and accuracy. Analyze the legal problem "
        "and strictly follow the IRAC (Issue, Rule, Application, Conclusion) methodology for step-by-step reasoning. "
        "The final conclusion (Final Answer) must be one of the given choices (A, B, C, D)."
    )
    
    # User prompt template enforcing the IRAC structure
    prompt = (
        f"Analyze the legal question and choices, then derive the correct answer through the IRAC reasoning process:\n\n"
        f"--- Problem ---\n{question_text}"
        f"\n--- Choices ---\n{choices}"
        f"\n----------------\n\n"
        f"ASSISTANT (Chain-of-Thought):\n"
        f"1. Issue: [Identify the core legal issue here.]\n"
        f"2. Rule: [State the applicable laws, legal principles, and precedents here.]\n"
        f"3. Application: [Apply the rule to the facts of the problem with logical analysis.]\n"
        f"4. Conclusion: [State the final conclusion here.]\n\n"
        f"Final Answer: [Write only the correct choice (A/B/C/D) here.]"
    )
    
    return system_instruction, prompt