# src/agents/cot_basic_prompt.py

def create_basic_cot_prompt(question):
    """
    Creates a standard Chain-of-Thought prompt without enforcing the IRAC structure.
    """
    
    question_text = f"**Question**: {question['question']}\n"
    
    # FIX: Use the 'choices' list from the question dictionary instead of individual 'choice_a', 'choice_b', etc.
    choices_list = question['choices']
    choices = (
        f"(A) {choices_list[0]}\n"
        f"(B) {choices_list[1]}\n"
        f"(C) {choices_list[2]}\n"
        f"(D) {choices_list[3]}"
    )
    
    # System Instruction: Simple CoT instruction
    system_instruction = (
        "You are a legal expert. Analyze the legal problem and provide a detailed, "
        "step-by-step reasoning process to reach the correct answer. "
        "The final conclusion (Final Answer) must be one of the given choices (A, B, C, D)."
    )
    
    # User prompt template enforcing the CoT structure
    prompt = (
        f"Analyze the legal question and choices, then derive the correct answer through step-by-step reasoning:\n\n"
        f"--- Problem ---\n{question_text}"
        f"\n--- Choices ---\n{choices}"
        f"\n----------------\n\n"
        f"ASSISTANT (Chain-of-Thought):\n"
        f"[Provide a detailed, step-by-step analysis here. State the relevant legal rules and apply them to the facts.]\n\n"
        f"Final Answer: [Write only the correct choice (A/B/C/D) here.]"
    )
    
    return system_instruction, prompt