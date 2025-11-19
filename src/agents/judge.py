import os
from groq import Groq

# Groq API Configuration (We inherit settings from the main script environment)
# For simplicity, we assume API_KEY, MODEL_NAME are available via environment or config.

class JudgeAgent:
    """
    JudgeAgent is responsible for evaluating the legal arguments presented by the
    Debater agents and providing the final ruling based on their quality and adherence
    to legal principles (IRAC).
    """
    def __init__(self, model_name: str, max_tokens: int = 500, temperature: float = 0.0):
        # Initialize Groq client
        self.client = Groq() 
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_instruction = self._get_system_instruction()

    def _get_system_instruction(self):
        """Sets the judicial persona and core instructions for the Judge."""
        return (
            "You are an impartial, highly respected Supreme Court Judge. Your role is to "
            "evaluate two conflicting legal arguments (Argument A and Argument B) regarding a single legal question. "
            "Your ruling must be based *solely* on the logical soundness, legal accuracy, "
            "and adherence to the IRAC framework in the presented arguments. Do not introduce new facts. "
            "You must select the better argument (A or B) and provide a concise, reasoned justification for your ruling."
        )

    def generate_ruling(self, question: dict, arguments: dict) -> str:
        """
        Generates the final ruling and justification based on the two debaters' arguments.
        
        Args:
            question (dict): The legal question details.
            arguments (dict): Contains 'A' and 'B' debater arguments.
        
        Returns:
            str: The final ruling text and chosen option.
        """
        # 1. Construct the user prompt
        prompt = self._construct_prompt(question, arguments)
        
        # 2. Call the Groq API
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_instruction},
                    {"role": "user", "content": prompt},
                ],
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return chat_completion.choices[0].message.content
        
        except Exception as e:
            return f"JUDGE API Error: {e}"

    def _construct_prompt(self, question, arguments):
        """Constructs the prompt detailing the problem and conflicting arguments."""
        
        # Format the question and choices
        problem_statement = f"**Legal Problem**: {question['question']}\n"
        choices = (f"(A) {question['choice_a']}\n(B) {question['choice_b']}\n"
                   f"(C) {question['choice_c']}\n(D) {question['choice_d']}")

        # Format the conflicting arguments
        argument_text = (
            f"\n\n--- Conflicting Arguments ---\n\n"
            f"**Argument A (Supporting Option A/C)**:\n{arguments['A']}\n\n"
            f"**Argument B (Supporting Option B/D)**:\n{arguments['B']}"
        )

        # Final instruction to the Judge
        return (
            f"Review the following legal problem, choices, and two conflicting legal arguments. "
            f"Provide your ruling and justification based on legal merit, not personal opinion.\n\n"
            f"{problem_statement}\n{choices}\n{argument_text}\n\n"
            f"RULING:\n"
            f"1. Justification: [Explain which argument is legally superior and why.]\n"
            f"2. Final Decision: [State the final choice from (A, B, C, D) based on your ruling.]"
        )