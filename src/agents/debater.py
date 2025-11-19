import os
from groq import Groq

# Groq API Configuration

class DebaterAgent:
    """
    DebaterAgent is responsible for constructing a strong legal argument (based on IRAC)
    to support a specific set of choices (e.g., A/C or B/D) related to a legal question.
    """
    def __init__(self, model_name: str, stance: str, max_tokens: int = 500, temperature: float = 0.7):
        # Stance must be 'A' or 'B', representing Argument A or B in the debate
        if stance not in ['A', 'B']:
            raise ValueError("Stance must be 'A' or 'B'.")
            
        self.client = Groq()
        self.model_name = model_name
        self.stance = stance
        self.max_tokens = max_tokens
        # Higher temperature (0.7) is used to encourage more diverse and creative arguments
        self.temperature = temperature
        self.system_instruction = self._get_system_instruction()

    def _get_system_instruction(self):
        """Sets the persona and core instructions for the Debater."""
        return (
            "You are a highly skilled litigation attorney specializing in legal analysis. "
            "Your sole mission is to build the most persuasive and legally accurate argument "
            "to support your assigned position. You must strictly follow the IRAC (Issue, Rule, Application, Conclusion) "
            "framework to structure your argument. State your conclusion clearly."
        )

    def generate_argument(self, question: dict, supporting_choices: tuple) -> str:
        """
        Generates a strong legal argument supporting the assigned choices.
        
        Args:
            question (dict): The legal question details.
            supporting_choices (tuple): The choices this debater must support (e.g., ('A', 'C') or ('B', 'D')).
            
        Returns:
            str: The IRAC-structured argument text.
        """
        # 1. Construct the user prompt
        prompt = self._construct_prompt(question, supporting_choices)
        
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
            return f"DEBATER API Error: {e}"

    def _construct_prompt(self, question, supporting_choices):
        """Constructs the prompt detailing the problem and the required stance."""
        
        # Format the question and choices
        question_text = f"**Question**: {question['question']}\n"
        choices = (f"(A) {question['choice_a']}\n(B) {question['choice_b']}\n"
                   f"(C) {question['choice_c']}\n(D) {question['choice_d']}")
        
        # Identify the opposing choices for context
        all_choices = ('A', 'B', 'C', 'D')
        opposing_choices = tuple(c for c in all_choices if c not in supporting_choices)

        # Final instruction to the Debater
        return (
            f"Your assigned task is to construct the most compelling legal argument that supports one of the following choices: **{', '.join(supporting_choices)}**. "
            f"You MUST use the IRAC framework.\n\n"
            f"--- Problem ---\n{question_text}"
            f"\n--- Choices ---\n{choices}"
            f"\n\n--- Your Argument (IRAC) ---\n"
            f"1. Issue: [Identify the core legal issue that leads to your conclusion.]\n"
            f"2. Rule: [State the applicable laws and principles that support your choices.]\n"
            f"3. Application: [Apply the rules to the facts to build your argument.]\n"
            f"4. Conclusion: [State your final conclusion supporting one of {', '.join(supporting_choices)}.]"
        )