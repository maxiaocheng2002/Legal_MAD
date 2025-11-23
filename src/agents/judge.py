"""
Judge agent for MAD system.
"""

from typing import Dict
from src.utils.api_client import GroqClient
from src.agents.prompts import get_judge_decision_prompt


class Judge:
    """Judge agent that synthesizes debate and makes final decision."""

    def __init__(self, client: GroqClient):
        """
        Initialize judge.

        Args:
            client: Groq API client
        """
        self.client = client

    def make_decision(
        self,
        question: str,
        prompt_context: str,
        choices: list,
        debate_history: Dict
    ) -> Dict:
        """
        Make final decision based on debate.

        Args:
            question: Legal question
            prompt_context: Question context/prompt
            choices: List of answer choices
            debate_history: Full debate history with structure:
                {
                    'debater_x': {'opening': {...}, 'rebuttal': {...}},
                    'debater_y': {'opening': {...}, 'rebuttal': {...}}
                }

        Returns:
            Dictionary with decision and rationale
        """
        prompt = get_judge_decision_prompt(
            question=question,
            prompt_context=prompt_context,
            choices=choices,
            debate_history=debate_history
        )

        response = self.client.generate_json(prompt, max_tokens=400)

        # Validate response
        if 'decision' not in response or 'rationale' not in response:
            raise ValueError(f"Invalid judge response: {response}")

        # Ensure decision is valid choice
        if response['decision'] not in ['A', 'B', 'C', 'D']:
            raise ValueError(f"Invalid decision choice: {response['decision']}")

        return response
