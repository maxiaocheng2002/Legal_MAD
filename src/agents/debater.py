"""
Debater agent for MAD system.
"""

from typing import Dict, Optional
from src.utils.api_client import GroqClient
from src.agents.prompts import get_debater_opening_prompt, get_debater_rebuttal_prompt


class Debater:
    """Debater agent that argues for a specific position."""

    def __init__(self, client: GroqClient, name: str = "Debater"):
        """
        Initialize debater.

        Args:
            client: Groq API client
            name: Debater name for logging
        """
        self.client = client
        self.name = name
        self.position = None
        self.opening_argument = None

    def generate_opening(
        self,
        question: str,
        prompt_context: str,
        choices: list,
        position: str
    ) -> Dict:
        """
        Generate opening argument.

        Args:
            question: Legal question
            prompt_context: Question context/prompt
            choices: List of answer choices
            position: Position to defend (A, B, C, or D)

        Returns:
            Dictionary with position, argument, and citations
        """
        self.position = position

        prompt = get_debater_opening_prompt(
            question=question,
            prompt_context=prompt_context,
            choices=choices,
            position=position
        )

        response = self.client.generate_json(prompt, max_tokens=500)

        # Validate response
        if 'position' not in response or 'argument' not in response:
            raise ValueError(f"Invalid debater response: {response}")

        self.opening_argument = response
        return response

    def generate_rebuttal(
        self,
        question: str,
        prompt_context: str,
        opponent_opening: Dict
    ) -> Dict:
        """
        Generate rebuttal argument.

        Args:
            question: Legal question
            prompt_context: Question context/prompt
            opponent_opening: Opponent's opening argument

        Returns:
            Dictionary with rebuttal, counterarguments, and citations
        """
        if not self.opening_argument:
            raise ValueError("Must generate opening argument before rebuttal")

        prompt = get_debater_rebuttal_prompt(
            question=question,
            prompt_context=prompt_context,
            my_position=self.position,
            my_opening=self.opening_argument,
            opponent_opening=opponent_opening
        )

        response = self.client.generate_json(prompt, max_tokens=400)

        # Validate response
        if 'rebuttal' not in response:
            raise ValueError(f"Invalid rebuttal response: {response}")

        return response
