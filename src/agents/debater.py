"""
Debater agent for MAD system.
"""

from typing import Dict, Optional
from src.utils.api_client import GroqClient
from src.agents.prompts import get_debater_opening_prompt, get_debater_rebuttal_prompt


class Debater:
    """Debater agent that argues for a specific position using IRAC structure."""

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
        self.irac_history = []  # Track IRAC components for analysis

    def generate_opening(
        self,
        question: str,
        prompt_context: str,
        choices: list,
        position: str
    ) -> Dict:
        """
        Generate opening argument with IRAC structure.

        Args:
            question: Legal question
            prompt_context: Question context/prompt
            choices: List of answer choices
            position: Position to defend (A, B, C, or D)

        Returns:
            Dictionary with position, IRAC structure, and citations
        """
        self.position = position

        prompt = get_debater_opening_prompt(
            question=question,
            prompt_context=prompt_context,
            choices=choices,
            position=position
        )

        # Reduced token limit since we're using structured format
        response = self.client.generate_json(prompt, max_tokens=350)

        # Validate IRAC structure
        if 'position' not in response:
            raise ValueError(f"Missing position in debater response: {response}")
        
        if 'irac' not in response:
            raise ValueError(f"Missing IRAC structure in response: {response}")
        
        if not all(key in response['irac'] for key in ['issue', 'rule', 'application', 'conclusion']):
            raise ValueError(f"Incomplete IRAC structure: {response.get('irac', {})}")

        self.opening_argument = response
        return response

    def generate_rebuttal(
        self,
        question: str,
        prompt_context: str,
        opponent_opening: Dict
    ) -> Dict:
        """
        Generate rebuttal argument with IRAC structure and token-efficient format.

        Args:
            question: Legal question
            prompt_context: Question context/prompt
            opponent_opening: Opponent's opening argument

        Returns:
            Dictionary with rebuttal IRAC, counterarguments, and citations
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

        # Reduced token limit
        response = self.client.generate_json(prompt, max_tokens=300)

        # Validate IRAC structure
        if 'rebuttal_irac' not in response:
            raise ValueError(f"Missing rebuttal IRAC structure: {response}")
        
        if not all(key in response['rebuttal_irac'] for key in ['issue', 'rule', 'application', 'conclusion']):
            raise ValueError(f"Incomplete rebuttal IRAC structure: {response.get('rebuttal_irac', {})}")

        return response
