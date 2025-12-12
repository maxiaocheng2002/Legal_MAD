"""
Debater agent for MAD system.
"""

from typing import Dict, Optional
from src.utils.api_client_experimental import GroqClient, OpenRouterClient
from src.agents.prompts_experimental import get_debater_opening_prompt, get_debater_rebuttal_prompt


class Debater:
    """Debater agent that argues for a specific position."""

    def __init__(self, client, name: str = "Debater"):
        """
        Initialize debater.

        Args:
            client: API client (GroqClient or OpenRouterClient)
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
        position: str = None
    ) -> Dict:
        """
        Generate opening argument.

        Args:
            question: Legal question
            prompt_context: Question context/prompt
            choices: List of answer choices
            position: Position to defend (A, B, C, or D). If None, debater chooses freely.

        Returns:
            Dictionary with position, argument, and citations
        """
        prompt = get_debater_opening_prompt(
            question=question,
            prompt_context=prompt_context,
            choices=choices,
            position=position
        )

        response = self.client.generate_json(prompt, max_tokens=1200)

        # Validate response
        if 'position' not in response or 'argument' not in response:
            raise ValueError(f"Invalid debater response: {response}")

        # Store the chosen/assigned position
        self.position = response['position']
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

        response = self.client.generate_json(prompt, max_tokens=1000)

        # Validate response
        if 'rebuttal' not in response:
            raise ValueError(f"Invalid rebuttal response: {response}")

        return response

    # ==================== IRAC METHODS ====================

    def generate_opening_irac(
        self,
        question: str,
        prompt_context: str,
        choices: list,
        position: str = None
    ) -> Dict:
        """
        Generate IRAC-structured opening argument.

        Args:
            question: Legal question
            prompt_context: Question context/prompt
            choices: List of answer choices
            position: Position to defend (A, B, C, or D). If None, debater chooses freely.

        Returns:
            Dictionary with position, irac structure, and citations
        """
        from src.agents.prompts_experimental import get_debater_opening_prompt_irac

        prompt = get_debater_opening_prompt_irac(
            question=question,
            prompt_context=prompt_context,
            choices=choices,
            position=position
        )

        response = self.client.generate_json(prompt, max_tokens=1200)

        # Validate response
        if 'position' not in response or 'irac' not in response:
            raise ValueError(f"Invalid IRAC debater response: {response}")

        # Validate IRAC structure
        irac = response.get('irac', {})
        required_keys = ['issue', 'rule', 'application', 'conclusion']
        for key in required_keys:
            if key not in irac:
                raise ValueError(f"Missing IRAC component '{key}': {response}")

        # Store the chosen/assigned position
        self.position = response['position']
        self.opening_argument = response
        return response

    def generate_rebuttal_irac(
        self,
        question: str,
        prompt_context: str,
        opponent_opening: Dict
    ) -> Dict:
        """
        Generate IRAC-structured rebuttal argument.

        Args:
            question: Legal question
            prompt_context: Question context/prompt
            opponent_opening: Opponent's IRAC opening argument

        Returns:
            Dictionary with structured rebuttal and citations
        """
        if not self.opening_argument:
            raise ValueError("Must generate opening argument before rebuttal")

        from src.agents.prompts_experimental import get_debater_rebuttal_prompt_irac

        prompt = get_debater_rebuttal_prompt_irac(
            question=question,
            prompt_context=prompt_context,
            my_position=self.position,
            my_opening=self.opening_argument,
            opponent_opening=opponent_opening
        )

        response = self.client.generate_json(prompt, max_tokens=1000)

        # Validate response
        if 'rebuttal' not in response:
            raise ValueError(f"Invalid IRAC rebuttal response: {response}")

        # Validate rebuttal structure
        rebuttal = response.get('rebuttal', {})
        expected_keys = ['issue_critique', 'rule_critique', 'application_critique', 'my_reinforcement']
        for key in expected_keys:
            if key not in rebuttal:
                raise ValueError(f"Missing rebuttal component '{key}': {response}")

        return response

    # ==================== OAB OPEN-ENDED METHODS ====================

    def generate_opening_oab(
        self,
        question: str,
        category: str,
        is_adversarial: bool = False
    ) -> Dict:
        """
        Generate opening argument for OAB open-ended question (NEUTRAL).

        Args:
            question: Open-ended legal question
            category: Law category (e.g., "Direito Penal")
            is_adversarial: Legacy parameter (ignored)

        Returns:
            Dictionary with irac structure, full_answer, and citations
        """
        from src.agents.prompts_oab import get_debater_opening_prompt_oab

        prompt = get_debater_opening_prompt_oab(
            question=question,
            category=category,
            is_adversarial=False  # Always neutral in this method
        )

        response = self.client.generate_json(prompt, max_tokens=2000)

        # Validate response
        if 'irac' not in response or 'full_answer' not in response:
            raise ValueError(f"Invalid OAB opening response: {response}")

        # Store opening argument
        self.opening_argument = response
        return response

    def generate_opening_oab_adversarial(
        self,
        question: str,
        category: str,
        opponent_opening: Dict
    ) -> Dict:
        """
        Generate ADVERSARIAL opening argument - sees opponent's position first.

        Args:
            question: Open-ended legal question
            category: Law category
            opponent_opening: Opponent's opening argument

        Returns:
            Dictionary with irac structure, full_answer, and citations
        """
        from src.agents.prompts_oab import get_debater_opening_prompt_oab_adversarial

        prompt = get_debater_opening_prompt_oab_adversarial(
            question=question,
            category=category,
            opponent_opening=opponent_opening
        )

        response = self.client.generate_json(prompt, max_tokens=2000)

        # Validate response
        if 'irac' not in response or 'full_answer' not in response:
            raise ValueError(f"Invalid OAB adversarial opening response: {response}")

        # Store opening argument
        self.opening_argument = response
        return response

    def generate_rebuttal_oab(
        self,
        question: str,
        category: str,
        opponent_opening: Dict
    ) -> Dict:
        """
        Generate rebuttal for OAB debate.

        Args:
            question: Open-ended legal question
            category: Law category
            opponent_opening: Opponent's opening argument

        Returns:
            Dictionary with critique, refined_answer, and citations
        """
        if not self.opening_argument:
            raise ValueError("Must generate opening argument before rebuttal")

        from src.agents.prompts_oab import get_debater_rebuttal_prompt_oab

        prompt = get_debater_rebuttal_prompt_oab(
            question=question,
            category=category,
            my_opening=self.opening_argument,
            opponent_opening=opponent_opening
        )

        response = self.client.generate_json(prompt, max_tokens=2000)

        # Validate response
        if 'refined_answer' not in response:
            raise ValueError(f"Invalid OAB rebuttal response: {response}")

        return response

    # ==================== OAB VANILLA METHODS (NO IRAC) ====================

    def generate_opening_oab_vanilla(
        self,
        question: str,
        category: str
    ) -> Dict:
        """
        Generate simple opening argument for OAB (NEUTRAL, no IRAC structure).

        Args:
            question: Open-ended legal question
            category: Law category (e.g., "Direito Penal")

        Returns:
            Dictionary with answer and citations
        """
        from src.agents.prompts_oab_vanilla import get_debater_opening_prompt_oab_vanilla

        prompt = get_debater_opening_prompt_oab_vanilla(
            question=question,
            category=category
        )

        response = self.client.generate_json(prompt, max_tokens=2000)

        # Validate response
        if 'answer' not in response:
            raise ValueError(f"Invalid OAB vanilla opening response: {response}")

        # Store opening argument
        self.opening_argument = response
        return response

    def generate_opening_oab_adversarial_vanilla(
        self,
        question: str,
        category: str,
        opponent_opening: Dict
    ) -> Dict:
        """
        Generate ADVERSARIAL opening argument - sees opponent's position first (no IRAC).

        Args:
            question: Open-ended legal question
            category: Law category
            opponent_opening: Opponent's opening argument

        Returns:
            Dictionary with answer and citations
        """
        from src.agents.prompts_oab_vanilla import get_debater_opening_prompt_oab_adversarial_vanilla

        prompt = get_debater_opening_prompt_oab_adversarial_vanilla(
            question=question,
            category=category,
            opponent_opening=opponent_opening
        )

        response = self.client.generate_json(prompt, max_tokens=2000)

        # Validate response
        if 'answer' not in response:
            raise ValueError(f"Invalid OAB vanilla adversarial opening response: {response}")

        # Store opening argument
        self.opening_argument = response
        return response

    def generate_rebuttal_oab_vanilla(
        self,
        question: str,
        category: str,
        opponent_opening: Dict
    ) -> Dict:
        """
        Generate rebuttal for OAB debate (no IRAC structure).

        Args:
            question: Open-ended legal question
            category: Law category
            opponent_opening: Opponent's opening argument

        Returns:
            Dictionary with critique, refined_answer, and citations
        """
        if not self.opening_argument:
            raise ValueError("Must generate opening argument before rebuttal")

        from src.agents.prompts_oab_vanilla import get_debater_rebuttal_prompt_oab_vanilla

        prompt = get_debater_rebuttal_prompt_oab_vanilla(
            question=question,
            category=category,
            my_opening=self.opening_argument,
            opponent_opening=opponent_opening
        )

        response = self.client.generate_json(prompt, max_tokens=2000)

        # Validate response
        if 'refined_answer' not in response:
            raise ValueError(f"Invalid OAB vanilla rebuttal response: {response}")

        return response
