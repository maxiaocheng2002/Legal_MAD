"""
Judge agent for MAD system.
"""

from typing import Dict
from src.utils.api_client import GroqClient
from src.agents.prompts import get_judge_decision_prompt


class Judge:
    """Judge agent that synthesizes debate and makes final decision using IRAC analysis."""

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
        Make final decision based on debate using IRAC analysis.

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
            Dictionary with decision, and IRAC analysis
        """
        prompt = get_judge_decision_prompt(
            question=question,
            prompt_context=prompt_context,
            choices=choices,
            debate_history=debate_history
        )

        # Reduced token limit due to structured format
        response = self.client.generate_json(prompt, max_tokens=300)

        # Validate response
        if 'decision' not in response:
            raise ValueError(f"Missing decision in judge response: {response}")

        # Require irac_analysis (core IRAC structure)
        if 'irac_analysis' not in response:
            raise ValueError(f"Missing irac_analysis in judge response: {response}")

        # Validate irac_analysis structure
        irac = response.get('irac_analysis', {})
        required_irac_keys = ['best_issue', 'best_rule', 'best_application', 'best_conclusion']
        missing_keys = [key for key in required_irac_keys if key not in irac or not str(irac.get(key, '')).strip()]
        if missing_keys:
            raise ValueError(f"Incomplete irac_analysis structure. Missing or empty keys: {missing_keys}. Response: {response}")

        # Synthesize rationale from irac_analysis if missing (for output completeness)
        if 'rationale' not in response or not response.get('rationale', '').strip():
            irac = response['irac_analysis']
            response['rationale'] = (
                f"Decision {response['decision']} is correct because "
                f"{irac.get('best_rule', 'the applicable legal rule')} "
                f"supports this conclusion. "
                f"{str(irac.get('best_application', 'The application to the facts'))[:200]}"
            )

        # Ensure decision is valid choice
        if response['decision'] not in ['A', 'B', 'C', 'D']:
            raise ValueError(f"Invalid decision choice: {response['decision']}")

        return response
