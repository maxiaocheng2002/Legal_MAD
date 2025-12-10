"""
SC1: Self-Consistency Baseline for Legal QA
Runs the same style as SingleAgentBaseline, but samples the model N times
and uses majority vote on the answers.
"""

import re
import random
from typing import Dict, List, Optional
from collections import Counter

from src.utils.api_client import GroqClient


class SelfConsistencyBaseline:
    """Self-consistency baseline for legal QA."""

    def __init__(
        self,
        client: Optional[GroqClient] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        num_samples: int = 10
    ):
        """
        Args:
            client: GroqClient instance (if None → create new one)
            api_key: Groq API key
            model: LLM model name
            num_samples: number of LLM samples to collect for voting
        """
        self.num_samples = num_samples

        if client is not None:
            self.client = client
        else:
            self.client = GroqClient(
                api_key=api_key,
                model=model or "llama-3.1-8b-instant"
            )

    # -------------------------------------------------------
    # Utility: Extract A/B/C/D from text
    # -------------------------------------------------------
    @staticmethod
    def extract_answer_letter(text: str) -> Optional[str]:
        if not text:
            return None

        text = text.strip().upper()

        # Strict match first
        m = re.search(r"\b([ABCD])\b", text)
        if m:
            return m.group(1)

        # Loose match fallback
        for letter in ["A", "B", "C", "D"]:
            if letter in text:
                return letter

        return None

    # -------------------------------------------------------
    # MULTI-SAMPLE SELF-CONSISTENCY MCQ ANSWER
    # -------------------------------------------------------
    def answer_mcq(
        self,
        question: str,
        prompt_context: str,
        choices: List[str]
    ) -> Dict:
        """
        Self-consistency multiple-sample MCQ answering.

        Returns:
            {
              "answer": "C",
              "samples": [...],
              "reasoning_samples": [...],
              "majority_count": 7
            }
        """
        choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])

        # Build full question text
        if prompt_context:
            full_question = f"{prompt_context}\n\n{question}"
        else:
            full_question = question

        # Prompt used for each sample
        base_prompt = f"""
You are a legal expert. Answer the following legal multiple-choice question.

{full_question}

Answer choices:
{choices_text}

Respond in JSON format:
{{
  "answer": "A",
  "reasoning": "legal reasoning..."
}}

IMPORTANT:
- "answer" must be ONE letter: A, B, C, or D.
"""

        answers = []
        reasoning = []

        # SAMPLE N TIMES
        for _ in range(self.num_samples):
            resp = self.client.generate_json(base_prompt, max_tokens=400)

            raw_ans = str(resp.get("answer", "")).strip().upper()
            clean_ans = self.extract_answer_letter(raw_ans)

            if clean_ans:
                answers.append(clean_ans)
            else:
                answers.append(random.choice(["A", "B", "C", "D"]))

            reasoning.append(resp.get("reasoning", ""))

        # Majority vote
        counter = Counter(answers)
        final_answer, final_count = counter.most_common(1)[0]

        return {
            "answer": final_answer,
            "samples": answers,
            "reasoning_samples": reasoning,
            "majority_count": final_count
        }