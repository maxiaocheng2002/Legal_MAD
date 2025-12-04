"""
SC1: Self-Consistency Baseline for Legal QA
Non-freezing, full reasoning, majority-vote implementation.
"""

import re
import time
import random
from typing import Dict, List, Optional
from collections import Counter

from src.utils.api_client import GroqClient


class SelfConsistencyBaseline:
    """Self-consistency baseline for legal MCQ QA."""

    def __init__(
        self,
        client: Optional[GroqClient] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        num_samples: int = 10,
        timeout: int = 20,        # prevents freezing
    ):
        """
        num_samples = number of LLM calls per question (self-consistency)
        timeout = max wait time per call
        """
        self.num_samples = num_samples
        self.timeout = timeout

        if client is not None:
            self.client = client
        else:
            # IMPORTANT: must be a REAL Groq chat model (NOT prompt-guard)
            self.client = GroqClient(
                api_key=api_key,
                model=model or "openai/gpt-oss-120b",
            )

    # -------------------------------------------------------
    # Extract A/B/C/D
    # -------------------------------------------------------
    @staticmethod
    def extract_answer_letter(text: str) -> Optional[str]:
        if not text:
            return None
        text = text.strip().upper()

        m = re.search(r"\b([ABCD])\b", text)
        if m:
            return m.group(1)

        for letter in ["A", "B", "C", "D"]:
            if letter in text:
                return letter
        return None

    # -------------------------------------------------------
    # Safe model call (no freezing)
    # -------------------------------------------------------
    def _safe_generate_json(self, prompt: str):
        """Guaranteed to return something — prevents infinite blocking."""
        start = time.time()

        while True:
            try:
                return self.client.generate_json(
                    prompt,
                    max_tokens=400,
                )
            except Exception:
                if time.time() - start > self.timeout:
                    print("⚠️  LLM timeout → Returning fallback guess.")
                    return {"answer": random.choice(["A", "B", "C", "D"]),
                            "reasoning": "Timeout fallback."}
                time.sleep(1)

    # -------------------------------------------------------
    # Self-consistency MCQ answer
    # -------------------------------------------------------
    def answer_mcq(
        self,
        question: str,
        prompt_context: str,
        choices: List[str],
    ) -> Dict:
        """
        Returns: final answer, samples, reasoning_samples, majority_count
        """

        # Build choices
        choices_text = "\n".join(
            f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)
        )

        # Build full question
        if prompt_context:
            full_question = f"{prompt_context}\n\n{question}"
        else:
            full_question = question

        # Base prompt
        base_prompt = f"""
You are a legal expert. This is a bar exam style multiple-choice question.
Think step-by-step, evaluate each option, and then pick the BEST answer.

{full_question}

Answer choices:
{choices_text}

Respond ONLY in JSON format:
{{
  "answer": "A",
  "reasoning": "Explain your full legal reasoning step by step."
}}

IMPORTANT:
- "answer" must be ONE letter A–D.
- "reasoning" MUST explain why the correct choice is best.
"""

        answers = []
        reasoning_samples = []

        # ---------------------------------------------------
        # Collect NUM_SAMPLES samples
        # ---------------------------------------------------
        for _ in range(self.num_samples):

            varied_prompt = (
                base_prompt
                + f"\n\nSelf-consistency sample ID: {random.randint(1, 10_000_000)}."
                  " Try a different line of reasoning from other samples."
            )

            resp = self._safe_generate_json(varied_prompt)

            raw_ans = str(resp.get("answer", "")).strip().upper()
            clean_ans = self.extract_answer_letter(raw_ans)

            if clean_ans:
                answers.append(clean_ans)
            else:
                answers.append(random.choice(["A", "B", "C", "D"]))

            reasoning_samples.append(resp.get("reasoning", ""))

        # Majority vote
        counter = Counter(answers)
        final_answer, count = counter.most_common(1)[0]

        return {
            "answer": final_answer,
            "samples": answers,
            "reasoning_samples": reasoning_samples,
            "majority_count": count,
        }
