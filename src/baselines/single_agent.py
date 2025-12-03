"""
B1: Single-Agent (Zero-shot) Baseline

Simple single-agent system that answers legal questions without debate.
Uses zero-shot prompting with structured JSON output.
"""

from typing import Dict, List, Optional
from src.utils.api_client import GroqClient


class SingleAgentBaseline:
    """Single-agent baseline for legal QA (zero-shot)."""

    def __init__(
        self,
        client: Optional[GroqClient] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize single-agent baseline.

        Args:
            client: GroqClient instance (if None, creates new one)
            api_key: API key (reads from env if None)
            model: Model name (default: llama-3.1-8b-instant)
        """
        if client is not None:
            self.client = client
        else:
            self.client = GroqClient(
                api_key=api_key,
                model=model or "llama-3.1-8b-instant"
            )

    def answer_mcq(
        self,
        question: str,
        prompt_context: str,
        choices: List[str]
    ) -> Dict:
        """
        Answer multiple-choice question (zero-shot).

        Args:
            question: Legal question
            prompt_context: Question context/prompt
            choices: List of answer choices

        Returns:
            Dictionary with answer, reasoning, and citations
        """
        choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])

        # Build full question
        if prompt_context:
            full_question = f"{prompt_context}\n\n{question}"
        else:
            full_question = question

        # Create prompt
        prompt = f"""You are a legal expert. Answer the following legal question by selecting the most legally correct answer choice.

{full_question}

Answer choices:
{choices_text}

Provide your answer with legal reasoning to support your choice.

Respond in JSON format:
{{
  "answer": "A",
  "reasoning": "Your legal reasoning here..."
}}

IMPORTANT: The "answer" field must be exactly one letter: A, B, C, or D (not multiple letters)."""

        response = self.client.generate_json(prompt, max_tokens=500)
        if 'answer' not in response:
            raise ValueError(f"Invalid response: missing 'answer' field: {response}")
        
        # Clean up answer (remove spaces, take first letter if multiple)
        answer = str(response['answer']).strip().upper()
        if ',' in answer:
            answer = answer.split(',')[0].strip()
        if len(answer) > 1:
            answer = answer[0]
        
        if answer not in ['A', 'B', 'C', 'D']:
            raise ValueError(f"Invalid answer choice: {response['answer']}")
        
        response['answer'] = answer

        return response

    def answer_open_ended(
        self,
        question: str,
        prompt_context: str = ""
    ) -> Dict:
        """
        Answer open-ended legal question (zero-shot).

        Args:
            question: Legal question
            prompt_context: Optional context

        Returns:
            Dictionary with answer
        """
        # Build full question
        if prompt_context:
            full_question = f"{prompt_context}\n\n{question}"
        else:
            full_question = question

        # Create prompt
        prompt = f"""You are a legal expert. Answer the following legal question with a comprehensive legal analysis.

{full_question}

Provide a clear and detailed legal answer.

Respond in JSON format:
{{
  "answer": "Your comprehensive legal answer here..."
}}"""

        # Generate response
        response = self.client.generate_json(prompt, max_tokens=800)

        # Validate response
        if 'answer' not in response:
            raise ValueError(f"Invalid response: missing 'answer' field: {response}")

        return response


if __name__ == "__main__":
    from src.utils.data_loader import load_bar_exam_qa

    print("Testing Single-Agent Baseline (B1) with Groq...")

    # Initialize baseline
    baseline = SingleAgentBaseline()

    questions = load_bar_exam_qa(sample_size=1)
    q = questions[0]

    # Test MCQ
    print("\n=== Testing MCQ ===")
    print(f"Question: {q['question'][:100]}...")
    print(f"Choices: {q['choices']}")
    print(f"Gold answer: {q['answer']}")

    result = baseline.answer_mcq(
        question=q['question'],
        prompt_context=q['prompt'],
        choices=q['choices']
    )

    print(f"\nModel answer: {result['answer']}")
    print(f"Correct: {result['answer'] == q['answer']}")
    print(f"Reasoning: {result['reasoning'][:200]}...")
    print(f"Citations: {result.get('citations', [])}")

    # Test open-ended
    print("\n=== Testing Open-Ended ===")
    open_question = "A company fired an employee without just cause. Analyze the legal implications under Brazilian labor law."

    result_open = baseline.answer_open_ended(
        question=open_question
    )

    print(f"Answer: {result_open['answer'][:200]}...")
    print(f"IRAC Issue: {result_open.get('irac', {}).get('issue', 'N/A')[:100]}...")
    print(f"Citations: {result_open.get('citations', [])}")

    print("\n Single-Agent Baseline (B1) working!")
