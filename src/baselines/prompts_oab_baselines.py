"""
Baseline prompts for OAB open-ended questions.
INTENTIONALLY SIMPLE - baselines should not outperform MAD.
"""


def get_single_agent_prompt_oab(question: str, category: str) -> str:
    """
    Single-agent baseline - MINIMAL prompt.

    Args:
        question: The open-ended legal question
        category: Law category (e.g., "Direito Penal")

    Returns:
        Minimal prompt string
    """
    prompt = f"""Você é um especialista em {category}.

Questão:
{question}

Responda esta questão da OAB de forma dissertativa.

Responda em JSON:
{{
  "answer": "Sua resposta dissertativa",
  "key_citations": ["Citações usadas"]
}}"""

    return prompt


def get_cot_prompt_oab(question: str, category: str) -> str:
    """
    Chain-of-Thought baseline - adds "think step by step".

    Args:
        question: The open-ended legal question
        category: Law category (e.g., "Direito Penal")

    Returns:
        CoT prompt string
    """
    prompt = f"""Você é um especialista em {category}.

Questão:
{question}

Pense passo a passo antes de responder esta questão da OAB.

Responda em JSON:
{{
  "reasoning": "Seu raciocínio passo a passo",
  "answer": "Sua resposta dissertativa final",
  "key_citations": ["Citações usadas"]
}}"""

    return prompt


def get_self_consistency_prompt_oab(question: str, category: str) -> str:
    """
    Self-Consistency baseline - same as CoT (called multiple times).

    Args:
        question: The open-ended legal question
        category: Law category (e.g., "Direito Penal")

    Returns:
        SC prompt string (identical to CoT)
    """
    # Self-consistency uses same prompt as CoT, but generates N times
    return get_cot_prompt_oab(question, category)
