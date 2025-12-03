"""
Prompt templates for MAD legal reasoning.
"""


def get_debater_opening_prompt(question: str, prompt_context: str, choices: list, position: str) -> str:
    """
    Generate opening argument prompt for debater with IRAC structure.

    Args:
        question: The legal question
        prompt_context: Context/prompt from dataset
        choices: List of 4 answer choices
        position: Which choice to defend (A, B, C, or D)

    Returns:
        Formatted prompt string
    """
    choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])

    if prompt_context:
        full_question = f"{prompt_context}\n\n{question}"
    else:
        full_question = question

    prompt = f"""You are a legal expert. Argue that choice {position} is correct using IRAC structure.

Question:
{full_question}

Choices:
{choices_text}

Respond in JSON with IRAC structure (be concise):
{{
  "position": "{position}",
  "irac": {{
    "issue": "Legal issue in 1-2 sentences",
    "rule": "Applicable legal rule/doctrine (cite authority if known)",
    "application": "Apply rule to facts (2-3 sentences)",
    "conclusion": "Why {position} is correct (1 sentence)"
  }},
  "key_citations": ["Most relevant authority 1", "Authority 2"],
  "argument_summary": "One-sentence summary of your position"
}}"""

    return prompt


def get_debater_rebuttal_prompt(
    question: str,
    prompt_context: str,
    my_position: str,
    my_opening: dict,
    opponent_opening: dict
) -> str:
    """
    Generate rebuttal prompt for debater with IRAC structure and token-efficient format.

    Args:
        question: The legal question
        prompt_context: Context from dataset
        my_position: This debater's position
        my_opening: This debater's opening argument (dict)
        opponent_opening: Opponent's opening argument (dict)

    Returns:
        Formatted prompt string
    """
    if prompt_context:
        full_question = f"{prompt_context}\n\n{question}"
    else:
        full_question = question

    # Extract only key points to reduce tokens
    opponent_summary = opponent_opening.get('argument_summary', '')
    opponent_rule = opponent_opening.get('irac', {}).get('rule', '')
    opponent_application = opponent_opening.get('irac', {}).get('application', '')
    my_rule = my_opening.get('irac', {}).get('rule', '')
    my_summary = my_opening.get('argument_summary', '')

    prompt = f"""Continue your legal debate. Your position: {my_position}

Question: {full_question}

Your opening (summary): {my_summary}
Your rule: {my_rule}

Opponent's position: {opponent_opening.get('position', '')}
Opponent's summary: {opponent_summary}
Opponent's rule: {opponent_rule}
Opponent's application: {opponent_application}

Your task (be concise):
1. Identify flaw in opponent's rule or application
2. Strengthen your IRAC reasoning
3. Provide counter-argument

Respond in JSON:
{{
  "rebuttal_irac": {{
    "issue": "Refined issue statement",
    "rule": "Your rule (reinforced)",
    "application": "Counter-application addressing opponent's flaw",
    "conclusion": "Why your position remains superior"
  }},
  "counter_argument": "Main flaw in opponent's reasoning (1-2 sentences)",
  "key_citations": ["Additional authority if needed"],
  "rebuttal_summary": "One-sentence summary of your rebuttal"
}}"""

    return prompt


def get_judge_decision_prompt(
    question: str,
    prompt_context: str,
    choices: list,
    debate_history: dict
) -> str:
    """
    Generate judge decision prompt with IRAC analysis and token-efficient format.

    Args:
        question: The legal question
        prompt_context: Context from dataset
        choices: List of 4 answer choices
        debate_history: Full debate history with openings and rebuttals

    Returns:
        Formatted prompt string
    """
    choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])

    if prompt_context:
        full_question = f"{prompt_context}\n\n{question}"
    else:
        full_question = question

    # Extract only summaries and key IRAC components (token reduction)
    debater_x = debate_history.get('debater_x', {})
    debater_y = debate_history.get('debater_y', {})
    
    x_opening = debater_x.get('opening', {})
    x_rebuttal = debater_x.get('rebuttal', {})
    y_opening = debater_y.get('opening', {})
    y_rebuttal = debater_y.get('rebuttal', {})

    # Use summaries instead of full arguments
    x_opening_summary = x_opening.get('argument_summary', '')
    x_rebuttal_summary = x_rebuttal.get('rebuttal_summary', '')
    y_opening_summary = y_opening.get('argument_summary', '')
    y_rebuttal_summary = y_rebuttal.get('rebuttal_summary', '')
    
    x_rule = x_opening.get('irac', {}).get('rule', '')
    x_application = x_opening.get('irac', {}).get('application', '')
    y_rule = y_opening.get('irac', {}).get('rule', '')
    y_application = y_opening.get('irac', {}).get('application', '')

    prompt = f"""You are an impartial legal judge. Evaluate the debate using IRAC analysis.

Question: {full_question}
Choices: {choices_text}

Debater X (position {x_opening.get('position', '')}):
Opening: {x_opening_summary}
Rebuttal: {x_rebuttal_summary}
Key rule: {x_rule}
Application: {x_application}

Debater Y (position {y_opening.get('position', '')}):
Opening: {y_opening_summary}
Rebuttal: {y_rebuttal_summary}
Key rule: {y_rule}
Application: {y_application}

Evaluate using IRAC:
- Which issue statement is most accurate?
- Which rule is most applicable and well-cited?
- Which application to facts is most sound?
- Which conclusion is most legally correct?

Respond in JSON:
{{
  "decision": "A, B, C, or D",
  "irac_analysis": {{
    "best_issue": "Which debater framed the issue better",
    "best_rule": "Which rule is most applicable",
    "best_application": "Which application is most sound",
    "best_conclusion": "Which conclusion is correct"
  }},
  "rationale": Explain why you made this decision.,
  "key_factors": ["Factor 1", "Factor 2"]
}}"""

    return prompt
