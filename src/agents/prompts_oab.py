"""
Prompt templates for OAB open-ended legal questions using MAD.
"""


def get_debater_opening_prompt_oab(question: str, category: str, is_adversarial: bool = False) -> str:
    """
    Generate opening argument prompt for OAB open-ended question (NEUTRAL debater).

    Args:
        question: The open-ended legal question
        category: Law category (e.g., "Direito Penal")
        is_adversarial: Legacy parameter (not used, kept for compatibility)

    Returns:
        Formatted prompt string
    """
    role_instruction = f"Você é um especialista em {category}. Escreva uma resposta dissertativa para a seguinte questão da OAB."

    prompt = f"""{role_instruction}

Questão:
{question}

Sua tarefa:
Desenvolva uma resposta completa e fundamentada, estruturada em IRAC (Issue, Rule, Application, Conclusion):
- Issue: Identifique a questão jurídica central
- Rule: Cite as normas, doutrinas e jurisprudências aplicáveis
- Application: Aplique as normas aos fatos apresentados
- Conclusion: Conclua de forma clara e objetiva

Responda em JSON:
{{
  "irac": {{
    "issue": "Questão jurídica identificada (2-3 frases)",
    "rule": "Normas aplicáveis com citações (art. X, lei Y, súmula Z)",
    "application": "Aplicação ao caso concreto (4-5 frases)",
    "conclusion": "Conclusão fundamentada (2-3 frases)"
  }},
  "full_answer": "Resposta dissertativa completa em formato de prova (5-8 parágrafos)",
  "key_citations": ["Lei/artigo 1", "Doutrina/jurisprudência 2", "..."]
}}

Importante: A resposta deve ter qualidade de prova da OAB - fundamentada, clara e objetiva."""

    return prompt


def get_debater_opening_prompt_oab_adversarial(
    question: str,
    category: str,
    opponent_opening: dict
) -> str:
    """
    Generate ADVERSARIAL opening prompt - debater sees opponent's position first.

    Args:
        question: The open-ended legal question
        category: Law category
        opponent_opening: Opponent's opening argument (to present alternative view)

    Returns:
        Formatted prompt string
    """
    opponent_answer = opponent_opening.get('full_answer', '')[:800]  # Truncate for context
    opponent_issue = opponent_opening.get('irac', {}).get('issue', '')
    opponent_rule = opponent_opening.get('irac', {}).get('rule', '')

    prompt = f"""Você é um especialista em {category}. Seu papel é ser ADVERSARIAL.

Você acaba de ver a posição do seu oponente sobre esta questão da OAB:

Questão:
{question}

POSIÇÃO DO OPONENTE:
Issue identificada: {opponent_issue}
Normas citadas: {opponent_rule}
Resposta resumida: {opponent_answer}...

Sua tarefa:
Apresente uma perspectiva DIFERENTE ou abordagem ALTERNATIVA para esta mesma questão.
- Você pode discordar da interpretação normativa do oponente
- Você pode identificar outras normas aplicáveis que ele não mencionou
- Você pode dar ênfase diferente aos fatos do caso
- Você pode chegar a conclusão diferente

Desenvolva sua resposta em IRAC (Issue, Rule, Application, Conclusion):

Responda em JSON:
{{
  "irac": {{
    "issue": "Questão jurídica identificada (pode ser diferente da do oponente)",
    "rule": "Normas aplicáveis - explore normas alternativas ou interpretações diferentes",
    "application": "Aplicação ao caso concreto com perspectiva diferente",
    "conclusion": "Conclusão fundamentada (pode divergir do oponente)"
  }},
  "full_answer": "Resposta dissertativa completa apresentando perspectiva ALTERNATIVA (5-8 parágrafos)",
  "key_citations": ["Lei/artigo 1", "Doutrina/jurisprudência 2", "..."]
}}

Importante: Seja genuinamente adversarial - não apenas repita o oponente com palavras diferentes."""

    return prompt


def get_debater_rebuttal_prompt_oab(
    question: str,
    category: str,
    my_opening: dict,
    opponent_opening: dict
) -> str:
    """
    Generate rebuttal prompt for OAB debate.

    Args:
        question: The legal question
        category: Law category
        my_opening: This debater's opening argument
        opponent_opening: Opponent's opening argument

    Returns:
        Formatted prompt string
    """
    my_answer = my_opening.get('full_answer', '')
    opp_answer = opponent_opening.get('full_answer', '')

    prompt = f"""Continue o debate sobre esta questão da OAB em {category}.

Questão: {question}

Sua resposta inicial:
{my_answer}

Resposta do oponente:
{opp_answer}

Sua tarefa (REBUTTAL):
1. Analise criticamente a resposta do oponente - identifique falhas, omissões ou fraquezas
2. Reforce sua própria argumentação com novos fundamentos/citações
3. Produza uma versão REFINADA e MELHORADA da sua resposta dissertativa

Responda em JSON:
{{
  "critique": "Principais falhas ou fraquezas na resposta do oponente (3-4 frases)",
  "refined_answer": "Sua resposta dissertativa COMPLETA e REFINADA (incorpore melhorias e novos argumentos)",
  "key_citations": ["Todas as citações relevantes na resposta refinada"]
}}

Importante: refined_answer deve ser uma resposta COMPLETA e autossuficiente."""

    return prompt


def get_judge_synthesis_prompt_oab(
    question: str,
    category: str,
    debater_x_rebuttal: dict,
    debater_y_rebuttal: dict
) -> str:
    """
    Generate judge synthesis prompt for OAB - creates final objective answer.

    Args:
        question: The legal question
        category: Law category
        debater_x_rebuttal: Debater X's rebuttal with refined answer
        debater_y_rebuttal: Debater Y's rebuttal with refined answer

    Returns:
        Formatted prompt string
    """
    x_answer = debater_x_rebuttal.get('refined_answer', '')
    y_answer = debater_y_rebuttal.get('refined_answer', '')
    x_critique = debater_x_rebuttal.get('critique', '')
    y_critique = debater_y_rebuttal.get('critique', '')

    prompt = f"""Você é avaliador imparcial OAB - {category}.

<question>
{question}
</question>

<debate>
<debater_x>
<refined_answer>
{x_answer}
</refined_answer>
<critique_of_y>
{x_critique}
</critique_of_y>
</debater_x>

<debater_y>
<refined_answer>
{y_answer}
</refined_answer>
<critique_of_x>
{y_critique}
</critique_of_x>
</debater_y>
</debate>

<task>
Analise o debate. Produza a melhor resposta para a questão OAB.

Diretrizes:
- Liberdade total para escolher abordagem (não precisa usar ambos debatedores)
- Use debate para identificar: norma aplicável, interpretação, pontos fortes/fracos

Formato resposta OAB (espelho de correção):
A resposta final deve ser COMPLETA e FUNDAMENTADA, seguindo este padrão:
1. Comece com posicionamento direto (Sim/Não, quando aplicável)
2. Desenvolva fundamentação jurídica aplicada ao caso (2-4 frases)
3. Cite normas precisas: Art. X, § Y, Lei Z / Súmula N STF/STJ
4. Extensão total: 4-6 frases completas (120-180 palavras)

IMPORTANTE: final_answer deve ser AUTOSSUFICIENTE - não apenas "Sim" ou "Não", mas resposta COMPLETA com fundamentos normativos e citações.
</task>

<example>
Exemplo de final_answer CORRETA (baseada em espelho OAB real):

"Sim, Jaqueline, como agente público responsável pelo controle interno, ao tomar conhecimento da ilegalidade por fraude contratual, deveria ter dado ciência ao Tribunal de Contas da União e, diante de sua omissão, está sujeita à responsabilidade solidária, conforme dispõe o Art. 74 § 1º, da CRFB/88."

Exemplo de final_answer INCORRETA:
"Sim." ← INCOMPLETO - falta fundamentação e citação normativa
</example>

<output_format>
Responda em JSON:
{{
  "final_answer": "Resposta COMPLETA e FUNDAMENTADA estilo espelho OAB. Estrutura: '[Posicionamento]. [Fundamentação jurídica aplicada ao caso em 2-4 frases]. [Citação normativa precisa: Art. X, § Y, Lei Z].' (4-6 frases, 120-180 palavras)",
  "rationale": "Como o debate fundamentou esta resposta (2-3 frases)",
  "sources_used": {{
    "from_debater_x": ["elementos úteis do X"],
    "from_debater_y": ["elementos úteis do Y"],
    "judge_reasoning": "Síntese/interpretação do debate"
  }},
  "key_citations": ["Art. X Lei Y", "Súmula Z STF/STJ"]
}}
</output_format>

<constraints>
- final_answer: COMPLETA (não apenas Sim/Não), FUNDAMENTADA, com CITAÇÕES PRECISAS
- Citações: Art. X, § Y, Lei Z (formato exato)
- Objetiva mas autossuficiente (4-6 frases)
</constraints>"""

    return prompt
