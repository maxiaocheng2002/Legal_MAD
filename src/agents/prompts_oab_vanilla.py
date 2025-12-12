"""
Prompt templates for OAB open-ended questions using MAD Vanilla (no IRAC structure).
Simple and direct prompts for fast debate.
"""


def get_debater_opening_prompt_oab_vanilla(question: str, category: str) -> str:
    """
    Generate simple opening argument prompt for OAB (NEUTRAL debater, no IRAC).

    Args:
        question: The open-ended legal question
        category: Law category (e.g., "Direito Penal")

    Returns:
        Formatted prompt string
    """
    prompt = f"""Você é um especialista em {category}. Responda esta questão da OAB de forma dissertativa.

Questão:
{question}

Sua tarefa:
Escreva uma resposta completa e fundamentada para a questão acima.
- Desenvolva sua argumentação jurídica de forma clara
- Cite as normas, doutrinas e jurisprudências aplicáveis
- Aplique os fundamentos ao caso concreto
- Conclua de forma objetiva

Responda em JSON:
{{
  "answer": "Sua resposta dissertativa completa (5-8 parágrafos)",
  "key_citations": ["Lei/artigo 1", "Doutrina/jurisprudência 2", "..."]
}}

Importante: Resposta com qualidade de prova da OAB - fundamentada, clara e objetiva."""

    return prompt


def get_debater_opening_prompt_oab_adversarial_vanilla(
    question: str,
    category: str,
    opponent_opening: dict
) -> str:
    """
    Generate ADVERSARIAL opening prompt - debater sees opponent's position first (no IRAC).

    Args:
        question: The open-ended legal question
        category: Law category
        opponent_opening: Opponent's opening argument (to present alternative view)

    Returns:
        Formatted prompt string
    """
    opponent_answer = opponent_opening.get('answer', '')[:800]  # Truncate for context

    prompt = f"""Você é um especialista em {category}. Seu papel é ser ADVERSARIAL.

Você acaba de ver a posição do seu oponente sobre esta questão da OAB:

Questão:
{question}

POSIÇÃO DO OPONENTE:
{opponent_answer}...

Sua tarefa:
Apresente uma perspectiva DIFERENTE ou abordagem ALTERNATIVA para esta mesma questão.
- Você pode discordar da interpretação normativa do oponente
- Você pode identificar outras normas aplicáveis que ele não mencionou
- Você pode dar ênfase diferente aos fatos do caso
- Você pode chegar a conclusão diferente

Responda em JSON:
{{
  "answer": "Sua resposta dissertativa ALTERNATIVA completa (5-8 parágrafos)",
  "key_citations": ["Lei/artigo 1", "Doutrina/jurisprudência 2", "..."]
}}

Importante: Seja genuinamente adversarial - não apenas repita o oponente com palavras diferentes."""

    return prompt


def get_debater_rebuttal_prompt_oab_vanilla(
    question: str,
    category: str,
    my_opening: dict,
    opponent_opening: dict
) -> str:
    """
    Generate rebuttal prompt for OAB debate (no IRAC structure).

    Args:
        question: The legal question
        category: Law category
        my_opening: This debater's opening argument
        opponent_opening: Opponent's opening argument

    Returns:
        Formatted prompt string
    """
    my_answer = my_opening.get('answer', '')
    opp_answer = opponent_opening.get('answer', '')

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


def get_judge_synthesis_prompt_oab_vanilla(
    question: str,
    category: str,
    debater_x_rebuttal: dict,
    debater_y_rebuttal: dict
) -> str:
    """
    Generate judge synthesis prompt for OAB - creates final objective answer (no IRAC).

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
A resposta final deve ser COMPLETA e FUNDAMENTADA:
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
