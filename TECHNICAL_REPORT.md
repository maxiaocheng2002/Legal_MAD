# Technical Report: Multi-Agent Debate for Brazilian Legal Reasoning

**Project**: MAD_legal
**Target**: ACL 2025 Submission
**Domain**: Legal Reasoning (US Bar Exam MCQ + Brazilian Bar Exam Open-Ended)
**Date**: December 2024

---

## 1. Research Question and Motivation

**Research Question**: Can Multi-Agent Debate (MAD) with structured legal reasoning (IRAC framework) improve LLM performance on legal reasoning tasks compared to single-agent baselines?

**Motivation**:
- Legal reasoning requires multi-perspective analysis, structured argumentation, and citation accuracy
- Brazilian legal system has unique characteristics (civil law, Portuguese language, specific legislation)
- Existing work on MAD focuses primarily on factual/mathematical tasks, not legal reasoning
- Need for benchmarks evaluating both multiple-choice and open-ended legal questions

---

## 2. Multi-Agent Debate Framework

### 2.1 MAD Architectures

**MAD IRAC (Structured)**:
- Debaters follow IRAC framework (Issue, Rule, Application, Conclusion)
- Explicit legal reasoning structure in prompts
- Judge synthesizes using IRAC methodology

**MAD Vanilla (Simple)**:
- General debate without IRAC constraints
- Simpler prompts focused on adversarial argumentation
- Judge performs general synthesis

### 2.2 Agent Roles

**Debater X (Neutral)**:
- Opens debate with initial legal analysis
- Provides balanced legal reasoning
- Generates rebuttal based on opponent's arguments

**Debater Y (Adversarial)**:
- Sees Debater X's opening position
- Takes adversarial stance (challenges or provides alternative view)
- Generates counter-arguments and alternative legal interpretations

**Judge**:
- Reviews both debaters' final rebuttals
- Synthesizes consensus answer
- Extracts key legal citations (for open-ended questions)
- Provides rationale for final decision

### 2.3 Debate Structure

**Round 1 - Openings**:
1. Debater X generates opening argument (neutral, IRAC-structured or vanilla)
2. Debater Y generates adversarial opening after seeing X's position

**Round 2 - Rebuttals**:
1. Debater X generates rebuttal addressing Y's counter-arguments
2. Debater Y generates rebuttal addressing X's arguments

**Synthesis**:
- Judge reviews both final rebuttals
- Generates final answer with legal citations and rationale

---

## 3. Datasets

### 3.1 Dataset 01: US Bar Exam Multiple-Choice Questions (MCQ)

**Source**: US Bar Exam (First Phase)
**Type**: Multiple-choice questions
**Size**: ~500 questions
**Answer Format**: A, B, C, or D (4 alternatives)
**Evaluation Metric**: Accuracy (exact match)

**Categories** (following US Bar Exam Phase 1 structure):
- Constitutional Law
- Contracts
- Criminal Law and Procedure
- Evidence
- Real Property
- Torts
- Civil Procedure

**Challenges**:
- Requires knowledge of US legal system (Common Law)
- Context-dependent reasoning with legal precedents
- Multiple plausible options requiring careful legal analysis

**MAD Adaptation for MCQ**:
- **Debater X**: Analyzes all 4 alternatives and chooses the one that makes most sense, providing argumentation for that choice
- **Debater Y**: Must choose a DIFFERENT alternative (excluding X's choice) and argue why it is better
- **Judge**: Reviews both arguments and selects final answer (A, B, C, or D)
- **Rationale**: Forces adversarial debate by requiring debaters to defend different alternatives, preventing agreement bias

### 3.2 Dataset 02: Brazilian OAB Open-Ended Questions (OAB-Bench)

**Source**: Brazilian Bar Exam (Exame da OAB) - Dissertative portion
**Type**: Open-ended legal questions
**Size**: 168 questions
**Answer Format**: Free-text response (dissertative)
**HuggingFace**: `ricardorudicz/oab_exams_guidelines`

**Ground Truth Structure**:
```json
{
  "question_id": "...",
  "statement": "Legal question...",
  "category": "Constitutional Law",
  "ground_truth": {
    "reference_answer": "Official answer key...",
    "key_citations_expected": [
      "Art. 5, inciso LXIX, CF/88",
      "Art. 1, Lei 12.016/2009"
    ]
  }
}
```

**Evaluation Metrics** (3 automatic + 1 human):
1. **Citation F1**: Precision, recall, F1 for exact legal citation matching
2. **BERTScore**: Semantic similarity using BERTimbau (Portuguese BERT)
3. **LLM-as-Judge**: GPT-4o-mini evaluation on 3 criteria:
   - Correctness (0-4): Legal accuracy
   - Reasoning (0-3): Logical structure
   - Citations (0-4): Citation quality (exact match, near match, wrong law, etc.)
4. **Human Evaluation** (planned): Expert lawyer annotation with Cohen's Kappa

**Citation Extraction**:
- Regex-based parser for Brazilian legal citations
- Captures: Articles, paragraphs, incisos, laws, súmulas, codes (CF/88, CP, CC, etc.)
- Preprocessing: Citations extracted from reference answers before experiments
- Handles complex patterns: "Art. 6º, inciso XXII, da Lei nº 14.133/21"

---

## 4. Experimental Design

### 4.1 Compared Methods

**MAD Variants**:
- MAD IRAC (structured legal reasoning)
- MAD Vanilla (simple debate)

**Baselines**:
- Single-agent (direct prompting)
- Chain-of-Thought (CoT)
- Self-Consistency (SC) with majority voting

### 4.2 Models Tested

All experiments run with 4 LLMs via OpenRouter (free tier):
1. **Llama 3.3 70B Instruct** (`meta-llama/llama-3.3-70b-instruct:free`)
2. **Llama 3.1 70B Instruct** (`meta-llama/llama-3.1-70b-instruct:free`)
3. **Qwen 2.5 72B Instruct** (`qwen/qwen-2.5-72b-instruct:free`)
4. **Mistral 7B Instruct** (`mistralai/mistral-7b-instruct:free`)

**Total Experiments**:
- Dataset 01: 2 MAD variants + 3 baselines = 5 methods × 4 models = 20 experiments
- Dataset 02: 2 MAD variants + 3 baselines = 5 methods × 4 models = 20 experiments
- **Total**: 40 experiments across both datasets

### 4.3 Experimental Protocol

**Dataset 01 (MCQ)**:
- Input: Question + 4 options (A-D)
- MAD Debate Structure:
  - Debater X selects one alternative and argues for it
  - Debater Y selects a DIFFERENT alternative (excluding X's choice) and argues for it
  - Judge synthesizes final answer based on both arguments
- Output: Selected letter (A, B, C, or D)
- Evaluation: Accuracy (% correct)

**Dataset 02 (Open-Ended)**:
- Input: Dissertative legal question
- Output: Free-text answer + extracted citations
- Evaluation: Citation F1 + BERTScore + LLM-as-Judge
- Sample size: 168 questions per experiment

**Implementation Details**:
- API: OpenRouter with retry logic (max 10 retries, exponential backoff)
- Checkpointing: Every 10 questions to prevent data loss
- Temperature: 0.7 (debate), 0.1 (evaluation)
- Max tokens: 2000 (generation), 500 (LLM-as-Judge)

---

## 5. Evaluation Methodology

### 5.1 Dataset 01: Multiple-Choice Evaluation

**Primary Metric**: Accuracy
- Exact match between predicted letter and ground truth
- Reported as percentage correct

**Analysis**:
- Per-category breakdown (Constitutional, Contracts, Criminal, etc.)
- Model comparison (statistical significance testing planned)

### 5.2 Dataset 02: Open-Ended Evaluation

#### 5.2.1 Citation F1 (Exact Match)

**Purpose**: Measure legal citation accuracy
**Calculation**:
- Precision = (Predicted ∩ Expected) / |Predicted|
- Recall = (Predicted ∩ Expected) / |Expected|
- F1 = 2 × (Precision × Recall) / (Precision + Recall)

**Strictness**: Exact match only
- "Art. 5, CF/88" ≠ "Art. 74, CF/88" (different articles)
- Captures perfect citation accuracy

**Limitation**: Does not capture "near misses" (e.g., citing correct law but wrong article)

#### 5.2.2 BERTScore (Semantic Similarity)

**Purpose**: Measure semantic similarity of answers
**Model**: BERTimbau (`neuralmind/bert-base-portuguese-cased`)
**Library**: `bert_score` with HuggingFace model specification

**Advantages**:
- Language-specific (Portuguese BERT)
- Captures semantic meaning beyond exact word match
- Model caching for efficiency (~30s first load, then fast)

**Metrics**: Precision, Recall, F1 on token embeddings

#### 5.2.3 LLM-as-Judge (Qualitative Evaluation)

**Purpose**: Comprehensive legal reasoning evaluation
**Model**: GPT-4o-mini (via OpenRouter)
**Rationale**: Different model from experiments to avoid bias

**Evaluation Rubric**:
```
1. CORRECTNESS (0-4 points):
   0 = Completely incorrect or irrelevant
   1 = Partially correct with serious errors
   2 = Correct but incomplete
   3 = Correct and complete
   4 = Exceptionally well-founded

2. REASONING (0-3 points):
   0 = No legal logic or incoherent
   1 = Basic reasoning present
   2 = Clear and structured reasoning
   3 = Excellent reasoning (e.g., IRAC-like structure)

3. CITATIONS (0-4 points):
   0 = No citation or completely wrong
   1 = Correct law/code but wrong article
   2 = Related/nearby article cited
   3 = Correct article but incomplete (missing paragraphs/incisos)
   4 = Perfect and complete citation
```

**Output**: Normalized score (0-1) = Total / 11

**Prompt Structure**: XML-tagged with task, question, reference, candidate answer, evaluation criteria

#### 5.2.4 Human Evaluation (Planned)

**Sample**: 30-50 questions (stratified by category)
**Annotators**: Lawyers with OAB membership
**Protocol**:
- Blind annotation (no model names shown)
- Same 3-criteria rubric as LLM-as-Judge
- Inter-annotator agreement: Cohen's Kappa
- Correlation with LLM-as-Judge: Pearson correlation

**Purpose**:
- Validate LLM-as-Judge reliability
- Provide human gold standard
- Identify model failure modes

---

## 6. Implementation Details

### 6.1 Citation Extraction

**Method**: Regex-based parser for Brazilian legal citations
**File**: `src/evaluation/citation_parser.py`

**Captured Patterns**:
1. **Article + Code**: "Art. 74, § 1º, CF/88"
2. **Article + Law**: "Art. 6º, inciso XXII, da Lei nº 14.133/21"
3. **Standalone Laws**: "Lei 8.112/1990"
4. **Súmulas**: "Súmula 473 STF", "Súmula Vinculante 13"
5. **Articles with Context**: "Art. 121 do CP"

**Normalization**:
- Year formats: "90" → "1990", "21" → "2021"
- Code abbreviations: "CF", "CRFB" → "CF/88"
- Ordinal indicators: "1º", "2º" preserved

**Deduplication**:
- Position tracking to avoid capturing "Lei 14.133/21" when already captured as part of "Art. 6º, inciso XXII, da Lei nº 14.133/21"

**Preprocessing**:
- Citations extracted from reference answers during dataset loading
- Stored in `ground_truth.key_citations_expected`
- Models also extract citations from their generated answers

### 6.2 Prompt Engineering

**IRAC Framework Prompts**:
- **Issue**: Identify the legal question
- **Rule**: State applicable legal rules and citations
- **Application**: Apply rules to facts
- **Conclusion**: Provide clear answer with legal basis

**Adversarial Prompting (MCQ)**:
- Debater X: Analyze all alternatives, select one, argue for it
- Debater Y: Select DIFFERENT alternative (excluding X's choice), argue for it
- Forces genuine debate by requiring different positions

**Adversarial Prompting (Open-Ended)**:
- Debater Y explicitly instructed to challenge or provide alternative view
- Sees Debater X's position to create genuine debate

**Judge Synthesis**:
- XML-structured prompt with both rebuttals
- Instruction to synthesize consensus answer
- Explicit request for legal citations and rationale

### 6.3 API Infrastructure

**Primary API**: OpenRouter
- Free-tier models (Llama, Qwen, Mistral)
- JSON mode support for structured outputs
- Rate limiting: 10 requests/minute

**Retry Logic**:
- Max retries: 10
- Exponential backoff: 2s, 4s, 8s, ...
- Handles: Rate limits, timeouts, API errors

**Checkpointing**:
- Save every 10 questions
- Resume from last checkpoint on failure
- Delete checkpoint on successful completion

**Error Handling**:
- Skip questions with errors (saved with `{"error": "..."}`)
- Continue processing remaining questions
- Report error count in final summary

---

## 7. Results Status

### 7.1 Dataset 01 (MCQ)

**Status**: Partial results available

**Completed Experiments**:
- MAD IRAC with Llama 3.3 70B: ~68% accuracy
- MAD Vanilla (partial results)
- Some baseline experiments in progress

**Pending**:
- Complete all 4 models for MAD IRAC and Vanilla
- Run all baselines (Single, CoT, SC) with 4 models
- Statistical analysis and significance testing

### 7.2 Dataset 02 (OAB-Bench)

**Status**: Experiments in progress

**Completed**:
- MAD Vanilla with 4 models (168 questions each)
- Evaluation pipeline tested on small samples

**In Progress**:
- MAD IRAC with 4 models (168 questions each)
- Baseline experiments being distributed to collaborators

**Pending**:
- Complete all experiments
- Run full evaluation with LLM-as-Judge
- Human evaluation protocol execution
- Statistical analysis

**Preliminary Observations** (from small samples):
- Citation F1: Low (0.0-0.2) - expected due to strict exact matching
- BERTScore: Moderate (0.6-0.7) - shows semantic similarity
- LLM-as-Judge: Moderate (0.5-0.6) - captures nuanced legal reasoning

---

## 8. Key Contributions

### 8.1 Novel MAD Application

**First MAD for Legal Reasoning**:
- Previous MAD work focused on factual/mathematical tasks
- Legal reasoning requires structured argumentation and citation accuracy
- Cross-language evaluation (English MCQ + Portuguese open-ended)

**IRAC-Structured Debate**:
- Integration of legal reasoning framework into MAD
- Explicit prompting for Issue-Rule-Application-Conclusion structure
- Comparison between structured (IRAC) and unstructured (Vanilla) debate

**MCQ Debate Adaptation**:
- Novel approach for MAD on multiple-choice questions
- Forced adversarial debate by requiring different alternative selections
- Prevents agreement bias common in MAD approaches

### 8.2 Benchmark Creation

**OAB-Bench for Open-Ended Legal Questions**:
- 168 dissertative questions with official answer keys
- Multi-metric evaluation (Citation F1, BERTScore, LLM-as-Judge)
- Ground truth with extracted legal citations
- HuggingFace dataset for reproducibility

**Dual-Dataset Approach**:
- Dataset 01 (US Bar MCQ): Tests legal knowledge and decision-making
- Dataset 02 (Brazilian OAB Open-Ended): Tests argumentation and citation accuracy
- Complementary evaluation of different legal reasoning aspects

### 8.3 Multi-Metric Evaluation

**Comprehensive Assessment**:
- Citation F1: Exact citation accuracy
- BERTScore: Semantic similarity (language-specific)
- LLM-as-Judge: Nuanced legal reasoning evaluation (correctness, reasoning, citations)
- Human evaluation: Gold standard validation

**Citation-Specific Metrics**:
- Citation F1 captures perfect match
- LLM-as-Judge captures near-misses (e.g., correct law, wrong article scored as 1/4 instead of 0/4)
- Complements semantic similarity with citation-specific evaluation

### 8.4 Methodological Insights

**MAD Architecture Comparison**:
- IRAC vs Vanilla: Does structured legal reasoning improve performance?
- Multi-round debate: Does adversarial argumentation help?
- Judge synthesis: Does debate convergence improve over single-agent?

**Baseline Comparisons**:
- Single-agent: Direct baseline
- CoT: Reasoning chain baseline
- Self-Consistency: Ensemble baseline
- Statistical significance of MAD improvements

### 8.5 Open-Source Contribution

**Released Artifacts**:
- OAB-Bench dataset (HuggingFace: ricardorudicz/oab_exams_guidelines)
- Citation extraction parser (Brazilian legal citations)
- Evaluation pipeline (Citation F1, BERTScore, LLM-as-Judge)
- MAD implementation (IRAC and Vanilla variants)

**Reproducibility**:
- Documented prompts
- API configurations
- Evaluation protocols
- Code available on GitHub

---

## 9. Technical Challenges Addressed

### 9.1 MCQ Debate Structure

**Challenge**: How to create adversarial debate for multiple-choice questions?

**Solution**:
- Debater X selects one alternative and argues for it
- Debater Y MUST select a different alternative (excluding X's choice)
- Forces genuine debate instead of agreement
- Judge synthesizes final answer based on arguments

### 9.2 Citation Extraction

**Challenge**: Brazilian legal citations have complex structure
- Articles with paragraphs, incisos, alíneas
- Laws with year normalization
- Multiple code abbreviations (CF, CRFB, CP, CC, etc.)

**Solution**: Multi-pattern regex parser with position tracking to avoid duplicates

### 9.3 Evaluation Pipeline

**Challenge**: Open-ended answers require multiple complementary metrics

**Solution**:
- Citation F1: Exact citation matching
- BERTScore: Semantic similarity (Portuguese-specific)
- LLM-as-Judge: Qualitative reasoning assessment
- Post-processing: Automatic evaluation pipeline

### 9.4 Scalability

**Challenge**: 168 questions × 5 methods × 4 models = 3,360 API calls per dataset

**Solution**:
- Checkpointing every 10 questions
- Retry logic with exponential backoff
- Removed duplicate BERTScore calculation (was computed during experiments AND evaluation)
- Parallel execution by collaborators (each runs 1 baseline × 4 models)

### 9.5 BERTScore Efficiency

**Challenge**: BERTScore was calculated twice (during experiments and evaluation)

**Solution**:
- Removed BERTScore from experiment pipeline
- Calculate only during evaluation phase
- 30-40% faster experiment execution
- Model caching for efficiency

---

---

**End of Technical Report**

