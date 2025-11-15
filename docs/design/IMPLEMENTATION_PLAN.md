# Implementation Plan - MAD Legal Reasoning

**Version:** 1.0
**Date:** November 12, 2024
**Timeline:** Week 1 (Days 1-7)
**Status:** ✅ Final - Ready to Execute

---

## 🎯 Goal

Build working MAD system for Phase 1 experiments:
1. Baseline (single-agent)
2. MAD for MCQ (2 rounds, adversarial, alternatives visible)
3. MAD for Open-ended (2 rounds, adversarial, free text)
4. Structured JSON outputs for all components

**Scope:** Minimum viable, scientifically valid, ready for Week 2 experiments.

---

## 📁 Code Structure

```
src/
├── agents/
│   ├── debater.py          # Debater agent (MCQ + open-ended)
│   ├── judge.py            # Judge agent
│   └── prompts.py          # All prompt templates
├── baselines/
│   ├── single_agent.py     # B1: Zero-shot
│   ├── chain_of_thought.py # B2: CoT
│   └── self_consistency.py # B3: Self-consistency
├── evaluation/
│   └── metrics.py          # Citation, argument metrics
├── utils/
│   ├── api_client.py       # Groq API wrapper
│   ├── data_loader.py      # Dataset loading
│   └── parsers.py          # JSON parsing, validation
└── experiments/
    ├── run_baseline.py     # Run baselines
    ├── run_mad.py          # Run MAD experiments
    └── config.yaml         # Configuration
```

---

## 🔧 Tech Stack

### Core
- Python 3.10+
- `groq` (Groq SDK)
- `datasets` (HuggingFace)
- `pydantic` (JSON validation)

### Evaluation
- `rouge-score`
- `bert-score`
- `scikit-learn`

### Utils
- `pyyaml`, `tqdm`, `python-dotenv`

---

## 🏗️ MAD Architecture (Final Design)

### **MCQ: 2 Rounds, Adversarial, Alternatives Visible**

```
Input: Question + 4 alternatives (A, B, C, D)

Round 1 (Opening - Simultaneous):
  - Debater X: Choose position (e.g., A), argue why A is correct
  - Debater Y: Choose different position (e.g., B), argue why B is correct
  - Both see all alternatives

Round 2 (Rebuttal - Sequential):
  - Debater X sees Y's argument → rebuts Y, reinforces X
  - Debater Y sees X's argument → rebuts X, reinforces Y

Judge:
  - Sees question + alternatives + full debate (Round 1 + 2)
  - Selects best-supported alternative
  - Provides rationale

Output: JSON with decision + debate logs
```

### **Open-Ended: 2 Rounds, Adversarial, Free Text**

```
Input: Question only (no rubrica)

Round 1 (Opening - Simultaneous):
  - Debater X: Constructs legal argument from scratch
  - Debater Y: Constructs alternative/complementary argument

Round 2 (Rebuttal - Sequential):
  - Debater X sees Y's argument → identifies gaps, refutes Y
  - Debater Y sees X's argument → identifies gaps, refutes X

Judge:
  - Sees question + full debate (Round 1 + 2)
  - Synthesizes best elements from both arguments
  - Produces final comprehensive answer

Output: JSON with synthesis + debate logs
```

---

## 📝 JSON Output Structure

### MCQ Output

```json
{
  "question_id": "bar_001",
  "question_text": "...",
  "alternatives": ["A: ...", "B: ...", "C: ...", "D: ..."],
  "gold_answer": "B",
  "debate": {
    "round_1": {
      "debater_x": {
        "position": "A",
        "argument": "...",
        "citations": ["Art. X", "Case Y"]
      },
      "debater_y": {
        "position": "B",
        "argument": "...",
        "citations": ["Art. Z"]
      }
    },
    "round_2": {
      "debater_x": {
        "rebuttal": "...",
        "citations": [...]
      },
      "debater_y": {
        "rebuttal": "...",
        "citations": [...]
      }
    }
  },
  "judge": {
    "decision": "B",
    "rationale": "...",
    "correct": true
  }
}
```

### Open-Ended Output

```json
{
  "question_id": "oab_001",
  "question_text": "...",
  "debate": {
    "round_1": {
      "debater_x": {
        "argument": "...",
        "citations": [...],
        "irac": {
          "issue": "...",
          "rule": "...",
          "application": "...",
          "conclusion": "..."
        }
      },
      "debater_y": {...}
    },
    "round_2": {
      "debater_x": {
        "rebuttal": "...",
        "counterarguments": [...]
      },
      "debater_y": {...}
    }
  },
  "judge": {
    "synthesis": "...",
    "final_answer": "...",
    "citations_used": [...]
  }
}
```

**Key:** All intermediate outputs (Round 1, Round 2) saved for later analysis/evaluation.

---

## 🚀 Week 1 Implementation Timeline

### Day 1-2: Setup & Baseline

- [ ] Python environment + dependencies
- [ ] Groq API key setup
- [ ] Load Bar_Exam_QA (sample 50 questions)
- [ ] Implement `api_client.py` (Groq wrapper)
- [ ] Implement `single_agent.py` (B1 baseline)
- [ ] Test B1 on 5 questions

**Deliverable:** B1 generating answers in JSON

---

### Day 3-4: MAD for MCQ

- [ ] Implement `debater.py` (MCQ mode)
  - Round 1: Opening with position selection
  - Round 2: Rebuttal seeing opponent
- [ ] Implement `judge.py` (MCQ mode)
  - Decision logic
- [ ] Implement `run_mad.py` (MCQ orchestration)
  - 2 rounds flow
  - JSON output
- [ ] Test MAD-MCQ on 10 questions

**Deliverable:** Working MAD for MCQ

---

### Day 5: MAD for Open-Ended

- [ ] Adapt `debater.py` (open-ended mode)
  - Free-form argument construction
  - IRAC-structured output
- [ ] Adapt `judge.py` (synthesis mode)
  - Combine best elements
- [ ] Test MAD-OpenEnded on OAB (5 questions)

**Deliverable:** Working MAD for open-ended

---

### Day 6: Other Baselines

- [ ] Implement `chain_of_thought.py` (B2)
- [ ] Implement `self_consistency.py` (B3)
- [ ] Test all baselines

**Deliverable:** All baselines working

---

### Day 7: Integration & Validation

- [ ] Run full pipeline on 50 Bar_Exam_QA
- [ ] Run on 20 OAB
- [ ] Verify JSON outputs valid
- [ ] Spot-check quality manually
- [ ] Document issues

**Deliverable:** End-to-end pipeline ready for Week 2

---

## 🔑 Key Implementation Details

### 1. Position Assignment (MCQ)

```python
import random

def assign_positions(question):
    """Randomly assign 2 different alternatives to debaters."""
    alternatives = question["choices"]  # ["A", "B", "C", "D"]
    positions = random.sample(alternatives, 2)
    return positions[0], positions[1]  # e.g., ("B", "D")
```

### 2. Debate Flow (MCQ)

```python
def run_mad_mcq(question):
    # Assign positions
    pos_x, pos_y = assign_positions(question)

    # Round 1: Opening (simultaneous)
    arg_x = debater_x.opening(question, position=pos_x, alternatives=question["choices"])
    arg_y = debater_y.opening(question, position=pos_y, alternatives=question["choices"])

    # Round 2: Rebuttal (sequential)
    reb_x = debater_x.rebuttal(question, my_arg=arg_x, opponent_arg=arg_y)
    reb_y = debater_y.rebuttal(question, my_arg=arg_y, opponent_arg=arg_x)

    # Judge
    decision = judge.decide_mcq(question, arg_x, arg_y, reb_x, reb_y)

    return {
        "debate": {"round_1": {}, "round_2": {}},
        "judge": decision
    }
```

### 3. Debate Flow (Open-Ended)

```python
def run_mad_open(question):
    # Round 1: Opening (simultaneous, independent)
    arg_x = debater_x.opening_open(question)
    arg_y = debater_y.opening_open(question)

    # Round 2: Rebuttal (sequential, see opponent)
    reb_x = debater_x.rebuttal_open(question, my_arg=arg_x, opponent_arg=arg_y)
    reb_y = debater_y.rebuttal_open(question, my_arg=arg_y, opponent_arg=arg_x)

    # Judge synthesizes
    synthesis = judge.synthesize_open(question, arg_x, arg_y, reb_x, reb_y)

    return {
        "debate": {"round_1": {}, "round_2": {}},
        "judge": synthesis
    }
```

### 4. Prompt Engineering (Critical)

**Debater Opening (MCQ):**
```
You are a legal expert participating in a debate.

Question: {question}

Alternatives:
A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Your task: Argue convincingly that Alternative {position} is the legally correct answer.
Cite relevant legal authorities (statutes, cases, doctrines).

Respond in JSON:
{
  "position": "{position}",
  "argument": "Your argument here...",
  "citations": ["Citation 1", "Citation 2"]
}
```

**Debater Rebuttal (MCQ):**
```
Your previous argument: {my_previous_argument}

Opponent's argument: {opponent_argument}

Your task:
1. Identify weaknesses in opponent's argument
2. Reinforce why your position ({position}) is stronger

Respond in JSON:
{
  "rebuttal": "Your rebuttal here...",
  "counterarguments": ["Point 1", "Point 2"],
  "citations": [...]
}
```

**Judge (MCQ):**
```
You are an impartial legal judge reviewing a debate.

Question: {question}
Alternatives: {alternatives}

Debater X (defending {pos_x}):
  Opening: {arg_x}
  Rebuttal: {reb_x}

Debater Y (defending {pos_y}):
  Opening: {arg_y}
  Rebuttal: {reb_y}

Select the most legally sound alternative based on the arguments presented.

Respond in JSON:
{
  "decision": "A/B/C/D",
  "rationale": "Brief explanation..."
}
```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
api:
  groq_api_key: ${GROQ_API_KEY}
  model: "llama-3.1-70b-versatile"
  max_tokens:
    debater_opening: 500
    debater_rebuttal: 400
    judge: 400
  temperature: 0.0  # Deterministic

datasets:
  bar_exam_qa:
    source: "TIGER-Lab/Bar_Exam_QA"
    sample_size: 200
  oab:
    source: "local"  # Download separately
    sample_size: 105

experiment:
  num_rounds: 2
  save_all_intermediates: true  # Save Round 1 + 2 outputs

output:
  results_dir: "results/"
  logs_dir: "logs/"
```

---

## ✅ Success Criteria (End of Week 1)

**Must Have:**
- [ ] B1 (single-agent baseline) working
- [ ] MAD-MCQ (2 rounds, adversarial) working
- [ ] MAD-OpenEnded (2 rounds, adversarial) working
- [ ] All outputs in valid JSON
- [ ] Tested on 50+ MCQ, 20+ open-ended

**Nice to Have:**
- [ ] B2, B3 baselines working
- [ ] Citation extraction basic implementation
- [ ] No critical bugs

---

## 📦 Week 1 Deliverables

1. **Code:** `src/` fully implemented
2. **Results:** JSON files (50 MCQ + 20 open-ended)
3. **Logs:** API calls, errors, timing
4. **Documentation:** README with setup + usage
5. **Status Report:** What works, blockers, next steps

---

## 🐛 Anticipated Challenges & Mitigation

| Challenge | Mitigation |
|-----------|----------|
| Groq rate limits | Add exponential backoff + sleep |
| JSON parsing errors | Retry with clarified prompt if malformed |
| Debaters agree on wrong answer | Log these cases, analyze later |
| Context too long (open-ended) | Truncate if needed, monitor token usage |
| Debate becomes circular | Prompt engineering: "avoid repetition" |

---

## 🎯 Next Steps (Week 2)

Once Week 1 complete:
- Run full experiments (200 Bar_Exam_QA, 105 OAB)
- Compute evaluation metrics
- Human validation (30-50 examples)
- Analysis for paper

---

**Owner:** Eryclis Silva
**Start:** Week 1, Day 1
**Ready to code:** ✅

---

## 📝 Notes for Future (Phase 2/ACL)

**Innovations to Explore:**
1. **Incremental Debate (Forgetting):** Round 2 replaces Round 1 for judge (reduce context)
2. **Structured Format:** Enforce IRAC JSON structure in outputs
3. **Web-RAG Integration:** Add retrieval component if Phase 1 successful

**Decision:** Implement after Phase 1 validates core MAD approach.
