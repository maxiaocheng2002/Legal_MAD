# Experimental Design Specification

**Document Status:** ✅ Final - Approved
**Last Updated:** November 12, 2024
**Version:** 2.0

---

## 🎯 Research Questions

**RQ1:** Does MAD improve legal reasoning quality beyond simple accuracy?
- Focus: Citation precision, argumentative coherence

**RQ2:** When does adversarial debate provide the most benefit?
- Focus: Error analysis by confidence, question types

**RQ3:** How does debate improve legal arguments?
- Focus: Argument structure (IRAC), counter-argument consideration

---

## 1. Datasets

### 1.1 Selected Datasets

| Dataset | Type | Size | Language | Purpose |
|---------|------|------|----------|---------|
| **Bar_Exam_QA** | MCQ | 1195 | English | Validation + Citation analysis (has gold passages) |
| **OAB 2nd Phase** | Open-ended | 105 → 300+ | Portuguese | Core contribution (complex reasoning with rubricas) |
| **Magistrate Exam** | MCQ/Open | TBD | Portuguese | Novel benchmark (conditional on Fase 1 success) |

### 1.2 Dataset Strategy

**Phase 1 (Weeks 1-3): Quick Validation**
- Bar_Exam_QA (sample 200-300 questions)
- OAB existing (105 questions)
- **Goal:** Prove MAD works, deliver paper for disciplina

**Phase 2 (Weeks 4-8): ACL Extension**
- Bar_Exam_QA (full 1195)
- OAB expanded (collect + preprocess ~30 editions → 300-400 questions)
- Magistrate (create new benchmark if Fase 1 successful)

### 1.3 Why These Datasets?

- **Uniformity:** All from official legal examinations (not synthetic)
- **Complexity:** Open-ended questions with official rubricas (not just MCQ)
- **Multilingual:** English + Portuguese (underrepresented in NLP)
- **Cross-legal:** Common law (US) + Civil law (Brazil)
- **Contribution:** Expanded OAB + new Magistrate benchmarks for community

---

## 2. Core Architecture

### 2.1 MAD System Components

**Debaters:** 2 agents (for MCQ and open-ended)
- Defend different positions (MCQ: different answer choices)
- Generate arguments with citations

**Judge:** 1 agent
- Synthesizes debate
- Selects final answer with rationale

**Cross-Examiner:** EXCLUDED (simplify for timeline)

### 2.2 Debate Structure

**Round 0:** Baseline (single-agent, no debate)
**Round 1:** Opening statements (both debaters argue simultaneously)
**Round 2:** Rebuttal (optional, if time permits)

**Focus:** Test 0 vs. 1 vs. 2 rounds (not 3+ due to diminishing returns expectation)

### 2.3 Token Budgets

| Role | Tokens per Turn |
|------|----------------|
| Debater (Opening) | 400-500 |
| Debater (Rebuttal) | 300-400 |
| Judge | 300-500 |

**Estimated cost per question:**
- 2 debaters × 1 round + judge ≈ 1,200 output tokens
- Total (input + output) ≈ 2,500-3,000 tokens

### 2.4 Position Assignment (MCQ)

**Method:** Random assignment
- Randomly assign answer choices to debaters
- Tests forced alternative consideration
- Simpler, more unbiased than model-preference

---

## 3. Models and APIs

### 3.1 Primary Model

**Groq API - Llama 3.1 70B**
- Free tier (14,400 requests/day)
- Fast inference
- Open-source (reproducible)

**Rationale:**
- Zero cost for validation phase
- Sufficient performance for legal reasoning
- Community values open-source models

### 3.2 Alternative/Comparison (if budget allows)

**GPT-4o-mini** (for ablation comparison in ACL version)
- Show MAD generalizes across models
- Relatively affordable ($0.15/$0.60 per 1M tokens)

---

## 4. Baseline Systems

### 4.1 Required Baselines

| Baseline | Description |
|----------|-------------|
| **B1: Single-Agent (Zero-shot)** | Standard prompting, no debate, no retrieval |
| **B2: Chain-of-Thought (CoT)** | Explicit step-by-step reasoning prompt |
| **B3: Self-Consistency** | Sample 3-5 responses, majority vote |

### 4.2 Optional Baseline (if time permits)

| Baseline | Description |
|----------|-------------|
| **B4: Single-Agent + Web-RAG** | Augmented with web search (e.g., Perplexity API, Tavily) |

**Web-RAG Notes:**
- Can be applied to BOTH baseline and MAD
- Tests if retrieval + debate compound benefits
- Phase 2 extension if Phase 1 shows strong results without RAG

---

## 5. Experimental Configurations

### 5.1 Core Experiments (Phase 1: Weeks 1-3)

**Experiment 1: Does MAD improve legal reasoning?**
- Compare: B1 (Single-Agent) vs. MAD (2 debaters, 1 round)
- Datasets: Bar_Exam_QA (sample), OAB (105 existing)
- Metrics: Accuracy, Citation F1, Argument Quality
- **Deliverable:** Paper for disciplina

**Experiment 2: Citation quality analysis (RQ1)**
- Deep dive into citation metrics
- Manual validation (50-100 examples)

**Experiment 3: Confidence-stratified error analysis (RQ2)**
- Measure confidence via consistency (3 runs)
- Stratify errors by quartile
- Test if MAD helps most on high-confidence errors

### 5.2 Extended Experiments (Phase 2: ACL)

**Experiment 4: Rounds ablation**
- Test 0 vs. 1 vs. 2 rounds
- Analyze cost-benefit trade-off

**Experiment 5: Web-RAG augmentation**
- Test: Baseline + Web-RAG vs. MAD + Web-RAG
- See if retrieval + debate compound

**Experiment 6: Model comparison**
- Llama 3.1 70B vs. GPT-4o-mini
- Show MAD generalizes across models

**Experiment 7: Full datasets**
- Bar_Exam_QA full (1195)
- OAB expanded (300+)
- Magistrate (if created)

---

## 6. Evaluation Metrics

**See:** [EVALUATION_PROTOCOL.md](./EVALUATION_PROTOCOL.md) for complete details.

**Summary:**

1. **Accuracy** (MCQ baseline)
2. **Citation Quality** (Precision, Recall, F1, Hallucination Rate)
3. **Argument Quality** (ROUGE-L, BERTScore, Argument Coverage, IRAC Structure)
4. **Confidence Measurement** (Consistency-based, 3 runs)
5. **Human Validation** (50-100 examples, inter-annotator agreement)

---

## 7. Analysis Plan

### 7.1 Main Results (RQ1)

**Tables:**
- Accuracy: Baseline vs. MAD
- Citation F1: Baseline vs. MAD
- Argument Quality: Baseline vs. MAD

**Statistical Tests:**
- McNemar (accuracy)
- Wilcoxon (continuous metrics)
- Effect sizes (Cohen's d/h)
- 95% confidence intervals

### 7.2 Error Analysis (RQ2)

**Stratify results by confidence quartiles:**
- Q1 (low), Q2, Q3, Q4 (high)
- Test: Is MAD improvement in Q4 > Q1-Q3?

### 7.3 Argument Structure Analysis (RQ3)

**IRAC component scores:**
- Issue, Rule, Application, Conclusion, Counter-argument
- Hypothesis: MAD especially improves counter-argument consideration

### 7.4 Qualitative Analysis

**Failure modes:**
- When does MAD not help or hurt?
- Groupthink (both debaters wrong)
- Circular debates

---

## 8. Timeline

### Phase 1: Quick Validation (Weeks 1-3)

**Week 1:**
- [ ] Setup: Groq API, datasets (Bar_Exam_QA, OAB 105)
- [ ] Implement: Baseline (B1, B2, B3)
- [ ] Implement: MAD system (2 debaters, 1 round, judge)

**Week 2:**
- [ ] Run experiments: Baseline vs. MAD
- [ ] Compute metrics: Accuracy, Citation F1, Argument Quality
- [ ] Human validation: 30-50 examples

**Week 3:**
- [ ] Analysis: RQ1, RQ2, RQ3
- [ ] Write paper for disciplina
- [ ] **Deliverable:** Paper submission

### Phase 2: ACL Extension (Weeks 4-8)

**Week 4-5:**
- [ ] Expand OAB dataset (collect + preprocess ~30 editions)
- [ ] Decide: Create Magistrate benchmark? (if Phase 1 strong)

**Week 6:**
- [ ] Run full experiments (all datasets, rounds ablation)
- [ ] Test Web-RAG augmentation (optional)
- [ ] Model comparison: Llama vs. GPT-4o-mini (optional)

**Week 7:**
- [ ] Human validation: 100 examples total
- [ ] Inter-annotator agreement analysis
- [ ] Comprehensive analysis (all RQs + failure modes)

**Week 8:**
- [ ] Write ACL paper
- [ ] **Deliverable:** ACL submission

---

## 9. Paper Structure (ACL Target)

**Title:** Multi-Agent Debate Improves Legal Reasoning: Evidence from Citation Quality and Argumentative Coherence

**Abstract:** (Problem → Approach → Contributions)

**1. Introduction**
- Legal reasoning requires selecting among competing interpretations
- Single agents exhibit premature commitment, weak citation grounding
- **Our approach:** Multi-agent debate for systematic exploration
- **Contributions:**
  1. MAD significantly improves citation quality and argument coherence (not just accuracy)
  2. Expanded Brazilian legal QA benchmarks (OAB, Magistrate)
  3. Analysis revealing MAD especially benefits high-confidence errors

**2. Related Work**
- Multi-agent collaboration (cooperative vs. adversarial)
- Legal reasoning benchmarks
- Citation quality in legal NLP

**3. Method**
- MAD architecture (debaters, judge)
- Datasets (Bar_Exam_QA, OAB, Magistrate)
- Baselines

**4. Evaluation**
- Metrics (accuracy, citation quality, argument quality)
- Human validation protocol

**5. Results**
- 5.1 Main Results (RQ1)
- 5.2 Citation Quality Analysis
- 5.3 Error Analysis by Confidence (RQ2)
- 5.4 Argument Structure Analysis (RQ3)
- 5.5 Human Validation

**6. Discussion**
- When and why MAD helps
- Failure modes
- Implications for legal AI

**7. Conclusion**

**8. Limitations & Future Work**

---

## 10. Key Decisions Summary

### ✅ Finalized Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Datasets** | Bar_Exam_QA + OAB + Magistrate (conditional) | Official exams, multilingual, novel contributions |
| **RAG** | Zero-shot primary, Web-RAG optional extension | Simplify for timeline, focus on reasoning |
| **Models** | Groq Llama 3.1 70B (free) | Zero cost, open-source, sufficient performance |
| **Debaters** | 2 agents | Simple, clear adversarial structure |
| **Rounds** | Test 0 vs. 1 vs. 2 | Sufficient to show benefit + diminishing returns |
| **Baselines** | Single-Agent, CoT, Self-Consistency | Standard, defensible |
| **Hypotheses** | Research Questions (RQ1, RQ2, RQ3) | Modern ML/NLP style, not formal hypotheses |
| **Validation** | Human (50-100 examples, 3 annotators) | Scientifically rigorous, feasible |

---

## 11. Risk Mitigation

**Risk 1: MAD doesn't improve performance**
- **Mitigation:** If no improvement, pivot to "understanding when/why debate fails" (still publishable analysis)

**Risk 2: Timeline too tight (3 weeks)**
- **Mitigation:** Phase 1 delivers disciplina paper (minimal viable). ACL adds depth.

**Risk 3: Data collection for OAB/Magistrate takes too long**
- **Mitigation:** Bar_Exam_QA alone is sufficient for validation. Brazilian datasets are "bonus" contributions.

**Risk 4: Groq rate limits**
- **Mitigation:** Together.ai backup ($25 free credits), stagger experiments

---

## 12. Success Criteria

### Phase 1 (Disciplina Paper)

**Minimum Viable:**
- ✅ MAD shows statistically significant improvement on at least 1 metric (accuracy OR citation F1)
- ✅ Results on 2 datasets (Bar_Exam_QA + OAB)
- ✅ Basic human validation (30 examples)

**Strong Success:**
- ✅ MAD improves both accuracy AND citation quality
- ✅ Clear evidence for RQ1 and RQ2

### Phase 2 (ACL Paper)

**Minimum Viable:**
- ✅ All Phase 1 criteria met
- ✅ Robust human validation (100 examples, κ ≥ 0.67)
- ✅ Comprehensive analysis (all RQs answered)

**Strong Success:**
- ✅ All above + expanded OAB dataset contributed to community
- ✅ Novel Magistrate benchmark created
- ✅ Web-RAG ablation showing compound benefits

---

**Document Owner:** Eryclis Silva
**Contributors:** Research Team
**Next Document:** IMPLEMENTATION_PLAN.md (technical implementation guide)
