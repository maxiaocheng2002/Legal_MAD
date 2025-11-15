# Critical Issues Identified in Original Proposal

**Analysis Date:** November 12, 2024
**Analyst:** Research Team + Claude Code Analysis
**Status:** 🚨 Requires Immediate Attention

This document summarizes the critical gaps, weaknesses, and concerns identified in the original MAD Legal Reasoning proposal that must be addressed before implementation.

---

## 🔴 CRITICAL (Blockers - Must Resolve Immediately)

### 1. Data Access and Licensing

**Issue:** Unclear whether we can legally access and use exam questions

**Specific Concerns:**
- ❌ MBE (Multistate Bar Exam) questions are **copyrighted by NCBE**
  - No discussion of licensing in proposal
  - May require expensive license or be unavailable for research
  - **Risk Level:** HIGH - Could block entire US Bar component

- ❓ Brazilian OAB exam question availability unknown
  - Typically administered by FGV - are past exams public?
  - Need to verify Portuguese copyright law
  - **Risk Level:** MEDIUM

- ❓ Brazilian Magistrate exam availability unknown
  - Varies by state - need to survey which states release questions
  - **Risk Level:** MEDIUM

**Action Required:**
1. **Week 1:** Contact NCBE about research license for MBE
2. **Week 1:** Investigate FGV OAB question repository
3. **Week 1:** Survey state magistrate exam availability
4. **Backup Plan:** If licensing fails, use publicly available practice questions or create synthetic questions based on official formats

**Owner:** [Assign team member]

---

### 2. Experimental Design Underspecification

**Issue:** Cannot implement system without resolving ambiguities

**Critical Missing Parameters:**

| Parameter | Current State | Impact | Urgency |
|-----------|---------------|---------|---------|
| **Number of debaters** | Not specified | Affects cost, feasibility, architecture | 🔴 Critical |
| **Debate rounds** | "Will vary" but no range given | Cannot estimate compute cost | 🔴 Critical |
| **Token budgets** | "Debaters submit tokens" (no number) | Cannot estimate API costs | 🔴 Critical |
| **Position assignment mechanism** | Not specified | Affects what we're actually testing | 🔴 Critical |
| **Judge decision procedure** | "Selects final answer" (how?) | Core architecture unclear | 🔴 Critical |

**Action Required:**
- **Week 1-2:** Team meeting to decide on all parameters (see EXPERIMENTAL_DESIGN.md for options)
- Document decisions with rationale
- Get team consensus before proceeding

**Owner:** Eryclis + Full Team

---

### 3. Insufficient Baselines

**Issue:** Cannot claim MAD improvement without proper comparison points

**Missing Baselines:**

| Missing Baseline | Why Critical | Implementation Effort |
|------------------|-------------|----------------------|
| **Chain-of-Thought (CoT)** | Standard reasoning baseline | Low (prompt engineering) |
| **Self-Refinement** | Tests if iteration alone helps | Medium (requires iteration loop) |
| **Retrieval-only** | Isolates RAG contribution | Low (single-agent + RAG) |

**Current Baselines:**
- ✅ Single-agent (no debate)
- ✅ Self-consistency (mentioned)

**Why This Matters:**
- Reviewers will ask: "Is improvement from debate or just from iteration?"
- Without self-refinement baseline, we can't claim adversarial structure matters
- Without CoT, we can't compare to standard reasoning approaches

**Action Required:**
- **Week 2:** Add baselines to experimental plan
- **Week 3:** Implement baseline protocols
- Budget token costs for fair comparison (self-refinement should use similar tokens as MAD)

**Owner:** [Assign team member]

---

### 4. Open-Ended Question Evaluation Unspecified

**Issue:** No clear protocol for evaluating essay/open-ended responses

**Problems:**

1. **"Official grading rubrics"** mentioned but:
   - How are rubrics adapted for automated assessment?
   - What's the validation procedure?
   - No sample rubric provided

2. **LLM-as-Judge** implied but not specified:
   - Which model serves as judge?
   - What prompts are used?
   - How is circularity avoided (using LLMs to judge LLM outputs)?
   - No validation against human judgments discussed

3. **Human evaluation** mentioned but:
   - Sample size: "a subset" - how large?
   - Annotator qualifications not specified
   - No annotation protocol provided
   - Inter-rater agreement method wrong (Cohen's κ requires 2 raters, but proposal mentions 3)

**Impact:**
- ~40-50% of evaluation depends on open-ended questions
- Without clear protocol, results won't be credible
- Reviewers will reject paper if evaluation is ad-hoc

**Action Required:**
- **Week 2:** Design complete evaluation protocol:
  1. Define LLM-as-judge procedure with example prompts
  2. Design human evaluation protocol (rubric, sample size, qualifications)
  3. Plan validation: correlate LLM-judge with human expert scores
- **Week 3:** Pilot evaluation protocol on 20 examples
- See EXPERIMENTAL_DESIGN.md Section 7.2 for detailed protocol

**Owner:** [Assign team member]

---

### 5. RAG Implementation Completely Unspecified

**Issue:** Core component of system has no implementation details

**Missing Details:**

1. **Corpus Construction:**
   - What sources will be indexed?
   - How will content be collected? (web scraping? APIs? manual?)
   - What's the indexing granularity? (document? paragraph? sentence?)
   - Estimated corpus size?

2. **Retrieval Model:**
   - Dense (e.g., DPR)? Sparse (BM25)? Hybrid?
   - Which embedding model for multilingual (English + Portuguese)?
   - Top-k value?

3. **Query Formulation:**
   - Do we pass question directly or do query expansion?
   - How does "adaptive retrieval" work when debaters "request one new snippet"?

4. **Quality Assurance:**
   - How do we ensure retrieval doesn't hallucinate?
   - How do we validate retrieval quality before using in experiments?

**Impact:**
- RAG quality confounds debate quality
- If retrieval is poor, MAD won't help (garbage in, garbage out)
- Corpus construction could take weeks-months if not planned

**Action Required:**
- **Week 2:** Design complete RAG architecture (see EXPERIMENTAL_DESIGN.md Section 6)
- **Week 3-6:** Implement and validate RAG pipeline BEFORE building debate system
- **Critical:** Test single-agent + RAG baseline first to ensure RAG actually helps

**Owner:** [Assign team member]

---

## 🟡 HIGH PRIORITY (Significantly Weakens Paper)

### 6. Hypotheses Lack Operational Precision

**Issue:** H1, H2, H3 are conceptually clear but not testable as stated

**Problems:**

**H1: "MAD improves citation quality"**
- No definition of "citation quality" measurement
- No prediction of effect size (how much improvement?)
- Mentions "correctness" and "coverage" but no formulas

**H2: "Disproportionately corrects high-confidence errors"**
- What does "disproportionately" mean quantitatively?
- How is confidence measured?
- What's the comparison group?

**H3: "Diminishing returns after 2-3 rounds"**
- What defines "diminishing returns"? (< X% improvement? Non-significant? Cost-benefit threshold?)
- Why 2-3 specifically?

**Impact:**
- Cannot determine if hypotheses are supported without precise definitions
- Reviewers will critique lack of specificity
- Risk of p-hacking or post-hoc hypothesis adjustment

**Action Required:**
- **Week 2:** Operationalize all hypotheses with:
  - Precise metrics and formulas
  - Numerical predictions (effect sizes)
  - Statistical tests to be used
  - Success criteria defined a priori
- See EXPERIMENTAL_DESIGN.md Section 3 for detailed specifications

**Owner:** [Assign team member]

---

### 7. No Computational Cost Analysis

**Issue:** No budget or feasibility analysis for proposed experiments

**Risks:**

1. **Underestimating Costs:**
   - 4 datasets × multiple configs × multiple rounds = thousands of experiment runs
   - Multi-round debate = high token consumption
   - Could easily exceed $10,000-$50,000 without planning

2. **Timeline Infeasibility:**
   - If experiments take too long to run, won't meet August deadline
   - No discussion of parallelization or optimization

3. **No Cost-Benefit Discussion:**
   - Reviewers will ask: "Is MAD worth the computational cost?"
   - Need to show when MAD provides value relative to cost

**Action Required:**
- **Week 2:** Calculate token cost estimates:
  - Per question for each configuration
  - Total for full experimental suite
  - Compare to available budget
- **Week 2:** Analyze inference time and timeline feasibility
- **During Experiments:** Track actual costs and include cost-benefit analysis in paper

**Owner:** [Assign team member]

---

### 8. Evaluation Relies Heavily on Unvalidated Automated Metrics

**Issue:** Citation quality and reasoning consistency use automated methods without validation

**Specific Concerns:**

**Citation Quality:**
- Proposal says "human annotators on 100 sampled cases" for correctness
- But coverage is automated based on "gold-labeled authoritative sources"
- **Problem:** Proposal doesn't mention that datasets HAVE gold authority labels - do they exist?
- If gold labels don't exist, coverage metric is impossible

**Reasoning Consistency:**
- Uses "semantic similarity" and "logical entailment detection"
- **Problem:** Which models? What thresholds?
- Not validated against human judgment of consistency

**Impact:**
- Automated metrics might not correlate with actual legal reasoning quality
- Reviewers will question validity if metrics are unvalidated

**Action Required:**
- **Week 3:** For citation evaluation:
  - Verify if gold authority labels exist in datasets
  - If not, decide: human annotation or drop coverage metric
- **Week 3:** For reasoning consistency:
  - Specify exact implementation (models, thresholds)
  - Validate on 50 examples with human judgments
- **Week 4:** Report correlation between automated and human metrics

**Owner:** [Assign team member]

---

### 9. Statistical Rigor Issues

**Issue:** Multiple comparisons and statistical testing underspecified

**Problems:**

1. **Multiple Comparisons:**
   - Testing 3 hypotheses + multiple ablations = many statistical tests
   - No mention of correction procedures (Bonferroni, FDR)
   - Risk of false positives

2. **No Power Analysis:**
   - How many questions needed to detect meaningful effects?
   - Especially critical for subgroup analyses (by domain, difficulty, etc.)

3. **No Effect Size Reporting:**
   - Proposal mentions "mean ± 95% CI" but not effect sizes
   - Reviewers need Cohen's d, odds ratios, etc. to judge practical significance

**Action Required:**
- **Week 2:** Statistical analysis plan:
  - Specify correction procedures
  - Conduct power analysis for primary hypotheses
  - Commit to reporting effect sizes for all comparisons
- See EXPERIMENTAL_DESIGN.md Section 8 for complete plan

**Owner:** [Assign team member]

---

## 🟢 MEDIUM PRIORITY (Good to Address)

### 10. Limited Engagement with Recent Literature

**Issue:** Missing key related work

**Gaps:**

**Multi-Agent Debate:**
- Missing recent 2024 debate papers (Khan et al., Chen et al., others)
- Limited differentiation from existing MAD frameworks

**Legal NLP:**
- No mention of Brazilian Portuguese legal NLP (critical for OAB/Magistrate datasets)
- Missing legal argument mining literature (relevant to debate structure)
- No comparison to recent legal reasoning systems

**Evaluation:**
- No mention of recent LLM-as-judge validation work

**Impact:**
- Reviewers may critique lack of awareness
- May miss insights that could improve design
- Novelty claims may be overstated

**Action Required:**
- **Week 2-3:** Comprehensive literature review:
  - Survey 2024 multi-agent papers
  - Survey Brazilian legal NLP
  - Survey legal reasoning and argument mining
- **Week 4:** Update related work section in proposal
- Create annotated bibliography in docs/related_work/

**Owner:** [Assign team member]

---

### 11. Cross-Examiner Role Completely Unspecified

**Issue:** Marked as "optional" but no implementation details if included

**Decision Required:**
- Include or drop cross-examiner?
- If include: How does it work? Who controls it? When does it intervene?
- If drop: Remove from proposal to avoid confusion

**Action Required:**
- **Week 2:** Team decision: include or drop
- If include: Specify in EXPERIMENTAL_DESIGN.md
- If drop: Remove from paper and focus on core MAD architecture

**Owner:** [Assign team member]

---

### 12. Failure Mode Analysis Missing

**Issue:** No discussion of when MAD might fail or perform worse

**Why This Matters:**
- Responsible research acknowledges limitations
- Understanding failure modes guides future improvements
- Reviewers appreciate nuanced analysis

**Examples of Potential Failures:**
- Both debaters converge on wrong answer (groupthink)
- Retrieved evidence is insufficient/misleading
- Debate becomes circular or unproductive
- Judge misaggregates good arguments

**Action Required:**
- **During Pilot (Week 10):** Manually identify failure patterns
- **During Experiments:** Collect questions where MAD < Single-Agent
- **Week 28:** Create failure mode taxonomy for paper
- See EXPERIMENTAL_DESIGN.md Section 8 for analysis plan

**Owner:** [Assign team member]

---

## 🔵 LOWER PRIORITY (Nice to Have)

### 13. Reproducibility Details

**Missing:**
- Model temperature settings
- Random seeds
- Number of runs per configuration
- Model version pinning

**Action Required:**
- Document in implementation code
- Include in methods section of paper

---

### 14. IRB and Data Statements

**Missing:**
- IRB determination (do we need approval for human annotation?)
- Data availability statements
- Privacy considerations (especially for Brazilian exam questions with case scenarios)

**Action Required:**
- **Week 1:** Consult UIUC IRB office
- **During Writing:** Prepare data availability and ethics statements

---

## Summary: Critical Path to Success

### Week 1-2 (Immediate Actions)
1. ✅ Review proposal and identify gaps (DONE)
2. 🚧 Verify data access (IN PROGRESS)
3. 🚧 Finalize experimental design decisions
4. 🚧 Create detailed specifications for H1, H2, H3

### Week 3-4 (Foundation)
5. ⬜ Design and document evaluation protocols
6. ⬜ Design RAG architecture
7. ⬜ Implement all baseline systems
8. ⬜ Literature review and related work update

### Week 5-7 (Infrastructure)
9. ⬜ Build and validate RAG pipeline
10. ⬜ Implement core MAD system
11. ⬜ Create evaluation harness

### Week 8-10 (Validation)
12. ⬜ Pilot study on 50 MMLU-Pro questions
13. ⬜ Validate metrics against human judgments
14. ⬜ Identify and fix failure modes

### Week 11+ (Execution)
15. ⬜ Run full experimental suite
16. ⬜ Human evaluation
17. ⬜ Analysis and writing

---

## Risk Mitigation

**If data access fails:**
- Fall back to MMLU-Pro + MEE (publicly available)
- Create synthetic questions based on official formats
- Scope to "feasibility study" rather than comprehensive evaluation

**If computational costs exceed budget:**
- Prioritize H1, H2, H3 core tests
- Reduce dataset size (stratified sample)
- Drop some architectural ablations

**If timeline is tight:**
- Focus on MMLU-Pro + US Bar (English only)
- Defer Brazilian datasets to extension paper
- Reduce human evaluation scope

---

**Document Owner:** Eryclis Silva
**Last Updated:** November 12, 2024
**Next Review:** After Week 2 team meeting
