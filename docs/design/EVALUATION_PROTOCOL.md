# Evaluation Protocol for MAD Legal Reasoning

**Version:** 1.0
**Date:** November 12, 2024
**Status:** ✅ Final - Ready for Implementation

---

## 🎯 Research Questions

Our evaluation addresses three core questions:

1. **RQ1:** Does MAD improve legal reasoning quality beyond simple accuracy?
2. **RQ2:** When does adversarial debate provide the most benefit?
3. **RQ3:** How does debate improve legal arguments?

---

## 📊 Evaluation Metrics

### 1. Accuracy (Baseline - MCQ Only)

**Applies to:** Bar_Exam_QA
**Definition:** % of questions answered correctly
**Purpose:** Baseline comparison, necessary but insufficient

---

### 2. Citation Quality (CORE - Addresses RQ1)

**Applies to:** Both datasets
**Multi-dimensional evaluation:**

#### **2.1 Citation Precision**
- What % of model's citations are correct?
- **Bar_Exam_QA:** Match against gold passages
- **OAB:** Match against expected authorities in rubrica

#### **2.2 Citation Recall (Coverage)**
- What % of expected citations were mentioned?
- Based on gold passages (Bar_Exam_QA) or rubrica (OAB)

#### **2.3 Citation F1**
- Harmonic mean of precision and recall

#### **2.4 Hallucination Rate**
- What % of citations are completely fabricated?
- Lower is better

**Implementation Notes:**
- Use regex patterns to extract citations (Brazilian: "Art. X", "CF/88", "STF"; US: "U.S.C. §", case names)
- Fuzzy string matching for minor variations
- Build validation corpus of valid authorities

---

### 3. Argument Quality (Open-ended - OAB)

**Multi-metric approach:**

#### **3.1 ROUGE-L**
- Lexical overlap with official rubrica
- Standard baseline metric

#### **3.2 BERTScore**
- Semantic similarity with rubrica
- Uses multilingual BERT embeddings (Portuguese)

#### **3.3 Argument Coverage (Custom)**
- What % of expected legal arguments (from rubrica) appear in model answer?
- Domain-specific metric

#### **3.4 Legal Argument Structure Score (Custom)**
- Does answer follow IRAC structure?
  - **I**ssue identification
  - **R**ule statement (legal principle)
  - **A**pplication to facts
  - **C**onclusion
  - **Counter-argument** (MAD-specific advantage)
- Score: 0-5 (one point per component)

**Implementation Notes:**
- IRAC detection via heuristics (keywords, citation presence, contrastive phrases)
- Can be refined with NLP models if time permits

---

### 4. Confidence Measurement (for RQ2)

**Method: Consistency-based**

- Run same question 3 times with temperature > 0
- Confidence = % of runs with same answer
  - 3/3 identical → High confidence (1.0)
  - 2/3 identical → Medium confidence (0.67)
  - 1/3 identical → Low confidence (0.33)

**Purpose:**
- Stratify errors by confidence quartiles (Q1-Q4)
- Test if MAD disproportionately helps high-confidence errors

**Cost Note:** Triples API calls for baseline

---

## 🧑‍⚖️ Human Validation

### Sample & Annotators

- **Sample Size:** 50-100 examples (25-50 per dataset)
- **Selection:** Stratified (correct/incorrect, confidence levels, domains)
- **Annotators:**
  - Primary: Author with legal expertise
  - Secondary: 1-2 co-annotators (for reliability check)

### Annotation Dimensions

For each model output, rate:

1. **Citation Quality** (1-3 scale)
   - 3 = Excellent, 2 = Adequate, 1 = Poor

2. **Argument Quality** (1-3 scale)
   - 3 = Excellent (comprehensive, well-structured)
   - 2 = Adequate (addresses question, some gaps)
   - 1 = Poor (incoherent, missing key points)

3. **Legal Correctness** (Binary: 0/1)

4. **Counter-argument Consideration** (Binary: 0/1)
   - Specific to MAD advantage

### Inter-Annotator Agreement

- Compute Krippendorff's Alpha (handles 3+ annotators, ordinal data)
- **Target:** α ≥ 0.67 (acceptable for complex tasks)
- **Interpretation:**
  - α ≥ 0.80: Excellent
  - α ≥ 0.67: Acceptable
  - α < 0.67: Guidelines need refinement

### Validation Analysis

- **Correlation:** Compare automated metrics with human ratings
  - Citation F1 vs. Human Citation Quality
  - BERTScore vs. Human Argument Quality
- **Purpose:** Validate that automated metrics capture actual quality

---

## 📈 Analysis Plan

### Main Results (RQ1)

**Tables:**
- Accuracy: Baseline vs. MAD
- Citation F1: Baseline vs. MAD
- Argument Quality (ROUGE-L, BERTScore, Coverage): Baseline vs. MAD

**Report:**
- Mean scores with 95% confidence intervals
- Statistical significance (McNemar's test for accuracy, Wilcoxon for continuous)
- Effect sizes (Cohen's d or h)

### Error Analysis by Confidence (RQ2)

**Table:** Accuracy by confidence quartile

| Quartile | Baseline Accuracy | MAD Accuracy | Improvement |
|----------|-------------------|--------------|-------------|
| Q1 (Low) | ? | ? | ? |
| Q2 | ? | ? | ? |
| Q3 | ? | ? | ? |
| Q4 (High) | ? | ? | ? |

**Key Question:** Is improvement in Q4 significantly larger than Q1-Q3?

### Argument Structure Analysis (RQ3)

**Table:** IRAC component scores (Baseline vs. MAD)

| Component | Baseline | MAD | Improvement |
|-----------|----------|-----|-------------|
| Issue | ? | ? | ? |
| Rule | ? | ? | ? |
| Application | ? | ? | ? |
| Conclusion | ? | ? | ? |
| Counter-argument | ? | ? | ? |

**Key Finding:** MAD should especially improve counter-argument consideration

---

## 🗂️ Output Format

### Per-Question Results (JSON)

```json
{
  "question_id": "...",
  "baseline": {
    "answer": "...",
    "correct": true/false,
    "confidence": 0.0-1.0,
    "response_text": "...",
    "metrics": {
      "citation_precision": 0.0-1.0,
      "citation_recall": 0.0-1.0,
      "citation_f1": 0.0-1.0,
      "hallucination_rate": 0.0-1.0
    }
  },
  "mad": {
    "answer": "...",
    "correct": true/false,
    "response_text": "...",
    "metrics": {...},
    "debate_logs": {...}
  }
}
```

### Aggregate Results (JSON)

```json
{
  "dataset": "bar_exam_qa",
  "baseline": {
    "accuracy": 0.652,
    "citation_f1": 0.584
  },
  "mad": {
    "accuracy": 0.698,
    "citation_f1": 0.712
  },
  "comparison": {
    "accuracy_improvement": 0.046,
    "citation_f1_improvement": 0.128
  }
}
```

---

## ✅ Implementation Checklist

### Week 1-2: Setup
- [ ] Implement metric computation functions
- [ ] Create evaluation harness
- [ ] Test on small sample (10 examples)

### Week 2-3: Experiments
- [ ] Run baseline and MAD experiments
- [ ] Compute all automated metrics
- [ ] Generate results (JSON format)

### Week 3: Human Validation
- [ ] Sample 50-100 examples (stratified)
- [ ] Prepare annotation guidelines
- [ ] Recruit 1-2 co-annotators
- [ ] Conduct annotations
- [ ] Compute inter-annotator agreement
- [ ] Analyze correlation with automated metrics

### Week 3: Analysis & Reporting
- [ ] Statistical tests (significance, effect sizes, CIs)
- [ ] Generate tables and plots
- [ ] Write results section

---

## 📝 Reporting in Paper

### Results Section Structure

**5.1 Main Results**
Table with accuracy, citation F1, argument quality (both datasets)

**5.2 Citation Quality Analysis (RQ1)**
Detailed breakdown: precision, recall, F1, hallucination rate

**5.3 Error Analysis by Confidence (RQ2)**
Stratified results showing MAD helps most on high-confidence errors

**5.4 Argument Structure Analysis (RQ3)**
IRAC component analysis, emphasizing counter-argument improvement

**5.5 Human Validation**
Inter-annotator agreement + correlation with automated metrics

---

## 🎯 Key Success Metrics

**For Paper Disciplina (3 weeks):**
- ✅ Accuracy improvement (statistical significance)
- ✅ Citation F1 improvement (primary contribution)
- ✅ Basic human validation (30-50 examples)

**For ACL (extended):**
- ✅ All above metrics robust across both datasets
- ✅ Comprehensive human validation (100 examples, κ ≥ 0.67)
- ✅ Detailed analysis (confidence stratification, IRAC, failure modes)

---

**Document Owner:** Eryclis Silva
**Next Steps:** Implement evaluation harness and metrics computation
