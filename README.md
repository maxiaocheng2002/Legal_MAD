# Multi-Agent Debate for Legal Reasoning

[![ACL 2025](https://img.shields.io/badge/ACL-2025-blue.svg)](https://2025.aclweb.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the implementation and experiments for **Multi-Agent Debate (MAD) for Legal Reasoning**, a research project investigating how structured adversarial argumentation improves AI performance on legal question-answering tasks.

**Status:** 🚧 In Development - Phase 1: Experimental Design Specification

### Key Research Questions

1. Does multi-agent debate improve citation quality through adversarial testing? (H1)
2. Does MAD disproportionately correct high-confidence errors? (H2)
3. Do debate benefits show diminishing returns after 2-3 rounds? (H3)

## Project Structure

```
MAD_legal/
├── docs/                          # Documentation
│   ├── proposals/                 # Research proposals and plans
│   ├── design/                    # Experimental design specifications
│   ├── evaluation/                # Evaluation protocols and rubrics
│   └── related_work/              # Literature reviews and summaries
├── src/                           # Source code
│   ├── agents/                    # Debater and Judge implementations
│   ├── retrieval/                 # RAG pipeline components
│   ├── evaluation/                # Evaluation metrics and protocols
│   └── utils/                     # Utility functions
├── data/                          # Datasets
│   ├── raw/                       # Original exam questions
│   ├── processed/                 # Preprocessed and standardized data
│   └── corpora/                   # Legal knowledge corpora for RAG
├── experiments/                   # Experiment configurations
│   ├── configs/                   # Configuration files
│   └── scripts/                   # Experiment execution scripts
├── notebooks/                     # Jupyter notebooks for analysis
├── results/                       # Experimental results and outputs
└── tests/                         # Unit and integration tests
```

## Research Team

- **Eryclis Silva** (Lead) - University of Illinois at Urbana-Champaign
- Sean Liao - UIUC
- Jared Buabeng - UIUC
- James Han - UIUC
- Arihant Singh - UIUC
- Xiaocheng Ma - UIUC

**Correspondence:** ersilva2@illinois.edu

## Datasets

We evaluate on four legal examination datasets:

1. **MMLU-Pro (Law)** - Enhanced MMLU with 10 answer choices
2. **US Bar Examination** - Multistate Bar Exam (MBE) + Multistate Essay Exam (MEE)
3. **Brazilian Bar Examination (OAB)** - First and Second Phase questions
4. **Brazilian National Magistrate Examination** - Highest-level judicial competency assessment

## Current Phase: Experimental Design Specification

### Immediate Tasks
- [ ] Define precise experimental parameters (debaters, rounds, tokens)
- [ ] Specify all baseline systems (CoT, self-refinement, etc.)
- [ ] Create operational definitions for H1, H2, H3
- [ ] Verify data access and licensing
- [ ] Design RAG pipeline architecture
- [ ] Develop evaluation protocols

See [docs/design/](docs/design/) for detailed specifications.

## Installation

*Coming soon - after Phase 1 completion*

## Usage

*Coming soon - pilot study implementation in progress*

## Citation

```bibtex
@article{silva2025mad,
  title={Multi-Agent Debate for Legal Reasoning},
  author={Silva, Eryclis and Liao, Sean and Buabeng, Jared and Han, James and Singh, Arihant and Ma, Xiaocheng},
  year={2025},
  note={In preparation for ACL 2025}
}
```

## License

*To be determined based on dataset licensing requirements*

## Acknowledgments

This research is conducted at the University of Illinois at Urbana-Champaign.

---

**Project Start Date:** November 2024
**Target Submission:** ACL 2025 (August deadline)
