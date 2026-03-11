# Mechanistic Interpretability of Tool Selection in LLMs

Artificial Intelligence research | Generated: 2026-03-11

## Overview

This project investigates **how LLMs decide which tool to call** — a critical but understudied question in AI agent design. Using GPT-4.1-mini for behavioral experiments and GPT-2-small for mechanistic probing, we characterize the computational basis of tool selection across four complementary experiments.

## Key Findings

1. **Semantic similarity partially drives selection (40% top-1 accuracy vs. 25% chance)** — but cosine similarity alone is insufficient; internal computation is more complex than simple nearest-neighbor matching.

2. **Strong position-0 bias in behavioral experiments (58% selection rate)**, but controlled rotation experiments show **90% selection stability** — content dominates over position when the semantic signal is clear.

3. **Adversarial description swapping causes selection change in 2/3 cases** — confirming descriptions causally affect selection, while tool *names* can partially resist manipulation.

4. **Tool category is encoded from layer 1 (100% probe accuracy)** across all GPT-2-small layers, while information about *which specific tool is semantically correct* emerges progressively from **35% at layer 1 to 68% at layer 9**, revealing a distributed semantic matching computation in middle-to-late layers.

5. **Last-token representation provides a strong selection signal from early layers (73-81%)**, suggesting the final generation token acts as an information aggregation site for the selection decision.

## How to Reproduce

### Setup

```bash
source .venv/bin/activate
# Or: uv venv && source .venv/bin/activate
# uv pip install numpy pandas matplotlib scipy scikit-learn openai sentence-transformers transformers torch
```

### Run Experiments

```bash
export OPENAI_API_KEY="your-openrouter-key"

python src/create_dataset.py          # Generate datasets
python src/llm_experiments.py          # LLM behavioral experiments (API)
python src/semantic_analysis.py        # Embedding-based analysis
python src/positional_analysis.py      # Positional bias statistics
python src/probing_analysis_v2.py      # Layer-wise probing (GPT-2)
python src/generate_figures.py         # All figures
```

## File Structure

```
.
├── REPORT.md              ← Full research report with all results
├── planning.md            ← Research design and motivation
├── src/
│   ├── create_dataset.py          ← Dataset generation (5 tool categories)
│   ├── llm_experiments.py         ← GPT-4.1-mini tool selection experiments
│   ├── semantic_analysis.py       ← Embedding cosine similarity analysis
│   ├── positional_analysis.py     ← Chi-square positional bias tests
│   ├── probing_analysis_v2.py     ← GPT-2-small layer probing (2 tasks)
│   └── generate_figures.py        ← 7 publication-ready figures
├── datasets/experiment_datasets/  ← Generated experiment data
├── results/                       ← All results (JSON, CSV)
├── figures/                       ← Figures (fig1-fig7 PNG)
├── papers/                        ← Downloaded research papers
└── literature_review.md           ← Pre-gathered literature review
```

## Background

This research addresses the gap between:
- **Healy et al. (2026)**: Shows final-layer reps encode tool selection info but doesn't identify circuits
- **BiasBusters (2025)**: Shows semantic alignment predicts selection but doesn't explain the mechanism

We bridge these by identifying *where* in the transformer the selection computation occurs and providing causal evidence that tool descriptions (not just names) drive selection.

See [REPORT.md](REPORT.md) for full methodology, result tables, and discussion.
