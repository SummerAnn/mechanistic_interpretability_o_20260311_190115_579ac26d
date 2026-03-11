# Research Planning: Mechanistic Interpretability of Tool Selection in LLMs

## Motivation & Novelty Assessment

### Why This Research Matters
LLM-based agents are deployed in high-stakes settings (finance, healthcare, coding) where tool selection errors cause cascading failures. Yet we have no mechanistic understanding of *how* models decide which tool to call — behavioral benchmarks (ToolBench, Berkeley FCL) measure success/failure but cannot predict or debug failures. Understanding the circuits underlying tool selection enables targeted interventions: fixing selection bugs, making agents more reliable, and ultimately building safer autonomous systems.

### Gap in Existing Work
The literature review reveals three key gaps:
1. **No circuit-level analysis**: 2601.05214 (Healy et al.) shows final-layer representations encode tool selection information, but does not identify which circuits perform the computation or how they work.
2. **Semantic matching is correlational, not causal**: BiasBusters (2510.00307) shows semantic alignment predicts selection but cannot explain *how* the model computes that alignment internally.
3. **No decomposition of semantic matching vs. routing**: Is selection a matter of matching query to tool description semantics (reusing existing entity-matching circuits), or is there dedicated "routing" logic?

### Our Novel Contribution
This work provides the first empirical characterization of tool selection representations across three complementary lenses: (1) semantic similarity analysis comparing query-tool embedding distances against actual LLM selections, (2) behavioral probing to identify which layer information is encoded, and (3) causal intervention analysis (positional bias, description perturbation) that goes beyond correlation to establish mechanistic hypotheses.

### Experiment Justification
- **Experiment 1 (Semantic Matching)**: Tests if cosine similarity between query and tool description embeddings predicts LLM selection → directly tests the "semantic matching hypothesis"
- **Experiment 2 (LLM Tool Selection Baseline)**: Establishes what tools modern LLMs actually select → needed ground truth for all other analyses
- **Experiment 3 (Positional Bias)**: Replicates BiasBusters finding; quantifies how much position vs. content drives selection
- **Experiment 4 (Description Perturbation)**: Causal test — if perturbing semantic content changes selection, content (not just position) drives decisions
- **Experiment 5 (Layer-wise Probing)**: With GPT-2-small, identifies which transformer layers encode tool selection information → locates the "computation" in the network

---

## Research Question
What mechanisms drive tool selection in LLMs? Specifically: (a) Is selection driven by semantic matching between query and tool description, (b) how strongly does positional order affect selection, and (c) at which layers does tool-relevant information emerge in a transformer?

## Background and Motivation
LLMs are increasingly deployed as agents that select from available tools to answer queries. Behavioral benchmarks evaluate success/failure but treat selection as a black box. Two recent papers provide complementary evidence: (1) Healy et al. (2026) show internal representations contain tool-selection information, and (2) BiasBusters (2025) show semantic alignment and positional biases drive behavioral patterns. This project provides a mechanistic bridge: what computations produce these behavioral patterns?

## Hypothesis Decomposition

1. **H1 (Semantic Matching)**: Cosine similarity between query embedding and tool description embedding is a strong predictor of LLM tool selection (better than random, capturing ≥70% of variance in selection).
2. **H2 (Positional Bias)**: LLMs show significant positional bias — earlier-listed tools are selected more often, even when semantically less relevant.
3. **H3 (Content vs. Position)**: Semantic content (description text) is a stronger predictor than position when the semantic signal is strong; position dominates when semantic signals are weak/ambiguous.
4. **H4 (Layer Emergence)**: In GPT-2-small, tool selection information emerges in later layers (layers 9-12 out of 12), consistent with 2601.05214 finding that final-layer representations are discriminative.

## Proposed Methodology

### Approach
Three-phase empirical analysis:
1. **Behavioral analysis**: Collect LLM tool selections via API across structured scenarios, establish ground truth
2. **Semantic analysis**: Compare embedding-based predictions against actual selections
3. **Mechanistic analysis**: Layer-wise probing on GPT-2-small to identify where tool selection is computed

### Experimental Steps

#### Phase A: Dataset Construction (2h)
1. Create extended tool selection dataset (50+ examples across 5 tool categories)
2. Create positional bias variants (same query, tools in different orders)
3. Create description perturbation variants (perturbed/noisy descriptions)

#### Phase B: LLM Behavioral Experiments via API (2h)
1. Use OpenRouter API with GPT-4.1-mini or similar model
2. Present tool selection prompts with 4 candidate tools
3. Collect: selected tool, selection rationale (if chain-of-thought enabled)
4. Run each prompt N=3 times to measure consistency

#### Phase C: Semantic Similarity Analysis (1h)
1. Embed queries and tool descriptions using sentence-transformers (all-MiniLM-L6-v2)
2. Compute cosine similarities between each query and each tool description
3. Compare similarity rankings against LLM selections
4. Statistical analysis: Pearson correlation, ranking accuracy

#### Phase D: Positional Bias Analysis (1h)
1. Re-run same queries with different tool orderings (swap positions)
2. Measure selection stability across orderings
3. Compute positional preference score

#### Phase E: Description Perturbation Analysis (1h)
1. Create semantically neutral descriptions (generic), then perturb toward query
2. Measure how selection probability changes with description relevance
3. Estimate causal effect of description content

#### Phase F: Layer-wise Probing with GPT-2-small (2h)
1. Install TransformerLens
2. Create simple tool selection prompts compatible with GPT-2 tokenization
3. Extract hidden states at each layer (12 layers)
4. Train linear probes on each layer to predict selected tool
5. Plot accuracy by layer to reveal where selection information emerges

### Baselines
1. **Random selection**: Uniform random over tools (lower bound)
2. **Positional baseline**: Always select first-listed tool
3. **Semantic similarity oracle**: Select highest-cosine-similarity tool (tests if embedding similarity = selection)

### Evaluation Metrics
- **Top-1 accuracy**: Does semantic similarity predict the actual selected tool?
- **Spearman rank correlation**: How well does similarity rank order match selection probability?
- **Positional stability score**: Fraction of examples where selection is invariant to tool ordering
- **Layer probe accuracy**: Accuracy of linear probe trained on layer-L representations
- **Cohen's kappa**: Inter-run consistency of LLM selections

### Statistical Analysis Plan
- Chi-square test for positional bias (uniform vs. observed selection distribution over positions)
- Paired t-test for semantic similarity vs. random baseline accuracy
- Significance level: α = 0.05 with Bonferroni correction for multiple comparisons

## Expected Outcomes
- H1 supported if cosine similarity achieves >65% top-1 accuracy on tool selection
- H2 supported if chi-square test rejects uniform distribution (p < 0.05) for position effects
- H3 supported if interaction effect (description quality × position) is significant
- H4 supported if probing accuracy rises sharply in layers 8-12 (vs. flat in early layers)

## Timeline and Milestones
| Phase | Time | Milestone |
|-------|------|-----------|
| Environment setup | 30min | .venv with all packages |
| Dataset creation | 45min | 50+ examples created |
| API experiments | 90min | 150+ LLM selections collected |
| Semantic analysis | 45min | Correlation results |
| Positional/perturbation | 60min | Bias analysis complete |
| Probing | 90min | Layer-wise probe curves |
| Analysis & viz | 45min | 8+ figures saved |
| Documentation | 45min | REPORT.md complete |

## Potential Challenges
1. **API rate limits**: Use retry logic with exponential backoff; cache results to JSON
2. **GPT-2-small CPU performance**: ~30s per forward pass; limit to 100 examples
3. **TransformerLens installation conflicts**: Pin compatible versions; fallback to raw PyTorch with huggingface transformers
4. **Model not following tool format**: Use explicit function calling format; post-process responses

## Success Criteria
- At least 100 LLM tool selections collected and analyzed
- Semantic similarity correlation reported with confidence intervals
- Positional bias quantified with statistical significance
- Layer-wise probing curve showing information emergence
- REPORT.md with all results
