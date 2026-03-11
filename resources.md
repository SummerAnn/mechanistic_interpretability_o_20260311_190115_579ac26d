# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project:
**Mechanistic Interpretability of Tool Selection in LLMs**

- Papers downloaded: 13
- Datasets downloaded: 6
- Repositories cloned: 4

---

## Papers

Total papers downloaded: 13

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Internal Representations as Indicators of Hallucinations in Agent Tool Selection | Healy et al. (Amazon) | 2026 | papers/2601.05214_internal_representations_tool_selection_hallucination.pdf | **MOST RELEVANT**: 86.4% accuracy detecting tool hallucinations from final-layer reps |
| BiasBusters: Uncovering and Mitigating Tool Selection Bias | Blankenstein et al. (Oxford) | 2025 | papers/2510.00307_biasbusters_tool_selection_bias.pdf | Semantic alignment = strongest predictor of tool choice |
| ToolLLM: Facilitating LLMs to Master 16000+ Real-world APIs | Qin et al. (Tsinghua) | 2023 | papers/2307.16789_toolllm_16000_apis.pdf | ToolBench dataset, DFSDT, ToolEval; ICLR 2024 Spotlight |
| Towards Automated Circuit Discovery for Mechanistic Interpretability (ACDC) | Conmy et al. | 2023 | papers/2304.14997_automated_circuit_discovery_acdc.pdf | Key MI method; automates circuit finding; NeurIPS 2023 |
| Open Problems in Mechanistic Interpretability | Sharkey et al. (30 authors) | 2025 | papers/2501.16496_open_problems_mechanistic_interpretability.pdf | 82-page survey; TMLR 2025 |
| Mechanistic Interpretability for AI Safety: A Review | Multiple | 2024 | papers/2404.14082_mechanistic_interpretability_ai_safety_review.pdf | Comprehensive MI survey |
| Circuit Component Reuse Across Tasks | Merullo et al. (Brown) | 2024 | papers/2310.08744_circuit_component_reuse_tasks.pdf | 78% circuit overlap between tasks; ICLR 2024 Spotlight |
| LLM Circuit Analyses Are Consistent Across Training and Scale | Tigges et al. (EleutherAI) | 2024 | papers/2407.10827_llm_circuit_analyses_consistent_training_scale.pdf | Circuits generalize across scale; NeurIPS 2024 |
| Toolformer: Language Models Can Teach Themselves to Use Tools | Schick et al. (Meta) | 2023 | papers/2302.04761_toolformer_llm_use_tools.pdf | Foundational tool use paper; NeurIPS 2023 |
| FrugalGPT: How to Use Large Language Models While Reducing Cost | Chen et al. (Stanford) | 2023 | papers/2305.05176_frugalgpt_llm_routing.pdf | LLM cascade routing; 98% cost reduction |
| AutoTool: Efficient Tool Selection for LLM Agents | Multiple | 2025 | papers/2511.14650_autotool_efficient_tool_selection.pdf | Graph-based tool selection using usage patterns |
| StableToolBench | Multiple | 2024 | papers/2403.07714_stabletoolbench_benchmarking.pdf | Stable ToolBench benchmark |
| Tool-to-Agent Retrieval | Multiple | 2025 | papers/2511.01854_tool_to_agent_retrieval.pdf | Semantic embedding space for tool-agent retrieval |

See papers/README.md for detailed descriptions.

---

## Datasets

Total datasets downloaded/created: 6

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Glaive Function Calling V2 (raw) | HuggingFace: glaiveai/glaive-function-calling-v2 | 500 samples | Function calling conversations | datasets/glaive_function_calling_v2/ | Used in 2601.05214; 110K total available |
| Tool Selection from Glaive | Extracted from Glaive V2 | 87 examples | Tool selection classification | datasets/tool_selection_from_glaive/ | {query, tools, selected_tool} format |
| Hermes Function Calling V1 | HuggingFace: NousResearch/hermes-function-calling-v1 | 200 samples | Multi-turn function calling | datasets/hermes_function_calling/ | Rich tool schemas; 12K total available |
| Berkeley Function Calling Leaderboard | HuggingFace: gorilla-llm/BFCL | 100 samples | Function calling eval | datasets/berkeley_function_calling/ | {question, function} pairs |
| IOI Circuit Dataset | Synthetically generated | 200 examples | Indirect Object Identification | datasets/ioi_circuit_dataset/ | Standard MI baseline; {prompt, target, corrupted} |
| Tool Selection MI Dataset | Synthetically generated | 21 examples | Tool selection (equiv. tools) | datasets/tool_selection_mi/ | 3 categories × 4 functionally equivalent tools |

See datasets/README.md for detailed descriptions and download instructions.

---

## Code Repositories

Total repositories cloned: 4

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| TransformerLens | https://github.com/TransformerLensOrg/TransformerLens | Core MI library — load LLMs, cache activations, activation patching | code/TransformerLens/ | Install: `pip install transformer_lens` |
| Automatic-Circuit-Discovery | https://github.com/ArthurConmy/Automatic-Circuit-Discovery | ACDC algorithm for automated circuit discovery | code/Automatic-Circuit-Discovery/ | From NeurIPS 2023 paper |
| ToolBench | https://github.com/OpenBMB/ToolBench | ToolLLM framework — dataset, training, evaluation | code/ToolBench/ | ICLR 2024 Spotlight; includes DFSDT and ToolEval |
| dictionary_learning | https://github.com/saprmarks/dictionary_learning | Sparse Autoencoder training and feature analysis | code/dictionary_learning/ | Pre-trained SAEs available for Pythia |

See code/README.md for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. **Manual web search** (arxiv API rate-limited): Used WebSearch tool with multiple queries:
   - "mechanistic interpretability tool selection LLM routing circuits"
   - "LLM tool calling decision mechanism neural circuits interpretability"
   - "tool selection routing LLM agent probing representation"
   - "sparse autoencoder features tool use function calling LLM"
   - Specific paper title searches for key papers identified

2. **Targeted paper identification** from search results and cross-references:
   - Identified 13 key papers spanning MI foundations, tool use, and routing
   - Downloaded all PDFs from arXiv

3. **Dataset search** via HuggingFace API:
   - Searched for "function calling", "tool calling", "tool use", "glaive"
   - Downloaded top accessible datasets
   - Created synthetic datasets for controlled experiments

4. **Code repository search** via GitHub/prior paper citations

### Selection Criteria
- **Papers**: Relevance to tool selection AND/OR mechanistic interpretability; citation impact; recency
- **Datasets**: Suitability for MI experiments (need clear tool selection examples with input-output pairs); accessibility; size
- **Code**: Active maintenance; compatibility with TransformerLens workflow; direct relevance

### Challenges Encountered
1. **arXiv API rate limiting**: HTTP 429 errors on first attempt; switched to web search for paper discovery
2. **Paper-finder service offline**: Fell back to manual search (well-documented in logs)
3. **Glaive format complexity**: Function calls use non-standard JSON escaping; required custom regex parsing
4. **Salesforce XLAM gated**: Could not download without HuggingFace authentication
5. **ToolBench HuggingFace mirror**: Not publicly accessible via datasets library; cloned GitHub repo instead

### Gaps and Workarounds
1. **No existing tool-selection circuit dataset**: Created synthetic IOI-style dataset for circuit analysis baseline
2. **No labeled "semantically equivalent tools" dataset**: Built custom dataset (tool_selection_mi) with 4 equivalent tools per category, based on BiasBusters methodology
3. **Limited functional call extraction from glaive**: Only 87 examples with multi-tool scenarios extracted; full dataset would require more sophisticated parsing or HF auth token

---

## Recommendations for Experiment Design

### 1. Primary Datasets
- **Glaive Function Calling V2** (full dataset) — Diverse tools, 110K examples, used by most relevant paper (2601.05214)
- **tool_selection_mi** (custom) — Controlled, 4 equivalent tools per category; ideal for bias/circuit analysis
- **ioi_circuit_dataset** (custom) — Standard MI baseline; validates methodology before tool experiments

### 2. Baseline Methods
- **Layer-wise probing**: Train linear classifiers at each layer → reveals where tool identity is encoded
- **NCP + Semantic Similarity**: Replication baselines from 2601.05214
- **Embedding cosine similarity**: Upper bound for semantic matching hypothesis
- **ACDC circuit discovery**: Primary mechanistic analysis method

### 3. Evaluation Metrics
- **Probe accuracy by layer** (where is information encoded?)
- **Circuit faithfulness** (does the circuit faithfully reproduce tool selection?)
- **Counterfactual accuracy** (can patching the circuit change tool selection?)
- **Feature interpretability** (are SAE features monosemantic for tool concepts?)

### 4. Recommended Model Sequence
1. **GPT-2-Small** (117M): Fast, well-studied, ACDC validated on it; start here
2. **Pythia-2.8B**: Larger, used in LLM Circuit Consistency paper; verify generalization
3. **Llama-3.1-8B or Qwen-7B**: State-of-the-art open models; used in 2601.05214

### 5. Key Experiment: Semantic Matching vs. Routing Circuits
The central question is: *Is tool selection driven by semantic matching circuits (similar to IOI) or dedicated routing circuits?*
- **Test**: Do the same attention heads that process entity names in IOI also process tool names in tool selection?
- **Method**: Check circuit overlap using Merullo et al. (2310.08744) approach
- **Expected finding**: Significant overlap (semantic matching reuses existing circuits)
