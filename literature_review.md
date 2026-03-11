# Literature Review: Mechanistic Interpretability of Tool Selection in LLMs

**Research Hypothesis**: LLMs increasingly use tools (search, code execution, APIs), but we have no mechanistic understanding of how they decide which tool to call. What representations drive tool choice? Is it semantic matching between tool descriptions and queries, or learned heuristics? Do models have dedicated "routing" circuits, or is selection distributed?

---

## 1. Research Area Overview

This research sits at the intersection of two major areas:
1. **Mechanistic interpretability (MI)**: Reverse-engineering neural network algorithms by identifying circuits—computational subgraphs responsible for specific behaviors
2. **LLM tool use / agents**: Training and evaluating LLMs to call external tools (APIs, functions, services)

The core open question is: *what internal mechanisms drive tool selection in LLMs?* The field currently has excellent behavioral benchmarks (ToolBench, Berkeley FCL) measuring success/failure, and early evidence that internal representations carry discriminative information about tool selection (2601.05214), but no circuit-level mechanistic analysis exists.

---

## 2. Key Papers

### 2.1 Most Directly Relevant

#### Paper 1: Internal Representations as Indicators of Hallucinations in Agent Tool Selection
- **Authors**: Kait Healy, Bharathi Srinivasan, Visakh Madathil, Jing Wu (Amazon)
- **Year**: 2026 (January, AAAI Workshop)
- **Source**: arXiv:2601.05214
- **Key Contribution**: First paper to directly study internal LLM representations in the context of tool selection. Shows that final-layer hidden states contain discriminative information for detecting tool-calling hallucinations.
- **Methodology**:
  1. Collects tool-calling instances across 5 agent categories (calculator, finance, health, sustainability, commerce) from Glaive Function-Calling V2
  2. Extracts final-layer hidden states at 3 positions: function name token, argument tokens (averaged), and closing delimiter
  3. Trains lightweight 2-layer MLP classifiers to detect hallucinations
  4. Evaluates on Qwen7B (7B), GPT-OSS-20B (20B), Llama-3.1-8B (8B)
- **Results**: 72.7–86.4% accuracy in detecting tool-calling hallucinations using single forward pass
  - GPT-OSS-20B: 86.4% accuracy, 86% precision on hallucination class
  - Llama-3.1-8B: 73% accuracy
  - Qwen-7B: 74% accuracy
- **Ablation Study**: Mean pooling of final layer outperforms token-specific methods; simple > complex aggregation
- **Key Finding**: Internal representations of the final layer are sufficient for real-time hallucination detection with O(n·d) complexity
- **Datasets Used**: Glaive Function-Calling V2 (2,411 instances per model)
- **Limitations**: Single-domain reasoning tasks; future work needs cross-architecture unified detector
- **Relevance**: Proof-of-concept that internal states encode tool-selection information. Our work extends this from hallucination detection to full circuit analysis — *where*, *when*, and *how* tool selection is computed.

#### Paper 2: BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models
- **Authors**: Blankenstein, Yu, Li, Plachouras, Sengupta, Torr, Gal, Paren, Bibi (Oxford/Microsoft)
- **Year**: 2025 (September)
- **Source**: arXiv:2510.00307
- **Key Contribution**: First empirical study of tool-selection bias across 7 LLMs using a benchmark of functionally equivalent API clusters
- **Methodology**:
  1. Builds on ToolLLM pipeline; clusters functionally equivalent APIs from RapidAPI
  2. Measures selection imbalance using total variation distance from uniform: δ_API, δ_pos, δ_model
  3. Controlled experiments manipulating tool metadata (name, description, parameters, position)
  4. Regression analysis to identify bias predictors
  5. Mitigation: filter-then-sample-uniformly strategy
- **Key Findings**:
  1. **Semantic alignment** between query and tool description is the **strongest predictor** of tool choice
  2. Perturbing tool descriptions significantly shifts selections (causal effect of metadata)
  3. Repeated pre-training exposure to a specific endpoint amplifies bias
  4. All 7 models tested show systematic bias (fixating on one provider or preferring earlier-listed tools)
- **Datasets Used**: ToolLLM/RapidAPI tool clusters (custom benchmark)
- **Code Available**: Yes (public GitHub)
- **Relevance**: Provides the behavioral data showing tool selection is driven by semantic matching — exactly what we want to study mechanistically. The "strongest predictor" finding directly motivates investigating semantic matching circuits.

---

### 2.2 Mechanistic Interpretability Foundations

#### Paper 3: Towards Automated Circuit Discovery for Mechanistic Interpretability (ACDC)
- **Authors**: Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso
- **Year**: 2023 (NeurIPS)
- **Source**: arXiv:2304.14997
- **Key Contribution**: Systematic workflow and algorithm for circuit discovery — automates the "find minimal faithful subgraph" step
- **Methodology**:
  1. Define behavior: choose dataset + metric that elicits behavior
  2. Represent model as DAG (nodes = attention heads + MLPs, edges = residual stream connections)
  3. ACDC algorithm: iteratively remove edges while preserving ≥threshold% of model performance
- **Key Results**: Rediscovered 5/5 component types in the Greater-Than circuit in GPT-2 Small; selected 68 of 32,000 edges
- **Baselines**: Also adapts Subnetwork Probing (SP) and Head Importance Score for Pruning (HISP)
- **Code Available**: https://github.com/ArthurConmy/Automatic-Circuit-Discovery
- **Relevance**: This is the primary methodological tool we will adapt for tool selection circuit discovery. The three-step workflow (dataset → granularity → patching) applies directly.

#### Paper 4: Open Problems in Mechanistic Interpretability
- **Authors**: Sharkey, Chughtai et al. (30 authors, TMLR 2025)
- **Year**: 2025
- **Source**: arXiv:2501.16496
- **Key Contribution**: Comprehensive 82-page review of open problems in MI, covering methods, applications, socio-technical challenges
- **Relevant Sections for Our Research**:
  - §2.1.2: Sparse dictionary learning (SAEs) for decomposing activations into interpretable features
  - §2.1.3: Describing functional roles of components — both what causes activation and downstream effects
  - §2.2: Concept-based probes — detecting when "tool X is appropriate" is encoded in activations
  - §2.3: Circuit discovery pipelines
  - §3.6: Mechanistic interpretability on broader model families
- **Key Open Problems Relevant to Us**:
  - How to validate that identified circuits faithfully explain behavior
  - Moving from individual-task circuits to reusable circuit components
  - Scaling circuit analysis to larger models
- **Relevance**: Frames our research within the open problems of the field; identifies validation strategies we need to use.

#### Paper 5: Mechanistic Interpretability for AI Safety: A Review
- **Authors**: Nanda et al.
- **Year**: 2024
- **Source**: arXiv:2404.14082
- **Key Contribution**: Overview of MI methods and their relationship to AI safety applications
- **Covers**: Circuits, probing, SAEs, activation patching, logit lens, representation engineering
- **Relevance**: Background/survey for understanding the full toolkit available.

#### Paper 6: Circuit Component Reuse Across Tasks in Transformer Language Models
- **Authors**: Merullo, Eickhoff, Pavlick (ICLR 2024 Spotlight)
- **Year**: 2024
- **Source**: arXiv:2310.08744
- **Key Contribution**: Shows that circuits are not task-specific; 78% overlap between IOI and Colored Objects circuits in GPT-2 — same attention heads implement similar operations across tasks
- **Methodology**: Studied IOI circuit on GPT-2-medium; measured head overlap on "Colored Objects" task
- **Relevance**: If tool-selection uses similar "copy-name" or "identify entity" heads to other tasks, this supports the hypothesis that tool selection is not separate but reuses existing circuits. This is directly testable.

#### Paper 7: LLM Circuit Analyses Are Consistent Across Training and Scale
- **Authors**: Tigges, Hanna, Yu, Biderman (EleutherAI/NeurIPS 2024)
- **Year**: 2024
- **Source**: arXiv:2407.10827
- **Key Contribution**: Tracks circuit emergence and evolution across 300B tokens in Pythia (70M–2.8B parameters)
- **Key Results**:
  - Task abilities and functional components emerge at similar token counts across model sizes
  - Circuit algorithms remain stable despite component-level changes (different attention heads, same function)
  - Circuit size correlates with model size
  - Results suggest small-model circuit analyses generalize to large models
- **Models**: Pythia suite (with training checkpoints) using EAP-IG (Edge Attribution Patching with Integrated Gradients)
- **Datasets Used**: IOI, Greater-Than, Docstring, Induction tasks
- **Relevance**: Justifies using smaller models (GPT-2, Pythia-70M) for initial circuit discovery — findings should generalize. Also provides the EAP-IG methodology for scalable circuit finding.

---

### 2.3 Tool Use Foundations

#### Paper 8: Toolformer: Language Models Can Teach Themselves to Use Tools
- **Authors**: Schick, Dwivedi-Yu et al. (NeurIPS 2023)
- **Year**: 2023
- **Source**: arXiv:2302.04761
- **Key Contribution**: Self-supervised approach for teaching LLMs to use APIs (calculator, QA, search, translation, calendar) via minimal demonstrations
- **Methodology**: Annotates corpus with API calls using few-shot prompting; filters by whether calls reduce loss; fine-tunes on annotated data
- **Key Finding**: Models learn to decide *when* and *how* to call APIs — not just which one
- **Relevance**: Foundational paper showing LLMs can learn tool selection internally. The learned representations are what we want to interpret.

#### Paper 9: ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
- **Authors**: Qin, Liang, Ye et al. (Tsinghua/ICLR 2024)
- **Year**: 2023
- **Source**: arXiv:2307.16789
- **Key Contribution**: ToolBench dataset (16K APIs, 126K instruction-solution pairs), DFSDT reasoning algorithm, ToolEval automatic evaluator
- **Methodology**: 3-stage pipeline: API collection from RapidAPI → ChatGPT-generated instructions → DFSDT solution annotation
- **Key Results**: ToolLLaMA performs on par with ChatGPT; DFSDT outperforms ReACT
- **Evaluation Metrics**: Pass rate (% successfully completed), Win rate (vs. ChatGPT baseline)
- **Datasets Used**: RapidAPI (16K APIs), ToolBench (126K pairs)
- **Code Available**: https://github.com/OpenBMB/ToolBench
- **Relevance**: Primary benchmark for tool selection research. The API retrieval component (neural retriever for tool selection) is a black-box approach — our work provides the mechanistic explanation.

#### Paper 10: StableToolBench
- **Authors**: Multiple
- **Year**: 2024
- **Source**: arXiv:2403.07714
- **Key Contribution**: Stabilized version of ToolBench with virtual server for reproducibility
- **Relevance**: Updated benchmark for reliable evaluation.

---

### 2.4 LLM Routing

#### Paper 11: FrugalGPT
- **Authors**: Chen, Zaharia, Zou (Stanford)
- **Year**: 2023
- **Source**: arXiv:2305.05176
- **Key Contribution**: LLM cascade framework — selects LLMs based on query to balance cost and performance (98% cost reduction, 4% accuracy improvement over GPT-4)
- **Relevance**: Related routing paradigm — routes *between LLMs* rather than *between tools*, but same black-box nature as tool routing. Provides context.

#### Paper 12: AutoTool: Efficient Tool Selection for Large Language Model Agents
- **Authors**: Multiple
- **Year**: 2025
- **Source**: arXiv:2511.14650
- **Key Contribution**: Graph-based framework exploiting "tool usage inertia" (predictable sequential patterns in tool calls)
- **Relevance**: Black-box behavioral approach — contrasts with our mechanistic approach; shows there are predictable patterns we should be able to find mechanistically.

---

## 3. Common Methodologies

### 3.1 Circuit Discovery Pipeline
Standard across papers (ACDC, Circuit Reuse, LLM Circuit Consistency):
1. **Define task**: Prompt format + metric (e.g., logit difference between tool A and tool B)
2. **Create corrupted dataset**: Prompts that elicit different tool selections
3. **Activation patching**: Run model on clean input, replace activations with corrupted-run activations
4. **Identify minimal circuit**: Keep only edges that maintain >threshold% of behavior
5. **Interpret components**: Assign human-interpretable roles to circuit nodes

### 3.2 Probing / Linear Probe Approach
Used in 2601.05214 and referenced throughout:
1. Cache hidden states h^(L) at each layer L for a set of inputs
2. Train linear/MLP classifier on h^(L) to predict a binary label (e.g., "correct tool" vs. "wrong tool")
3. Accuracy as function of layer reveals *where* information is encoded
4. Caution: probes detect correlations, not necessarily causal variables (Open Problems §2.2.3)

### 3.3 Sparse Autoencoder (SAE) Analysis
1. Train SAE on residual stream activations
2. Each SAE feature = a potentially monosemantic direction in activation space
3. Identify features active for tool-related concepts
4. Steer model by activating/deactivating specific features

### 3.4 Semantic Matching Analysis (BiasBusters approach)
1. Compute embedding similarity between query and tool descriptions
2. Regress tool selection on embedding similarity + positional/metadata features
3. Permutation importance to rank features
4. Intervention: perturb descriptions → measure selection change

---

## 4. Standard Baselines

| Baseline | Description | Used In |
|---------|-------------|---------|
| Non-Contradiction Probability (NCP) | Multi-sample consistency check for hallucination detection | 2601.05214 |
| Semantic Similarity | Cosine similarity between tool call samples | 2601.05214 |
| ReACT | Reasoning + Acting with tool calls | ToolLLM |
| DFSDT | Depth-first search decision tree for multi-step reasoning | ToolLLM |
| Uniform random selection | Unbiased baseline for fairness measurement | BiasBusters |
| IOI circuit | Indirect Object Identification — standard MI circuit baseline | ACDC, Circuit Reuse |
| Greater-Than circuit | Arithmetic comparison circuit in GPT-2 | ACDC, LLM Circuit Consistency |

---

## 5. Evaluation Metrics

| Metric | Description | Used For |
|--------|-------------|---------|
| Pass Rate | % of tool-use instructions successfully completed | ToolBench |
| Win Rate | % win vs. ChatGPT baseline | ToolBench |
| Precision/Recall/F1/Accuracy | Classification metrics for hallucination detection | 2601.05214 |
| Total Variation Distance (δ) | Distance from uniform selection distribution | BiasBusters |
| Circuit faithfulness | % of model performance retained by circuit | ACDC, all MI |
| AUC | Area under ROC curve | 2601.05214 ablations |

---

## 6. Datasets in the Literature

| Dataset | Source | Size | Task | Key Papers |
|---------|--------|------|------|------------|
| Glaive Function Calling V2 | HuggingFace (glaiveai) | ~110K | Function calling | 2601.05214 |
| ToolBench | OpenBMB/RapidAPI | 126K+ | Multi-tool API calling | ToolLLM, BiasBusters |
| Berkeley FCL | Gorilla/HuggingFace | ~2K | Function calling eval | Gorilla |
| IOI Dataset | Synthetic | ~1K | Indirect object identification | ACDC, MI circuit papers |
| Greater-Than | Synthetic | ~100 | Arithmetic comparison | ACDC, Hanna et al. |
| Hermes Function Calling | NousResearch | ~12K | Multi-turn function calling | Tool LLM fine-tuning |
| APIBench | Gorilla/Patil et al. | 17K | API call accuracy | ToolLLM (OOD eval) |

---

## 7. Gaps and Opportunities

### Gap 1: No Circuit-Level Analysis of Tool Selection
The most directly relevant paper (2601.05214) shows *that* internal representations encode tool selection information but does not identify *which* circuits or *how* the computation is organized. This is the primary research gap we address.

### Gap 2: Semantic Matching vs. Learned Heuristics
BiasBusters shows semantic alignment is the "strongest predictor" of tool choice but doesn't explain *how* this alignment is computed internally. Is there a dedicated matching circuit, or does it piggyback on existing entity-matching circuits (like those in IOI)?

### Gap 3: Tool Description Encoding
How does the model encode tool descriptions (name, description, parameters) into its context representation? Is there dedicated processing for API documentation, or is it treated as generic text?

### Gap 4: Position/Order Effects vs. Content Effects
BiasBusters shows models prefer earlier-listed tools. Is this a positional attention pattern (like positional heads in IOI circuits) or something else?

### Gap 5: Hallucination Mechanism
2601.05214 shows representations differ between correct and hallucinated tool calls but doesn't explain *why* hallucinations occur. What fails in the circuit?

### Gap 6: Cross-model Generalization
2407.10827 shows circuits generalize across model scales, but all existing MI tool research uses one model snapshot. Does the tool-selection circuit look the same across GPT-2, Pythia-2.8B, and LLaMA-3.1-8B?

---

## 8. Recommendations for Experiment Design

### Recommended Approach
Based on the literature, we recommend a staged approach:

**Stage 1: Establish Behavioral Baseline**
- Use TransformerLens to cache activations during tool-call generation on glaive/ToolBench data
- Train probes at each layer to predict: (a) which tool is selected, (b) correct vs. hallucinated
- Map where tool-identity information is encoded (expected: later layers, final layer is best per 2601.05214)

**Stage 2: Circuit Discovery**
- Design tool-selection prompts with 2-4 equivalent candidate tools
- Create "corrupted" prompts (swap tool descriptions, change semantics)
- Apply ACDC to find minimal faithful circuit for tool selection
- Compare circuit structure to known circuits (IOI, Greater-Than) — check for reuse (2310.08744)

**Stage 3: Feature-Level Analysis (if time permits)**
- Use pre-trained SAEs (from dictionary_learning) to find features active during tool selection
- Identify if any SAE feature represents "tool relevance" or "semantic match"
- Test activation steering: can we force a different tool selection by activating/deactivating features?

### Recommended Primary Dataset
**Glaive Function Calling V2** (already downloaded) — has diverse tool types, multi-domain coverage, and is the dataset used by the most relevant paper (2601.05214).

### Recommended Models
1. **GPT-2-Small** (117M params): Standard MI benchmark model; ACDC and most circuits work uses it; fastest to run
2. **Pythia-2.8B** (from EleutherAI): Used in LLM Circuit Consistency paper; has training checkpoints for developmental analysis

### Recommended Baselines
1. **Probing at random layers** (to show layer-wise information emergence)
2. **NCP / Semantic Similarity** baselines (per 2601.05214)
3. **Uniform random tool selection** (lower bound, per BiasBusters)
4. **Embedding similarity matching** (upper bound — how well does simple cosine similarity predict tool selection?)

### Key Metrics
1. **Circuit faithfulness**: Does the identified circuit reproduce model behavior on tool selection tasks?
2. **Probe accuracy by layer**: Where is tool identity encoded?
3. **Feature specificity**: Are SAE features monosemantic for specific tools/categories?
4. **Counterfactual accuracy**: When we patch the circuit, does tool selection change predictably?
