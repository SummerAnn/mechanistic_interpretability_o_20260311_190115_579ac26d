# Downloaded Papers

Papers for research on Mechanistic Interpretability of Tool Selection in LLMs.

## Directly Relevant (Tool Selection + Interpretability)

1. **[Internal Representations as Indicators of Hallucinations in Agent Tool Selection](2601.05214_internal_representations_tool_selection_hallucination.pdf)**
   - Authors: Kait Healy, Bharathi Srinivasan, Visakh Madathil, Jing Wu (Amazon)
   - Year: 2026 (Jan)
   - arXiv: 2601.05214
   - Why relevant: **MOST RELEVANT** — directly studies internal representations of LLMs during tool call generation to detect hallucinations. Shows final-layer hidden states contain discriminative information about tool selection correctness (up to 86.4% accuracy). Uses Glaive Function-Calling dataset.

2. **[BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models](2510.00307_biasbusters_tool_selection_bias.pdf)**
   - Authors: Blankenstein, Yu, Li, Plachouras, Sengupta, Torr, Gal, Paren, Bibi (Oxford/Microsoft)
   - Year: 2025 (Sep)
   - arXiv: 2510.00307
   - Why relevant: Studies what drives tool selection biases — finds semantic alignment between query and tool description is the strongest predictor, plus positional/metadata biases. Uses ToolLLM pipeline.

## Mechanistic Interpretability Foundations

3. **[Towards Automated Circuit Discovery for Mechanistic Interpretability (ACDC)](2304.14997_automated_circuit_discovery_acdc.pdf)**
   - Authors: Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso
   - Year: 2023 (NeurIPS)
   - arXiv: 2304.14997
   - Why relevant: Key method paper for circuit discovery — identifies minimal subgraphs implementing behaviors. Will be used to find tool-selection circuits.

4. **[Open Problems in Mechanistic Interpretability](2501.16496_open_problems_mechanistic_interpretability.pdf)**
   - Authors: Sharkey, Chughtai, Batson, Lindsey et al. (30 authors)
   - Year: 2025 (Jan) — TMLR
   - arXiv: 2501.16496
   - Why relevant: Comprehensive survey of open problems — decomposition methods, circuit discovery pipelines, validation, applications. Defines research agenda.

5. **[Mechanistic Interpretability for AI Safety: A Review](2404.14082_mechanistic_interpretability_ai_safety_review.pdf)**
   - Authors: Multiple
   - Year: 2024
   - arXiv: 2404.14082
   - Why relevant: Overview of MI methods (circuits, probing, SAEs) with safety framing.

6. **[Circuit Component Reuse Across Tasks in Transformer Language Models](2310.08744_circuit_component_reuse_tasks.pdf)**
   - Authors: Merullo, Eickhoff, Pavlick (ICLR 2024 Spotlight)
   - Year: 2024
   - arXiv: 2310.08744
   - Why relevant: Shows circuits generalize — 78% overlap between IOI and Colored Objects circuits. Suggests tool-routing circuits may also be reused.

7. **[LLM Circuit Analyses Are Consistent Across Training and Scale](2407.10827_llm_circuit_analyses_consistent_training_scale.pdf)**
   - Authors: Tigges, Hanna, Yu, Biderman (EleutherAI/NeurIPS 2024)
   - Year: 2024
   - arXiv: 2407.10827
   - Why relevant: Shows circuits found in small models generalize to larger models. Important for scaling our analysis.

## Tool Use Foundations

8. **[Toolformer: Language Models Can Teach Themselves to Use Tools](2302.04761_toolformer_llm_use_tools.pdf)**
   - Authors: Schick, Dwivedi-Yu, Dessì et al. (NeurIPS 2023)
   - Year: 2023
   - arXiv: 2302.04761
   - Why relevant: Foundational paper on LLM tool use — establishes when/how models learn to call APIs. Starting point for understanding tool selection mechanisms.

9. **[ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](2307.16789_toolllm_16000_apis.pdf)**
   - Authors: Qin, Liang, Ye, Zhu et al. (Tsinghua/ICLR 2024)
   - Year: 2023
   - arXiv: 2307.16789
   - Why relevant: Key dataset (ToolBench, 16K APIs) and evaluation framework (ToolEval) for tool selection. DFSDT reasoning strategy.

10. **[StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning](2403.07714_stabletoolbench_benchmarking.pdf)**
    - Authors: Multiple
    - Year: 2024
    - arXiv: 2403.07714
    - Why relevant: Updated stable version of ToolBench for evaluation.

## LLM Routing

11. **[FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](2305.05176_frugalgpt_llm_routing.pdf)**
    - Authors: Chen, Zaharia, Zou (Stanford)
    - Year: 2023
    - arXiv: 2305.05176
    - Why relevant: LLM cascade/routing framework — related concept to tool routing.

12. **[AutoTool: Efficient Tool Selection for Large Language Model Agents](2511.14650_autotool_efficient_tool_selection.pdf)**
    - Authors: Multiple
    - Year: 2025
    - arXiv: 2511.14650
    - Why relevant: Graph-based tool selection using usage inertia — black-box approach, contrast with our MI approach.

13. **[Tool-to-Agent Retrieval: Bridging Tools and Agents for Scalable LLM Multi-Agent Systems](2511.01854_tool_to_agent_retrieval.pdf)**
    - Authors: Multiple
    - Year: 2025
    - arXiv: 2511.01854
    - Why relevant: Embeds tools in shared vector space for retrieval — relates to semantic matching hypothesis.
