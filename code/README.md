# Cloned Code Repositories

## 1. TransformerLens
- **URL**: https://github.com/TransformerLensOrg/TransformerLens
- **Location**: code/TransformerLens/
- **Purpose**: Core mechanistic interpretability library for GPT-style language models. Enables:
  - Loading 50+ open-source LLMs with cached internal activations
  - Activation patching (causal intervention)
  - Hook functions to intercept and modify activations
  - Circuit analysis tools
- **Key Entry Points**:
  - `transformer_lens/HookedTransformer` — main model class with hooks
  - `demos/` — demonstration notebooks
  - Install: `pip install transformer_lens`
- **Relevance**: PRIMARY tool for mechanistic interpretability experiments. Will use to:
  1. Cache hidden states during tool call generation
  2. Perform activation patching to identify tool-selection circuits
  3. Study attention head specialization for tool semantics

## 2. Automatic-Circuit-Discovery (ACDC)
- **URL**: https://github.com/ArthurConmy/Automatic-Circuit-Discovery
- **Location**: code/Automatic-Circuit-Discovery/
- **Purpose**: Automates the circuit discovery step in mechanistic interpretability research
- **Key Algorithm**: ACDC — iteratively removes edges from the computational graph while preserving task performance, finding minimal faithful circuits
- **Key Entry Points**:
  - `acdc/` — main algorithm implementation
  - `notebooks/` — demonstration notebooks
  - `experiments/` — experimental scripts
- **Relevance**: Can be adapted to find circuits responsible for tool selection decisions. The workflow: define a tool-selection metric → run ACDC on tool-call prompts → get minimal circuit.

## 3. ToolBench (ToolLLM)
- **URL**: https://github.com/OpenBMB/ToolBench
- **Location**: code/ToolBench/
- **Purpose**: Platform for training, serving, and evaluating LLMs on tool use (ICLR 2024 Spotlight)
- **Key Components**:
  - Dataset: 126K+ instruction-solution path pairs across 16K APIs
  - DFSDT: Depth-first search decision tree for reasoning
  - ToolEval: Automated evaluation framework
  - API Retriever: Neural retrieval for API selection
- **Key Entry Points**:
  - `toolbench/` — core framework
  - `data/` — dataset scripts
- **Relevance**: Source of tool selection data and evaluation framework. The model's tool selection behavior on ToolBench is what we want to interpret mechanistically.

## 4. dictionary_learning (SAE)
- **URL**: https://github.com/saprmarks/dictionary_learning
- **Location**: code/dictionary_learning/
- **Purpose**: Sparse Autoencoder (SAE) training and feature analysis for neural networks
- **Key Components**:
  - Multiple SAE architectures: standard, Gated, TopK (JumpReLU)
  - Uses nnsight for activation access
  - Pre-trained SAEs available for Pythia models
- **Key Entry Points**:
  - `dictionary.py` — SAE implementations
  - `trainers/` — training scripts
  - `utils.py` — loading utilities
- **Relevance**: SAE features can decompose LLM activations into interpretable sparse features. Can identify which features activate during tool selection and whether specific features represent "tool routing" computations.

## Usage for Mechanistic Interpretability of Tool Selection

### Experiment Pipeline:
1. Use **TransformerLens** to load a model (e.g., GPT-2, Pythia-2.8B) and cache activations during tool-call generation
2. Probe hidden states using linear classifiers (similar to 2601.05214 approach) to identify which layers encode tool identity
3. Apply **ACDC** to find the minimal circuit for tool selection (given tool descriptions in context)
4. Use **dictionary_learning** SAEs to find monosemantic features for different tool types
5. Validate using **ToolBench** evaluation framework

### Key Observations from Literature:
- Internal representations (final layer) can distinguish correct vs. hallucinated tool calls with 86.4% accuracy (2601.05214)
- Semantic alignment between query and tool description is the strongest predictor of tool choice (2510.00307)
- Circuit algorithms remain stable across model scale (2407.10827)
