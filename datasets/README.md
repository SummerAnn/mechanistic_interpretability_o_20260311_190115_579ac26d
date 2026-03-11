# Datasets for Mechanistic Interpretability of Tool Selection in LLMs

This directory contains datasets for research experiments. Data files (large CSVs, HuggingFace datasets) are excluded from git.

---

## Dataset 1: Glaive Function Calling V2 (tool_selection_from_glaive/)

### Overview
- **Source**: `glaiveai/glaive-function-calling-v2` on HuggingFace
- **Size**: ~110K total samples; 87 extracted tool-selection examples (from 5K sample)
- **Format**: JSON — `{query, available_tools, selected_tool, arguments, num_tools}`
- **Task**: Tool selection classification — given a query and available tools, predict which tool to call
- **License**: Creative Commons

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
ds.save_to_disk("datasets/glaive_full")
```

### Loading the Dataset (already partially downloaded)

```python
import json
with open("datasets/tool_selection_from_glaive/train_examples.json") as f:
    examples = json.load(f)
```

### Sample Data (see samples.json)

```json
{
  "query": "I need some inspiration. Can you give me a quote?",
  "available_tools": ["get_random_quote", "get_stock_price"],
  "selected_tool": "get_random_quote",
  "arguments": "{}",
  "num_tools": 2
}
```

---

## Dataset 2: Glaive Function Calling V2 (Raw Samples) (glaive_function_calling_v2/)

### Overview
- **Source**: `glaiveai/glaive-function-calling-v2` on HuggingFace
- **Size**: 500 raw samples
- **Format**: JSON — `{system: "SYSTEM: You are...", chat: "USER: ... ASSISTANT: ..."}`
- **Task**: Raw function-calling interactions with system prompts defining available tools

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
```

---

## Dataset 3: NousResearch Hermes Function Calling V1 (hermes_function_calling/)

### Overview
- **Source**: `NousResearch/hermes-function-calling-v1` on HuggingFace
- **Size**: 200 samples downloaded; ~12K total
- **Format**: JSON — `{id, conversations, tools, category, subcategory, task}`
- **Task**: Multi-turn function-calling conversations with structured tool schemas

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("NousResearch/hermes-function-calling-v1", split="train")
ds.save_to_disk("datasets/hermes_function_calling_full")
```

---

## Dataset 4: Berkeley Function Calling Leaderboard (berkeley_function_calling/)

### Overview
- **Source**: `gorilla-llm/Berkeley-Function-Calling-Leaderboard` on HuggingFace
- **Size**: 100 samples downloaded
- **Format**: JSON — `{question, function}` where `function` is a tool schema
- **Task**: Single-turn function calling evaluation benchmark

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard", split="train")
```

---

## Dataset 5: IOI Circuit Analysis Dataset (ioi_circuit_dataset/)

### Overview
- **Source**: Synthetically generated (based on Wang et al. 2023 IOI task)
- **Size**: 200 examples
- **Format**: JSON — `{prompt, target, corrupted_prompt, subject, indirect_object, place, object}`
- **Task**: Indirect Object Identification — standard MI circuit analysis task
- **Use**: Baseline circuit analysis comparison; validates MI methodology before applying to tool selection

### Loading

```python
import json
with open("datasets/ioi_circuit_dataset/ioi_examples.json") as f:
    ioi_data = json.load(f)
```

---

## Dataset 6: Tool Selection MI Dataset (tool_selection_mi/)

### Overview
- **Source**: Synthetically generated for this research
- **Size**: 21 examples with multiple categories
- **Format**: JSON — `{query, category, tools, correct_category}`
- **Task**: Tool selection from functionally equivalent APIs (weather, search, calculator)
- **Use**: Primary dataset for mechanistic interpretability experiments — designed to have multiple functionally equivalent tools per category, enabling bias analysis

### Categories
- **weather** (6 examples): WeatherAPI, World Weather Online, OpenWeatherMap, Weather Forecast 14
- **search** (6 examples): Google Search, Bing, SERP API, Brave Search
- **calculator** (6 examples): Wolfram Alpha, MathJS, Calculator API, SymPy
- **mixed** (3 examples): Mixed weather + search tools

### Loading

```python
import json
with open("datasets/tool_selection_mi/tool_selection_examples.json") as f:
    data = json.load(f)
```

---

## Notes for Experiment Runner

### Priority Datasets for Experiments
1. **tool_selection_from_glaive** — Real LLM tool selection data with known ground truth
2. **tool_selection_mi** — Synthetic but controlled, ideal for ablations
3. **ioi_circuit_dataset** — Baseline validation of MI pipeline
4. **hermes_function_calling** — Rich multi-turn data with tool schemas

### Key for Interpretability Experiments
- Use `available_tools` + `query` as input to model
- Probe hidden states at each layer to find where tool identity is encoded
- Compare activations between correct and incorrect tool selection
- Apply ACDC on minimal tool-selection prompt to find circuits
