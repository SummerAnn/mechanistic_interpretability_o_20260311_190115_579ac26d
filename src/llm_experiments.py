"""
LLM Tool Selection Experiments via OpenRouter API.
Tests which tool LLMs select given structured prompts.

Experiments:
1. Main tool selection (same-category and mixed-category)
2. Positional bias (same examples, rotated tool positions)
3. Description perturbation (original vs. adversarial descriptions)
"""

import json
import os
import time
import random
from pathlib import Path
from openai import OpenAI

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# API Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-4.1-mini"  # Cost-effective yet capable

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def build_tool_selection_prompt(query: str, tools: list) -> str:
    """Build a structured prompt for tool selection."""
    tools_text = ""
    for i, tool in enumerate(tools, 1):
        tools_text += f"\n{i}. **{tool['name']}**: {tool['description']}"
        params = ", ".join(f"{k}: {v}" for k, v in tool.get("parameters", {}).items())
        if params:
            tools_text += f"\n   Parameters: {params}"

    prompt = f"""You are an AI assistant that selects the most appropriate tool for a given task.

Available tools:{tools_text}

User request: {query}

Which tool should be used to handle this request? Respond with ONLY the tool name (exactly as listed above), nothing else."""
    return prompt


def call_llm_with_retry(prompt: str, max_retries: int = 3, temperature: float = 0.0) -> str:
    """Call LLM with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=50,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  Retry {attempt + 1} after {wait_time}s error: {e}")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return "ERROR"


def extract_tool_name(response: str, valid_tools: list) -> str:
    """Extract the selected tool name from LLM response."""
    response_lower = response.lower().strip()
    # First try exact match
    for tool in valid_tools:
        if tool["name"].lower() == response_lower:
            return tool["name"]
    # Then try substring match
    for tool in valid_tools:
        if tool["name"].lower() in response_lower:
            return tool["name"]
    # Return the raw response if no match
    return response


def run_tool_selection_experiment(dataset_path: str, output_path: str, n_runs: int = 3, max_examples: int = 50):
    """
    Run the main tool selection experiment.
    For each example, call LLM n_runs times and record selections.
    """
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Sample a manageable subset
    same_cat = [e for e in dataset if e["scenario_type"] == "same_category"]
    mixed_cat = [e for e in dataset if e["scenario_type"] == "mixed_category"]

    # Take 8 examples per category for same_cat, all mixed_cat
    sampled = []
    seen_categories = {}
    for ex in same_cat:
        cat = ex["category"]
        seen_categories.setdefault(cat, 0)
        if seen_categories[cat] < 8:
            sampled.append(ex)
            seen_categories[cat] += 1
    sampled.extend(mixed_cat)

    if len(sampled) > max_examples:
        sampled = sampled[:max_examples]

    print(f"Running experiment on {len(sampled)} examples, {n_runs} runs each")
    print(f"Total API calls: {len(sampled) * n_runs}")

    results = []
    for i, example in enumerate(sampled):
        print(f"Example {i+1}/{len(sampled)}: {example['id']} - {example['query'][:60]}...")

        example_results = []
        for run in range(n_runs):
            prompt = build_tool_selection_prompt(example["query"], example["tools"])
            raw_response = call_llm_with_retry(prompt, temperature=0.0)
            selected_tool = extract_tool_name(raw_response, example["tools"])

            # Determine if correct (for mixed-category, check if in correct category)
            correct_tools = example.get("correct_tool_names", [])
            correct_category = example.get("correct_category", "")
            tool_category_map = {t["name"]: example["category"] for t in example["tools"]}

            is_correct = (
                selected_tool in correct_tools or
                (correct_category and any(
                    correct_category in t["name"] or correct_category in t.get("description", "").lower()
                    for t in example["tools"] if t["name"] == selected_tool
                ))
            )

            example_results.append({
                "run": run,
                "raw_response": raw_response,
                "selected_tool": selected_tool,
                "is_valid_tool": selected_tool in [t["name"] for t in example["tools"]],
            })

        # Analyze consistency across runs
        selected_tools = [r["selected_tool"] for r in example_results]
        most_common = max(set(selected_tools), key=selected_tools.count)

        results.append({
            "id": example["id"],
            "query": example["query"],
            "scenario_type": example["scenario_type"],
            "category": example["category"],
            "correct_category": example.get("correct_category", example["category"]),
            "tool_names": [t["name"] for t in example["tools"]],
            "tool_positions": {t["name"]: i for i, t in enumerate(example["tools"])},
            "runs": example_results,
            "consensus_selection": most_common,
            "consistency": selected_tools.count(most_common) / n_runs,
            "selected_position": [t["name"] for t in example["tools"]].index(most_common) if most_common in [t["name"] for t in example["tools"]] else -1,
        })

        # Small delay between examples
        time.sleep(0.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to {output_path}")
    return results


def run_positional_bias_experiment(dataset_path: str, output_path: str):
    """
    Run positional bias experiment.
    Same query, different tool orderings — does position affect selection?
    """
    with open(dataset_path) as f:
        dataset = json.load(f)

    print(f"Running positional bias experiment on {len(dataset)} variants")

    results = []
    for i, example in enumerate(dataset):
        print(f"Variant {i+1}/{len(dataset)}: {example['variant_id']}")

        prompt = build_tool_selection_prompt(example["query"], example["tools"])
        raw_response = call_llm_with_retry(prompt, temperature=0.0)
        selected_tool = extract_tool_name(raw_response, example["tools"])

        selected_position = -1
        for pos, tool in enumerate(example["tools"]):
            if tool["name"] == selected_tool:
                selected_position = pos
                break

        results.append({
            "base_id": example["base_id"],
            "variant_id": example["variant_id"],
            "query": example["query"],
            "category": example["category"],
            "rotation": example["rotation"],
            "tool_order": [t["name"] for t in example["tools"]],
            "raw_response": raw_response,
            "selected_tool": selected_tool,
            "selected_position": selected_position,  # 0=first, 1=second, etc.
            "correct_category": example["correct_category"],
        })

        time.sleep(0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} positional bias results to {output_path}")
    return results


def run_description_perturbation_experiment(dataset_path: str, output_path: str, n_runs: int = 3):
    """
    Run description perturbation experiment.
    Tests if changing description text changes selection — causal analysis.
    """
    with open(dataset_path) as f:
        dataset = json.load(f)

    print(f"Running description perturbation experiment on {len(dataset)} examples")

    results = []
    for i, example in enumerate(dataset):
        print(f"Example {i+1}/{len(dataset)}: {example['id']} (condition: {example['condition']})")

        runs = []
        for run in range(n_runs):
            prompt = build_tool_selection_prompt(example["query"], example["tools"])
            raw_response = call_llm_with_retry(prompt, temperature=0.0)
            selected_tool = extract_tool_name(raw_response, example["tools"])
            runs.append({"run": run, "raw_response": raw_response, "selected_tool": selected_tool})

        selected_tools = [r["selected_tool"] for r in runs]
        most_common = max(set(selected_tools), key=selected_tools.count)

        results.append({
            "id": example["id"],
            "query": example["query"],
            "condition": example["condition"],
            "tool_names": [t["name"] for t in example["tools"]],
            "tool_descriptions": {t["name"]: t["description"] for t in example["tools"]},
            "runs": runs,
            "consensus_selection": most_common,
            "consistency": selected_tools.count(most_common) / n_runs,
        })

        time.sleep(0.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} perturbation results to {output_path}")
    return results


if __name__ == "__main__":
    base_dir = "results"

    # Run main experiment
    print("=" * 60)
    print("EXPERIMENT 1: Main Tool Selection")
    print("=" * 60)
    run_tool_selection_experiment(
        "datasets/experiment_datasets/tool_selection_dataset.json",
        f"{base_dir}/main_selection_results.json",
        n_runs=3,
        max_examples=45
    )

    # Run positional bias experiment
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Positional Bias")
    print("=" * 60)
    run_positional_bias_experiment(
        "datasets/experiment_datasets/positional_bias_dataset.json",
        f"{base_dir}/positional_bias_results.json"
    )

    # Run description perturbation experiment
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Description Perturbation")
    print("=" * 60)
    run_description_perturbation_experiment(
        "datasets/experiment_datasets/description_perturbation_dataset.json",
        f"{base_dir}/perturbation_results.json",
        n_runs=3
    )
