"""
Revised Layer-wise Probing Analysis for Tool Selection.
Uses actual tool-selection prompts (query + tool descriptions) to probe
where in GPT-2 the semantic matching information is encoded.

This is more meaningful than simple category classification because:
- The "label" is which tool position contains the semantically correct tool
- The probe must learn to track semantic relevance across a long context
- This directly tests the mechanistic question: where is tool selection computed?
"""

import json
import random
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import os

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def load_gpt2_model():
    """Load GPT-2-small for activation extraction."""
    from transformers import GPT2Tokenizer, GPT2Model
    print("Loading GPT-2-small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
    model.eval()
    return model, tokenizer


def create_semantic_probing_dataset(n_per_category: int = 25):
    """
    Create a probing dataset where:
    - Prompts include query + 4 tool descriptions
    - The "correct" tool (most semantically relevant) is at a random position
    - Labels are the position (0-3) of the correct tool

    This tests whether hidden states encode "which position has the relevant tool"
    — the core information needed for tool selection.
    """
    categories = {
        "weather": {
            "correct_tool": {
                "name": "weather_api",
                "description": "Access real-time weather data, forecasts, and meteorological conditions"
            },
            "distractors": [
                {"name": "calc_api", "description": "Mathematical computation and algebraic equation solving"},
                {"name": "search_api", "description": "Web search and information retrieval from internet"},
                {"name": "code_api", "description": "Execute programming code and software scripts"},
            ],
            "queries": [
                "What's the temperature outside right now?",
                "Should I carry an umbrella today?",
                "How cold will it be tomorrow?",
                "What are outdoor conditions in Seattle?",
                "Will it be hot this afternoon?",
                "What is the chance of precipitation today?",
                "What weather should I dress for?",
                "What atmospheric pressure is it outside?",
                "Is there a storm warning in my area?",
                "What is the humidity level today?",
                "What are wind speeds like right now?",
                "Will it be clear skies this evening?",
                "What's the UV index today?",
                "Is it safe to fly a kite today?",
                "What climate conditions are expected this weekend?",
                "What temperature was it yesterday?",
                "Is there fog expected this morning?",
                "What are today's morning conditions?",
                "Will it snow tonight?",
                "What's the air quality like outside?",
                "How warm will it get at midday?",
                "What's the dew point right now?",
                "What visibility is expected today?",
                "Is it safe to go hiking today weather-wise?",
                "What are the overnight low temperatures?",
            ]
        },
        "calculator": {
            "correct_tool": {
                "name": "math_api",
                "description": "Evaluate mathematical expressions, solve equations, and compute numerical results"
            },
            "distractors": [
                {"name": "weather_api", "description": "Retrieve weather forecasts and atmospheric conditions"},
                {"name": "translate_api", "description": "Convert text between different languages and dialects"},
                {"name": "search_api", "description": "Find information and resources on the internet"},
            ],
            "queries": [
                "What is the numerical answer to this equation?",
                "Help me figure out the result of this arithmetic",
                "I need to find the value of this expression",
                "What number do I get from this operation?",
                "Determine the output of this formula",
                "I need to perform a quantitative computation",
                "What is the solution to this numerical problem?",
                "Evaluate this expression for me",
                "Find the answer to this math problem",
                "I need to compute a value from these numbers",
                "What does this quantity equal?",
                "Resolve this number computation",
                "Get me the result of multiplying these values",
                "What is the numerical output here?",
                "Determine the sum of these quantities",
                "I need to divide these numbers and find the quotient",
                "What's the final number after these operations?",
                "Give me the exact numerical value",
                "Compute this for me precisely",
                "I need an exact numeric answer to this problem",
                "What is the product of these two numbers?",
                "Find the difference between these values",
                "Determine the square root numerically",
                "What percentage is one value of another?",
                "Solve this equation for its numerical solution",
            ]
        },
        "search": {
            "correct_tool": {
                "name": "web_search",
                "description": "Search the internet and retrieve information from web pages and databases"
            },
            "distractors": [
                {"name": "math_api", "description": "Compute numerical expressions and solve mathematical equations"},
                {"name": "weather_api", "description": "Access atmospheric conditions and meteorological forecasts"},
                {"name": "code_api", "description": "Run programming code in a sandboxed execution environment"},
            ],
            "queries": [
                "I need to find out what's known about this topic",
                "Look up some information for me",
                "Find relevant documents about this subject",
                "What do people know about this issue?",
                "Gather information from available sources",
                "I need to research this topic thoroughly",
                "Find out what has been published about this",
                "What are the latest facts on this matter?",
                "Retrieve all available data on this subject",
                "Help me discover what's out there about this",
                "Look up the answer from public information",
                "What can you find about this topic online?",
                "I need to investigate this subject further",
                "Find all the relevant material on this issue",
                "What do current sources say about this?",
                "Explore information about this specific topic",
                "Find out how others describe this concept",
                "I want to read about this from external sources",
                "Search for documentation on this",
                "What information is available about this?",
                "Look up recent developments on this issue",
                "Find background information on this subject",
                "What does the public record say about this?",
                "Discover information from the world wide web",
                "I need to check facts about this claim",
            ]
        },
        "translation": {
            "correct_tool": {
                "name": "translate_api",
                "description": "Convert text and documents between different languages using neural translation"
            },
            "distractors": [
                {"name": "weather_api", "description": "Get current weather conditions and climate forecasts"},
                {"name": "math_api", "description": "Solve mathematical problems and perform calculations"},
                {"name": "search_api", "description": "Search the web to find information on any topic"},
            ],
            "queries": [
                "Help me express this in another tongue",
                "I need this text in a different linguistic form",
                "Convert this message to another idiom",
                "Make this readable in a foreign dialect",
                "Render this phrase in an alternate vernacular",
                "I need this document in a different script",
                "Adapt this text for a different language speaker",
                "Put this into the right words for another country",
                "I need this understood by people who speak differently",
                "Convert the meaning of this into another language",
                "How do people in another culture say this?",
                "Encode this message in a foreign linguistic system",
                "I need a multilingual version of this text",
                "Help non-English speakers understand this",
                "Transform this into words from another land",
                "Represent this idea in a different language",
                "I need this localized for a foreign audience",
                "Express this concept in a different tongue",
                "Make this meaningful in another linguistic context",
                "Render this in the native language of another country",
                "I need to cross the language barrier with this text",
                "Adapt this statement for speakers of another language",
                "Provide an equivalent in a foreign language",
                "Help me communicate this across languages",
                "Convert these words for a different language speaker",
            ]
        }
    }

    prompts = []
    labels = []
    positions = []  # Which position the correct tool is at
    metadata = []

    for category, cat_data in categories.items():
        correct_tool = cat_data["correct_tool"]
        distractors = cat_data["distractors"]
        queries = cat_data["queries"][:n_per_category]

        for query in queries:
            # Create 4 possible positions for the correct tool
            pos = random.randint(0, 3)

            # Arrange tools: place correct tool at position `pos`
            all_tools = distractors.copy()
            all_tools.insert(pos, correct_tool)

            # Build prompt text
            tools_text = ""
            for i, tool in enumerate(all_tools, 1):
                tools_text += f"\nTool {i}. {tool['name']}: {tool['description']}"

            prompt = f"Select the best tool for this request.\n\nRequest: {query}\n\nAvailable tools:{tools_text}\n\nBest tool:"

            prompts.append(prompt)
            labels.append(category)
            positions.append(pos)
            metadata.append({
                "category": category,
                "query": query,
                "correct_position": pos,
                "tool_order": [t["name"] for t in all_tools]
            })

    return prompts, labels, positions, metadata


def extract_hidden_states(model, tokenizer, prompts: list, max_length: int = 512) -> np.ndarray:
    """
    Extract hidden states at each transformer layer.
    Uses mean pooling over sequence for each layer.
    """
    all_hidden_states = []

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            if i % 20 == 0:
                print(f"  Extracting layer activations: {i+1}/{len(prompts)}")

            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=False
            )

            outputs = model(**inputs)
            # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden_size)

            # Strategy 1: Mean pool over all tokens (captures full context)
            layer_reps = []
            for layer_hs in outputs.hidden_states[1:]:  # Skip embedding layer
                mean_rep = layer_hs[0].mean(dim=0).numpy()
                layer_reps.append(mean_rep)

            all_hidden_states.append(layer_reps)

    return np.array(all_hidden_states)  # (n_prompts, n_layers, hidden_size)


def extract_last_token_states(model, tokenizer, prompts: list, max_length: int = 512) -> np.ndarray:
    """
    Extract hidden states using LAST TOKEN only (more like autoregressive generation).
    """
    all_hidden_states = []

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=False
            )

            outputs = model(**inputs)

            layer_reps = []
            for layer_hs in outputs.hidden_states[1:]:
                last_tok_rep = layer_hs[0, -1, :].numpy()  # Last token
                layer_reps.append(last_tok_rep)

            all_hidden_states.append(layer_reps)

    return np.array(all_hidden_states)


def run_layerwise_probing(hidden_states: np.ndarray, labels_or_positions, n_layers: int = 12) -> list:
    """
    Train linear probe at each layer, return cross-validated accuracy.
    """
    le = LabelEncoder()
    y = le.fit_transform(labels_or_positions)
    n_classes = len(le.classes_)

    results = []
    for layer_idx in range(n_layers):
        X = hidden_states[:, layer_idx, :]

        # Logistic regression with cross-validation
        # Use StratifiedKFold for balanced splits
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        probe = LogisticRegression(max_iter=2000, random_state=42, C=1.0, solver="lbfgs")

        try:
            scores = cross_val_score(probe, X, y, cv=cv, scoring="accuracy")
            mean_acc = float(scores.mean())
            std_acc = float(scores.std())
        except Exception as e:
            print(f"  Layer {layer_idx+1}: Error - {e}")
            mean_acc = 0.0
            std_acc = 0.0

        results.append({
            "layer": layer_idx + 1,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
        })
        print(f"  Layer {layer_idx+1:2d}: {mean_acc:.3f} ± {std_acc:.3f}")

    return results, le


def run_revised_probing_analysis(output_dir: str):
    """
    Run the revised probing analysis with tool-selection format prompts.
    Two probing tasks:
    1. Predict tool CATEGORY from full prompt (should be easy; baseline)
    2. Predict correct tool POSITION (harder; tests selection mechanism)
    """
    print("\n=== Revised Layer-wise Probing Analysis (GPT-2-small) ===")
    print("Task: Probe for tool category and correct tool position in selection prompts")

    # Create dataset
    print("\nCreating semantic probing dataset...")
    prompts, categories, positions, metadata = create_semantic_probing_dataset(n_per_category=25)
    print(f"Created {len(prompts)} prompts across {len(set(categories))} categories")
    print(f"Category distribution: { {c: categories.count(c) for c in set(categories)} }")
    print(f"Position distribution: { {p: positions.count(p) for p in set(positions)} }")

    # Load model
    model, tokenizer = load_gpt2_model()

    # Extract hidden states (mean pooling)
    print("\nExtracting hidden states via mean pooling...")
    mean_states = extract_hidden_states(model, tokenizer, prompts, max_length=512)
    print(f"Hidden states shape: {mean_states.shape}")

    # Extract hidden states (last token)
    print("\nExtracting hidden states via last token...")
    last_states = extract_last_token_states(model, tokenizer, prompts, max_length=512)
    print(f"Last token states shape: {last_states.shape}")

    n_layers = mean_states.shape[1]

    # ── Task 1: Probe for CATEGORY (should be >chance) ──
    print("\n[Task 1] Probing for tool category (4 categories):")
    cat_chance = 1.0 / len(set(categories))
    print(f"Chance level: {cat_chance:.3f}")
    cat_results_mean, cat_le = run_layerwise_probing(mean_states, categories, n_layers)

    # ── Task 2: Probe for POSITION (should be ~chance if model doesn't encode position) ──
    print("\n[Task 2] Probing for correct tool position (0-3):")
    pos_chance = 1.0 / 4  # 4 positions
    print(f"Chance level: {pos_chance:.3f}")
    pos_results_mean, pos_le = run_layerwise_probing(mean_states, positions, n_layers)

    # ── Task 3: Probe for POSITION using last-token states ──
    print("\n[Task 3] Probing for correct tool position using last-token states:")
    pos_results_last, _ = run_layerwise_probing(last_states, positions, n_layers)

    # ── Summary ──
    cat_accs = [r["mean_accuracy"] for r in cat_results_mean]
    pos_accs_mean = [r["mean_accuracy"] for r in pos_results_mean]
    pos_accs_last = [r["mean_accuracy"] for r in pos_results_last]

    print("\n=== Summary ===")
    print(f"Category Probe (mean pool):")
    print(f"  Early (1-4): {np.mean(cat_accs[:4]):.3f}")
    print(f"  Middle (5-8): {np.mean(cat_accs[4:8]):.3f}")
    print(f"  Late (9-12): {np.mean(cat_accs[8:]):.3f}")
    print(f"  Peak: Layer {np.argmax(cat_accs)+1} = {max(cat_accs):.3f}")

    print(f"\nPosition Probe (mean pool):")
    print(f"  Early (1-4): {np.mean(pos_accs_mean[:4]):.3f}")
    print(f"  Middle (5-8): {np.mean(pos_accs_mean[4:8]):.3f}")
    print(f"  Late (9-12): {np.mean(pos_accs_mean[8:]):.3f}")
    print(f"  Peak: Layer {np.argmax(pos_accs_mean)+1} = {max(pos_accs_mean):.3f}")
    print(f"  Chance: {pos_chance:.3f}")

    print(f"\nPosition Probe (last token):")
    print(f"  Early (1-4): {np.mean(pos_accs_last[:4]):.3f}")
    print(f"  Middle (5-8): {np.mean(pos_accs_last[4:8]):.3f}")
    print(f"  Late (9-12): {np.mean(pos_accs_last[8:]):.3f}")
    print(f"  Peak: Layer {np.argmax(pos_accs_last)+1} = {max(pos_accs_last):.3f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "n_prompts": len(prompts),
        "n_categories": len(set(categories)),
        "n_positions": 4,
        "category_chance": cat_chance,
        "position_chance": pos_chance,
        "category_probe_mean_pool": {
            "layer_results": cat_results_mean,
            "accuracies": cat_accs,
            "early_mean": float(np.mean(cat_accs[:4])),
            "middle_mean": float(np.mean(cat_accs[4:8])),
            "late_mean": float(np.mean(cat_accs[8:])),
            "best_layer": int(np.argmax(cat_accs)) + 1,
            "best_accuracy": float(max(cat_accs)),
        },
        "position_probe_mean_pool": {
            "layer_results": pos_results_mean,
            "accuracies": pos_accs_mean,
            "early_mean": float(np.mean(pos_accs_mean[:4])),
            "middle_mean": float(np.mean(pos_accs_mean[4:8])),
            "late_mean": float(np.mean(pos_accs_mean[8:])),
            "best_layer": int(np.argmax(pos_accs_mean)) + 1,
            "best_accuracy": float(max(pos_accs_mean)),
        },
        "position_probe_last_token": {
            "layer_results": pos_results_last,
            "accuracies": pos_accs_last,
            "early_mean": float(np.mean(pos_accs_last[:4])),
            "middle_mean": float(np.mean(pos_accs_last[4:8])),
            "late_mean": float(np.mean(pos_accs_last[8:])),
            "best_layer": int(np.argmax(pos_accs_last)) + 1,
            "best_accuracy": float(max(pos_accs_last)),
        },
        "baseline_accuracy": 1.0 / len(set(categories)),
    }

    with open(f"{output_dir}/probing_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    np.save(f"{output_dir}/mean_hidden_states.npy", mean_states)
    np.save(f"{output_dir}/last_hidden_states.npy", last_states)
    with open(f"{output_dir}/prompts_metadata.json", "w") as f:
        json.dump({"prompts": prompts, "categories": categories, "positions": positions, "metadata": metadata}, f, indent=2)

    print(f"\nSaved probing analysis to {output_dir}")
    return summary


if __name__ == "__main__":
    run_revised_probing_analysis("results/probing_analysis")
