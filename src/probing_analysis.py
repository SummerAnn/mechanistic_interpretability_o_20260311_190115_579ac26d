"""
Layer-wise Probing Analysis for Tool Selection in GPT-2-Small.
Tests H4: Tool selection information emerges in later layers of a transformer.

Method:
1. Load GPT-2-small via transformers (no GPU needed)
2. Create simple tool selection prompts
3. Extract hidden states at each layer
4. Train linear probes on each layer
5. Plot accuracy by layer
"""

import json
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import os


def load_gpt2_model():
    """Load GPT-2-small model for activation extraction."""
    print("Loading GPT-2-small model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
    model.eval()
    print(f"GPT-2 loaded: 12 layers, hidden_size=768")
    return model, tokenizer


def create_probing_prompts(n_per_category: int = 20):
    """
    Create simple prompts for probing.
    Each prompt pairs a query with a tool type;
    the label is the correct tool category.
    """
    # Simple patterns that GPT-2 can process
    templates = {
        "weather": [
            "The weather in {city} today is",
            "Current temperature in {city}:",
            "Will it rain in {city} tomorrow?",
            "What is the forecast for {city}?",
            "Weather conditions in {city}:",
        ] * 4,
        "calculator": [
            "The result of {a} + {b} is",
            "Calculate {a} times {b}:",
            "Solve the equation {a} x = {b}:",
            "What is {a} divided by {b}?",
            "Compute {a} squared:",
        ] * 4,
        "search": [
            "Search results for '{query}':",
            "Looking up information about {query}:",
            "Wikipedia article on {query}:",
            "Google search for {query}:",
            "Find information about {query}:",
        ] * 4,
        "translation": [
            "Translation from English to French:",
            "The French word for '{word}' is",
            "In Spanish, 'hello' is",
            "Translate the following to German:",
            "The Italian translation of '{word}' is",
        ] * 4,
        "code": [
            "def calculate_sum(a, b):",
            "import numpy as np",
            "# Python function to sort a list:",
            "Execute the following code:",
            "Run this Python script:",
        ] * 4,
    }

    cities = ["Paris", "London", "Tokyo", "Berlin", "New York", "Sydney", "Moscow"]
    numbers = [(12, 7), (25, 4), (100, 3), (15, 8), (42, 6)]
    queries = ["machine learning", "climate change", "quantum physics", "ancient Rome"]
    words = ["cat", "house", "water", "time", "friend"]

    prompts = []
    labels = []

    for category, templates_list in templates.items():
        for i, template in enumerate(templates_list[:n_per_category]):
            # Fill template with values
            city = cities[i % len(cities)]
            a, b = numbers[i % len(numbers)]
            query = queries[i % len(queries)]
            word = words[i % len(words)]

            text = template.format(
                city=city, a=a, b=b, query=query, word=word
            )
            prompts.append(text)
            labels.append(category)

    return prompts, labels


def extract_hidden_states(model, tokenizer, prompts: list, max_length: int = 50) -> np.ndarray:
    """
    Extract hidden states at each layer for a list of prompts.
    Returns array of shape (n_prompts, n_layers, hidden_size).
    """
    all_hidden_states = []

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            if i % 10 == 0:
                print(f"  Processing prompt {i+1}/{len(prompts)}...")

            # Tokenize (truncate to max_length)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=False
            )

            outputs = model(**inputs)

            # outputs.hidden_states: tuple of (n_layers+1) tensors, shape (1, seq_len, hidden_size)
            # Use mean pooling of last token across layers
            hidden_states = outputs.hidden_states  # (13,) tuple for 12-layer GPT-2

            # Mean pool over sequence length for each layer
            layer_representations = []
            for layer_hs in hidden_states[1:]:  # Skip embedding layer (index 0)
                # Shape: (1, seq_len, hidden_size) -> mean -> (hidden_size,)
                layer_rep = layer_hs[0].mean(dim=0).numpy()
                layer_representations.append(layer_rep)

            all_hidden_states.append(layer_representations)

    # Shape: (n_prompts, n_layers, hidden_size)
    return np.array(all_hidden_states)


def train_probes(hidden_states: np.ndarray, labels: list) -> dict:
    """
    Train a linear probe at each layer to predict tool category.
    Returns accuracy per layer.
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_layers = hidden_states.shape[1]

    print(f"Training probes at {n_layers} layers with {len(labels)} examples...")
    print(f"Classes: {le.classes_}")

    layer_results = {}
    for layer_idx in range(n_layers):
        X = hidden_states[:, layer_idx, :]  # (n_prompts, hidden_size)

        # Logistic regression probe with cross-validation
        probe = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        cv_scores = cross_val_score(probe, X, y, cv=5, scoring="accuracy")

        layer_results[layer_idx] = {
            "layer": layer_idx + 1,
            "mean_accuracy": float(cv_scores.mean()),
            "std_accuracy": float(cv_scores.std()),
            "cv_scores": cv_scores.tolist(),
        }
        print(f"  Layer {layer_idx+1:2d}: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return layer_results, le


def compute_baseline_accuracy(labels: list) -> float:
    """Majority class baseline accuracy."""
    from collections import Counter
    counts = Counter(labels)
    return max(counts.values()) / len(labels)


def run_probing_analysis(output_dir: str):
    """Full probing analysis pipeline."""
    print("\n=== Layer-wise Probing Analysis (GPT-2-small) ===")

    # Create probing dataset
    print("Creating probing prompts...")
    prompts, labels = create_probing_prompts(n_per_category=20)
    print(f"Created {len(prompts)} prompts across {len(set(labels))} categories")

    # Load model
    model, tokenizer = load_gpt2_model()

    # Extract hidden states
    print("\nExtracting hidden states...")
    hidden_states = extract_hidden_states(model, tokenizer, prompts)
    print(f"Hidden states shape: {hidden_states.shape}")  # (n_prompts, 12, 768)

    # Baseline
    baseline = compute_baseline_accuracy(labels)
    chance = 1.0 / len(set(labels))
    print(f"\nBaseline (majority class): {baseline:.3f}")
    print(f"Chance (uniform): {chance:.3f}")

    # Train probes
    print("\nTraining linear probes at each layer:")
    layer_results, label_encoder = train_probes(hidden_states, labels)

    # Summary
    print("\n=== Probe Accuracy by Layer ===")
    accuracies = [layer_results[i]["mean_accuracy"] for i in range(12)]
    print(f"Early layers (1-4): {np.mean(accuracies[:4]):.3f}")
    print(f"Middle layers (5-8): {np.mean(accuracies[4:8]):.3f}")
    print(f"Late layers (9-12): {np.mean(accuracies[8:]):.3f}")
    print(f"Best layer: {np.argmax(accuracies)+1} with accuracy {max(accuracies):.3f}")
    print(f"Final layer accuracy: {accuracies[-1]:.3f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    probing_summary = {
        "n_prompts": len(prompts),
        "n_categories": len(set(labels)),
        "categories": list(label_encoder.classes_),
        "baseline_accuracy": baseline,
        "chance_accuracy": chance,
        "layer_results": layer_results,
        "accuracies_by_layer": accuracies,
        "early_layers_mean": float(np.mean(accuracies[:4])),
        "middle_layers_mean": float(np.mean(accuracies[4:8])),
        "late_layers_mean": float(np.mean(accuracies[8:])),
        "best_layer": int(np.argmax(accuracies)) + 1,
        "best_layer_accuracy": float(max(accuracies)),
        "final_layer_accuracy": float(accuracies[-1]),
    }

    with open(f"{output_dir}/probing_results.json", "w") as f:
        json.dump(probing_summary, f, indent=2)

    # Save the raw data too
    np.save(f"{output_dir}/hidden_states.npy", hidden_states)
    with open(f"{output_dir}/prompts_labels.json", "w") as f:
        json.dump({"prompts": prompts, "labels": labels}, f, indent=2)

    print(f"\nSaved probing analysis to {output_dir}")
    return probing_summary


if __name__ == "__main__":
    run_probing_analysis("results/probing_analysis")
