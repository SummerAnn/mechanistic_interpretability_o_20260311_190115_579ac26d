"""
Semantic Similarity Analysis for Tool Selection.
Tests H1: Does cosine similarity between query and tool description predict LLM selection?

Methods:
1. Compute embeddings for queries and tool descriptions
2. Compute cosine similarities
3. Compare similarity ranking against LLM selections
4. Statistical tests: Spearman correlation, top-1 accuracy
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, kendalltau
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_sentence_transformer():
    """Load a small, fast sentence transformer model."""
    from sentence_transformers import SentenceTransformer
    print("Loading sentence transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def compute_embeddings(model, texts: list) -> np.ndarray:
    """Compute sentence embeddings for a list of texts."""
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def compute_similarity_predictions(model, dataset: list) -> list:
    """
    For each example, compute cosine similarity between query and each tool description.
    Return predicted rankings and compare against actual LLM selections.
    """
    predictions = []

    for example in dataset:
        query = example["query"]
        tools = example["tools"]

        # Encode query and all tool descriptions
        query_emb = compute_embeddings(model, [query])
        tool_texts = [t["description"] for t in tools]
        tool_embs = compute_embeddings(model, tool_texts)

        # Compute cosine similarities
        sims = cosine_similarity(query_emb, tool_embs)[0]
        sim_ranking = np.argsort(-sims)  # Descending: most similar first

        predictions.append({
            "id": example.get("id", "unknown"),
            "query": query,
            "scenario_type": example.get("scenario_type", "unknown"),
            "category": example.get("category", "unknown"),
            "tool_names": [t["name"] for t in tools],
            "tool_descriptions": [t["description"] for t in tools],
            "similarities": sims.tolist(),
            "similarity_ranking": sim_ranking.tolist(),  # Index of tools by similarity
            "predicted_tool": tools[sim_ranking[0]]["name"],  # Top-1 prediction
            "correct_category": example.get("correct_category", example.get("category", "unknown")),
        })

    return predictions


def compare_with_llm_results(sim_predictions: list, llm_results: list) -> pd.DataFrame:
    """
    Compare semantic similarity predictions with actual LLM selections.
    """
    # Index LLM results by example ID
    llm_by_id = {r["id"]: r for r in llm_results}

    comparison_rows = []
    for pred in sim_predictions:
        ex_id = pred["id"]
        if ex_id not in llm_by_id:
            continue

        llm = llm_by_id[ex_id]
        llm_selected = llm["consensus_selection"]

        # Check if semantic similarity top-1 matches LLM selection
        sim_top1 = pred["predicted_tool"]
        sim_correct = (sim_top1 == llm_selected)

        # Rank of LLM-selected tool in similarity ranking
        tool_names = pred["tool_names"]
        if llm_selected in tool_names:
            llm_selected_idx = tool_names.index(llm_selected)
            llm_selected_sim = pred["similarities"][llm_selected_idx]
            # What rank (1-indexed) does LLM's choice have in similarity ranking?
            sim_rank_of_llm_choice = list(pred["similarity_ranking"]).index(llm_selected_idx) + 1
        else:
            llm_selected_sim = None
            sim_rank_of_llm_choice = None

        # LLM selected position in tool list
        llm_selected_position = tool_names.index(llm_selected) if llm_selected in tool_names else -1

        comparison_rows.append({
            "id": ex_id,
            "scenario_type": pred["scenario_type"],
            "category": pred["category"],
            "query": pred["query"],
            "llm_selected": llm_selected,
            "sim_predicted": sim_top1,
            "sim_matches_llm": sim_correct,
            "sim_top1_similarity": pred["similarities"][pred["similarity_ranking"][0]],
            "llm_selected_similarity": llm_selected_sim,
            "sim_rank_of_llm_choice": sim_rank_of_llm_choice,
            "llm_selected_position": llm_selected_position,
            "llm_consistency": llm.get("consistency", 1.0),
            "n_tools": len(tool_names),
        })

    return pd.DataFrame(comparison_rows)


def compute_similarity_metrics(df: pd.DataFrame) -> dict:
    """Compute summary statistics for semantic similarity analysis."""
    metrics = {}

    # Top-1 accuracy: similarity prediction matches LLM selection
    metrics["top1_accuracy"] = df["sim_matches_llm"].mean()
    metrics["top1_accuracy_same_cat"] = df[df["scenario_type"] == "same_category"]["sim_matches_llm"].mean()
    metrics["top1_accuracy_mixed_cat"] = df[df["scenario_type"] == "mixed_category"]["sim_matches_llm"].mean()

    # Average rank of LLM's chosen tool in similarity ranking
    valid_ranks = df["sim_rank_of_llm_choice"].dropna()
    metrics["mean_rank_of_llm_choice"] = valid_ranks.mean()
    metrics["median_rank_of_llm_choice"] = valid_ranks.median()
    metrics["chance_mean_rank"] = (df["n_tools"].mean() + 1) / 2  # Expected rank under random

    # Correlation between similarity rank and LLM selection
    # If similarity drives selection, lower rank should be preferred
    valid_df = df.dropna(subset=["sim_rank_of_llm_choice"])
    n = len(valid_df)
    metrics["n_examples"] = n

    if n > 5:
        # Test: does high similarity correlate with selection?
        # Proxy: sim_top1_similarity vs llm_selected_similarity
        valid_sims = df.dropna(subset=["llm_selected_similarity"])
        metrics["mean_selected_sim"] = valid_sims["llm_selected_similarity"].mean()
        metrics["mean_top_sim"] = valid_sims["sim_top1_similarity"].mean()

    # Positional bias check
    # Is position 0 disproportionately selected?
    metrics["position_0_rate"] = (df["llm_selected_position"] == 0).mean()
    metrics["expected_position_0_rate"] = 1.0 / df["n_tools"].mean()

    return metrics


def analyze_per_category(df: pd.DataFrame) -> pd.DataFrame:
    """Per-category breakdown of similarity accuracy."""
    cat_stats = df.groupby("category").agg(
        n_examples=("id", "count"),
        top1_accuracy=("sim_matches_llm", "mean"),
        mean_consistency=("llm_consistency", "mean"),
        position_0_rate=("llm_selected_position", lambda x: (x == 0).mean()),
    ).reset_index()
    return cat_stats


def run_semantic_analysis(main_results_path: str, output_dir: str):
    """
    Full semantic analysis pipeline.
    1. Load LLM results
    2. Reload original dataset for tool information
    3. Compute similarity predictions
    4. Compare and analyze
    """
    import sys
    print("Starting semantic similarity analysis...")

    # Load LLM results
    with open(main_results_path) as f:
        llm_results = json.load(f)
    print(f"Loaded {len(llm_results)} LLM results")

    # Reconstruct dataset-like objects from LLM results for embedding computation
    # (we need query + tool descriptions)
    # The LLM results don't have descriptions, so reload from original dataset
    dataset_path = "datasets/experiment_datasets/tool_selection_dataset.json"
    with open(dataset_path) as f:
        full_dataset = json.load(f)

    # Index original dataset by ID
    dataset_by_id = {ex["id"]: ex for ex in full_dataset}

    # Match LLM results with full dataset examples
    matched_dataset = []
    for result in llm_results:
        ex_id = result["id"]
        if ex_id in dataset_by_id:
            matched_dataset.append(dataset_by_id[ex_id])

    print(f"Matched {len(matched_dataset)} examples with full dataset")

    # Load embedding model
    model = load_sentence_transformer()

    # Compute similarity predictions
    print("Computing semantic similarity predictions...")
    sim_predictions = compute_similarity_predictions(model, matched_dataset)

    # Compare with LLM results
    print("Comparing similarity predictions with LLM selections...")
    comparison_df = compare_with_llm_results(sim_predictions, llm_results)

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_similarity_metrics(comparison_df)

    print("\n=== Semantic Similarity Analysis Results ===")
    print(f"Top-1 Accuracy (overall): {metrics['top1_accuracy']:.3f}")
    print(f"Top-1 Accuracy (same-category): {metrics.get('top1_accuracy_same_cat', 'N/A'):.3f}")
    print(f"Top-1 Accuracy (mixed-category): {metrics.get('top1_accuracy_mixed_cat', 'N/A'):.3f}")
    print(f"Mean rank of LLM choice: {metrics['mean_rank_of_llm_choice']:.2f} (chance: {metrics['chance_mean_rank']:.2f})")
    print(f"Position-0 selection rate: {metrics['position_0_rate']:.3f} (expected: {metrics['expected_position_0_rate']:.3f})")

    per_cat = analyze_per_category(comparison_df)
    print("\nPer-category breakdown:")
    print(per_cat.to_string(index=False))

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    comparison_df.to_csv(f"{output_dir}/semantic_comparison.csv", index=False)
    with open(f"{output_dir}/semantic_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{output_dir}/sim_predictions.json", "w") as f:
        json.dump(sim_predictions, f, indent=2)
    per_cat.to_csv(f"{output_dir}/per_category_semantic.csv", index=False)

    print(f"\nSaved semantic analysis results to {output_dir}")
    return comparison_df, metrics, per_cat


if __name__ == "__main__":
    run_semantic_analysis(
        "results/main_selection_results.json",
        "results/semantic_analysis"
    )
