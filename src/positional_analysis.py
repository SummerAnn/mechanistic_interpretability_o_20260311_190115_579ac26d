"""
Positional Bias Analysis for Tool Selection.
Tests H2: Do LLMs show significant positional bias (prefer earlier tools)?

Analysis:
1. Load positional bias experiment results
2. Compute selection rates by position
3. Statistical test: chi-square for uniform distribution
4. Compute stability across rotations
5. Visualize position preferences
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
import os


def load_positional_results(results_path: str) -> list:
    """Load positional bias experiment results."""
    with open(results_path) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} positional bias results")
    return results


def compute_positional_bias(results: list) -> dict:
    """
    Analyze position preferences in LLM tool selection.

    Returns:
    - Selection rates by position (0-3)
    - Chi-square test result
    - Selection stability across rotations
    """
    # Group by base_id
    by_base = {}
    for r in results:
        base_id = r["base_id"]
        by_base.setdefault(base_id, []).append(r)

    # Count selections by position
    position_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total = 0

    stability_scores = []
    examples_data = []

    for base_id, variants in by_base.items():
        # For each rotation variant, record which position was selected
        selected_positions = [v["selected_position"] for v in variants if v["selected_position"] >= 0]

        for pos in selected_positions:
            if 0 <= pos <= 3:
                position_counts[pos] += 1
                total += 1

        # Stability: for each base_id, how consistent is the TOOL selection across rotations?
        # Map selected position to tool name (accounting for rotation)
        selected_tools = []
        for v in variants:
            sel_tool = v["selected_tool"]
            sel_pos = v["selected_position"]
            selected_tools.append(sel_tool)

        if len(selected_tools) > 1:
            most_common_tool = max(set(selected_tools), key=selected_tools.count)
            stability = selected_tools.count(most_common_tool) / len(selected_tools)
            stability_scores.append(stability)

        examples_data.append({
            "base_id": base_id,
            "category": variants[0]["category"],
            "n_variants": len(variants),
            "selected_tools": selected_tools,
            "stability": stability_scores[-1] if stability_scores else None,
        })

    # Statistical test: chi-square for uniform distribution over positions
    observed = [position_counts.get(i, 0) for i in range(4)]
    expected = [total / 4] * 4

    if total > 0:
        chi2, pvalue = stats.chisquare(observed, expected)
    else:
        chi2, pvalue = 0, 1.0

    # Compute selection rate by position
    position_rates = {pos: count / total for pos, count in position_counts.items()} if total > 0 else {}

    # Compute position bias score: how much do rates deviate from uniform?
    total_variation = sum(abs(rate - 0.25) for rate in position_rates.values()) / 2

    return {
        "position_counts": position_counts,
        "position_rates": position_rates,
        "total_examples": total,
        "chi2_statistic": float(chi2),
        "chi2_pvalue": float(pvalue),
        "position_0_rate": position_rates.get(0, 0),
        "mean_stability": float(np.mean(stability_scores)) if stability_scores else None,
        "total_variation_distance": float(total_variation),
        "n_base_examples": len(by_base),
        "stability_per_example": examples_data,
    }


def analyze_stability_by_category(results: list) -> pd.DataFrame:
    """Per-category stability analysis."""
    # Group by base_id and get category
    by_base = {}
    for r in results:
        base_id = r["base_id"]
        by_base.setdefault(base_id, []).append(r)

    rows = []
    for base_id, variants in by_base.items():
        category = variants[0]["category"]
        selected_tools = [v["selected_tool"] for v in variants]
        if not selected_tools:
            continue
        most_common = max(set(selected_tools), key=selected_tools.count)
        stability = selected_tools.count(most_common) / len(selected_tools)
        rows.append({
            "base_id": base_id,
            "category": category,
            "stability": stability,
            "n_variants": len(variants),
            "always_same_tool": stability == 1.0,
        })

    return pd.DataFrame(rows)


def analyze_position_of_correct_tool(results: list) -> dict:
    """
    For mixed-category examples where we know the correct category,
    analyze if the semantically correct tool is selected more or less
    depending on its position.
    """
    # Look at which tools were selected at each position in the original list
    position_sims = {i: [] for i in range(4)}

    for r in results:
        sel_pos = r["selected_position"]
        if 0 <= sel_pos <= 3:
            position_sims[sel_pos].append(1)  # Selected
        for pos in range(4):
            if pos != sel_pos and pos < len(r["tool_order"]):
                position_sims[pos].append(0)  # Not selected

    return {pos: np.mean(vals) for pos, vals in position_sims.items() if vals}


def run_positional_analysis(results_path: str, output_dir: str) -> dict:
    """Full positional bias analysis pipeline."""
    results = load_positional_results(results_path)

    print("\n=== Positional Bias Analysis ===")
    bias_stats = compute_positional_bias(results)

    print(f"Total selections: {bias_stats['total_examples']}")
    print(f"Selection rates by position:")
    for pos, rate in bias_stats["position_rates"].items():
        print(f"  Position {pos}: {rate:.3f} (expected: 0.250)")
    print(f"Position-0 rate: {bias_stats['position_0_rate']:.3f}")
    print(f"Chi-square statistic: {bias_stats['chi2_statistic']:.3f}")
    print(f"P-value: {bias_stats['chi2_pvalue']:.4f}")
    print(f"Total variation distance: {bias_stats['total_variation_distance']:.3f}")
    print(f"Mean stability (same tool across rotations): {bias_stats['mean_stability']:.3f}")

    # Significance interpretation
    if bias_stats['chi2_pvalue'] < 0.05:
        print("=> SIGNIFICANT positional bias (p < 0.05)")
    else:
        print("=> No significant positional bias detected")

    # Stability analysis
    stab_df = analyze_stability_by_category(results)
    print("\nStability by category:")
    if not stab_df.empty:
        cat_stab = stab_df.groupby("category")["stability"].mean()
        print(cat_stab.to_string())

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save full stats (excluding non-serializable data for now)
    stats_to_save = {k: v for k, v in bias_stats.items() if k != "stability_per_example"}
    with open(f"{output_dir}/positional_bias_stats.json", "w") as f:
        json.dump(stats_to_save, f, indent=2)

    stab_df.to_csv(f"{output_dir}/stability_by_category.csv", index=False)

    print(f"\nSaved positional analysis to {output_dir}")
    return bias_stats


if __name__ == "__main__":
    run_positional_analysis(
        "results/positional_bias_results.json",
        "results/positional_analysis"
    )
