"""
Visualization module for tool selection analysis results.
Creates all figures for the final report.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import os


# Style settings
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 100,
    "figure.figsize": (8, 5),
})

COLORS = {
    "semantic": "#2196F3",  # Blue
    "position": "#FF5722",  # Deep Orange
    "probing": "#4CAF50",   # Green
    "baseline": "#9E9E9E",  # Grey
    "adversarial": "#F44336",  # Red
    "original": "#2196F3",  # Blue
}

CATEGORY_COLORS = {
    "weather": "#03A9F4",
    "calculator": "#FF9800",
    "search": "#9C27B0",
    "translation": "#4CAF50",
    "code_execution": "#F44336",
    "ambiguous_search_translation": "#795548",
    "ambiguous_code_calculator": "#607D8B",
}


def plot_semantic_accuracy(comparison_csv: str, output_path: str):
    """
    Figure 1: Semantic similarity prediction accuracy vs baselines.
    Bar chart comparing: random, position-0 baseline, semantic similarity.
    """
    df = pd.read_csv(comparison_csv)

    # Compute metrics
    overall_acc = df["sim_matches_llm"].mean()
    random_acc = 1.0 / 4  # 4 tools
    position_acc = (df["llm_selected_position"] == 0).mean()

    # Per-scenario type
    same_acc = df[df["scenario_type"] == "same_category"]["sim_matches_llm"].mean()
    mixed_acc = df[df["scenario_type"] == "mixed_category"]["sim_matches_llm"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Baselines comparison
    methods = ["Random\n(Baseline)", "Position-0\n(Baseline)", "Semantic\nSimilarity"]
    accs = [random_acc, position_acc, overall_acc]
    bar_colors = [COLORS["baseline"], COLORS["position"], COLORS["semantic"]]

    bars = axes[0].bar(methods, accs, color=bar_colors, edgecolor="white", linewidth=1.5, width=0.5)
    axes[0].axhline(y=random_acc, color=COLORS["baseline"], linestyle="--", alpha=0.5, label="Chance level")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Top-1 Accuracy (vs LLM selection)")
    axes[0].set_title("Semantic Similarity vs Baselines")
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{acc:.2f}", ha="center", va="bottom", fontweight="bold")
    axes[0].legend()

    # Right: Per-scenario type
    scenario_accs = []
    scenario_labels = []
    if not np.isnan(same_acc):
        scenario_accs.append(same_acc)
        scenario_labels.append("Same\nCategory")
    if not np.isnan(mixed_acc):
        scenario_accs.append(mixed_acc)
        scenario_labels.append("Mixed\nCategory")

    if scenario_accs:
        axes[1].bar(scenario_labels, scenario_accs, color=[COLORS["semantic"], COLORS["probing"]], width=0.4)
        axes[1].axhline(y=random_acc, color=COLORS["baseline"], linestyle="--", alpha=0.5, label="Chance level")
        axes[1].set_ylim(0, 1.0)
        axes[1].set_ylabel("Top-1 Accuracy")
        axes[1].set_title("Accuracy by Scenario Type")
        for i, acc in enumerate(scenario_accs):
            axes[1].text(i, acc + 0.02, f"{acc:.2f}", ha="center", va="bottom", fontweight="bold")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_category_accuracy(per_cat_csv: str, output_path: str):
    """
    Figure 2: Semantic accuracy per tool category.
    """
    df = pd.read_csv(per_cat_csv)

    if df.empty:
        print("Skipping per-category plot: no data")
        return

    # Sort by accuracy
    df = df.sort_values("top1_accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))

    bar_colors = [CATEGORY_COLORS.get(cat, "#607D8B") for cat in df["category"]]
    bars = ax.bar(range(len(df)), df["top1_accuracy"], color=bar_colors, edgecolor="white", linewidth=1)
    ax.axhline(y=0.25, color=COLORS["baseline"], linestyle="--", alpha=0.7, label="Chance (0.25)")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["category"], rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Top-1 Accuracy (Semantic Similarity)")
    ax.set_title("Tool Selection Prediction Accuracy by Category\n(Semantic Similarity vs LLM Ground Truth)")

    for i, (bar, row) in enumerate(zip(bars, df.itertuples())):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{row.top1_accuracy:.2f}", ha="center", va="bottom", fontsize=9)

    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_positional_bias(positional_stats_path: str, output_path: str):
    """
    Figure 3: Positional bias analysis.
    """
    with open(positional_stats_path) as f:
        stats = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Selection rates by position
    positions = list(range(4))
    observed_rates = [stats["position_rates"].get(i, 0) for i in positions]
    expected_rates = [0.25] * 4

    x = np.arange(4)
    width = 0.35
    bars1 = axes[0].bar(x - width/2, observed_rates, width, label="Observed", color=COLORS["position"], alpha=0.8)
    bars2 = axes[0].bar(x + width/2, expected_rates, width, label="Expected (uniform)", color=COLORS["baseline"], alpha=0.5)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Position {i}\n(nth tool)" for i in range(4)])
    axes[0].set_ylim(0, max(observed_rates) * 1.3 + 0.1)
    axes[0].set_ylabel("Selection Rate")
    axes[0].set_title(f"Tool Selection Rate by List Position\n(χ²={stats['chi2_statistic']:.2f}, p={stats['chi2_pvalue']:.3f})")
    axes[0].legend()

    for bar, rate in zip(bars1, observed_rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{rate:.2f}", ha="center", va="bottom", fontsize=9)

    # Right: Selection stability across rotations
    # Show distribution of stability scores
    stability_mean = stats.get("mean_stability", 0)
    stability_label = f"Mean stability\nacross rotations\n= {stability_mean:.2f}"

    axes[1].bar(
        ["Selection\nStability", "Position\n0 Preference"],
        [stability_mean, stats["position_0_rate"]],
        color=[COLORS["probing"], COLORS["position"]],
        width=0.4,
    )
    axes[1].axhline(y=0.25, color=COLORS["baseline"], linestyle="--", alpha=0.7, label="Chance (0.25)")
    axes[1].axhline(y=1.0, color="#333", linestyle=":", alpha=0.3, label="Perfect stability")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel("Rate / Score")
    axes[1].set_title("Stability and Positional Preference\nAcross Tool Orderings")
    axes[1].legend()
    for i, v in enumerate([stability_mean, stats["position_0_rate"]]):
        axes[1].text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_description_perturbation(perturbation_results_path: str, output_path: str):
    """
    Figure 4: Description perturbation analysis.
    Compare selections under original vs. adversarial descriptions.
    """
    with open(perturbation_results_path) as f:
        results = json.load(f)

    # Group by query (base_id from the format pert_XXX_condition)
    groups = {}
    for r in results:
        base = r["id"].rsplit("_", 1)[0]
        groups.setdefault(base, {})[r["condition"]] = r

    # For each query, compare original vs adversarial selection
    comparisons = []
    for base_id, conditions in groups.items():
        if "original" in conditions and "adversarial" in conditions:
            orig_sel = conditions["original"]["consensus_selection"]
            adv_sel = conditions["adversarial"]["consensus_selection"]
            selection_changed = (orig_sel != adv_sel)

            # In original, does the semantically correct tool match?
            orig_tools = conditions["original"]["tool_names"]

            comparisons.append({
                "base_id": base_id,
                "original_selection": orig_sel,
                "adversarial_selection": adv_sel,
                "selection_changed": selection_changed,
                "orig_consistency": conditions["original"].get("consistency", 1.0),
                "adv_consistency": conditions["adversarial"].get("consistency", 1.0),
            })

    if not comparisons:
        print("Skipping perturbation plot: no comparison pairs found")
        return

    n_changed = sum(1 for c in comparisons if c["selection_changed"])
    n_total = len(comparisons)

    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["Selection\nChanged", "Selection\nStable"]
    values = [n_changed / n_total, (n_total - n_changed) / n_total]
    colors = [COLORS["adversarial"], COLORS["semantic"]]

    bars = ax.bar(categories, values, color=colors, width=0.4, edgecolor="white")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Fraction of Examples")
    ax.set_title("Effect of Adversarial Description Perturbation\non Tool Selection")

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{v:.2f}\n({int(v*n_total)}/{n_total})", ha="center", va="bottom", fontweight="bold")

    ax.text(0.5, 0.95, f"n={n_total} query pairs tested", transform=ax.transAxes,
            ha="center", va="top", style="italic", color="gray")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_layer_probing(probing_results_path: str, output_path: str):
    """
    Figure 5: Layer-wise probing accuracy curve.
    """
    with open(probing_results_path) as f:
        results = json.load(f)

    accuracies = results["accuracies_by_layer"]
    n_layers = len(accuracies)
    layers = list(range(1, n_layers + 1))

    # Get std deviations if available
    stds = []
    for i in range(n_layers):
        layer_res = results["layer_results"].get(str(i), results["layer_results"].get(i, {}))
        stds.append(layer_res.get("std_accuracy", 0))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(layers, accuracies, "o-", color=COLORS["probing"], linewidth=2.5,
            markersize=8, markerfacecolor="white", markeredgewidth=2, label="Linear Probe Accuracy")

    if stds:
        ax.fill_between(layers,
                         [a - s for a, s in zip(accuracies, stds)],
                         [a + s for a, s in zip(accuracies, stds)],
                         alpha=0.2, color=COLORS["probing"])

    ax.axhline(y=results["chance_accuracy"], color=COLORS["baseline"], linestyle="--",
               linewidth=1.5, label=f"Chance ({results['chance_accuracy']:.2f})")
    ax.axhline(y=results["baseline_accuracy"], color="#FF9800", linestyle=":",
               linewidth=1.5, label=f"Majority class ({results['baseline_accuracy']:.2f})")

    # Highlight early/mid/late regions
    ax.axvspan(1, 4, alpha=0.05, color="blue", label="Early layers (1-4)")
    ax.axvspan(5, 8, alpha=0.05, color="green", label="Middle layers (5-8)")
    ax.axvspan(9, 12, alpha=0.05, color="red", label="Late layers (9-12)")

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Cross-Validated Probe Accuracy")
    ax.set_title("Layer-wise Linear Probing for Tool Category\n(GPT-2-Small, 12 layers)")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Annotate best layer
    best_layer = results["best_layer"]
    best_acc = results["best_layer_accuracy"]
    ax.annotate(f"Best: Layer {best_layer}\n({best_acc:.2f})",
                xy=(best_layer, best_acc),
                xytext=(best_layer + 1, best_acc - 0.1),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_similarity_distribution(sim_predictions_path: str, comparison_csv: str, output_path: str):
    """
    Figure 6: Distribution of cosine similarities for selected vs. non-selected tools.
    """
    with open(sim_predictions_path) as f:
        predictions = json.load(f)

    df = pd.read_csv(comparison_csv)

    # Collect similarities of selected vs. non-selected tools
    selected_sims = []
    nonselected_sims = []

    # Map comparison df by id
    llm_sel_by_id = {row["id"]: row["llm_selected"] for _, row in df.iterrows()}

    for pred in predictions:
        ex_id = pred["id"]
        if ex_id not in llm_sel_by_id:
            continue

        llm_selected = llm_sel_by_id[ex_id]
        tool_names = pred["tool_names"]
        sims = pred["similarities"]

        for tool_name, sim in zip(tool_names, sims):
            if tool_name == llm_selected:
                selected_sims.append(sim)
            else:
                nonselected_sims.append(sim)

    if not selected_sims or not nonselected_sims:
        print("Skipping similarity distribution plot: insufficient data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(min(selected_sims + nonselected_sims) - 0.05,
                       max(selected_sims + nonselected_sims) + 0.05, 20)

    ax.hist(selected_sims, bins=bins, alpha=0.7, color=COLORS["semantic"], label=f"LLM-Selected Tools\n(n={len(selected_sims)})", density=True)
    ax.hist(nonselected_sims, bins=bins, alpha=0.5, color=COLORS["baseline"], label=f"Non-Selected Tools\n(n={len(nonselected_sims)})", density=True)

    ax.axvline(np.mean(selected_sims), color=COLORS["semantic"], linestyle="--", linewidth=2,
               label=f"Mean selected: {np.mean(selected_sims):.3f}")
    ax.axvline(np.mean(nonselected_sims), color=COLORS["baseline"], linestyle="--", linewidth=2,
               label=f"Mean non-selected: {np.mean(nonselected_sims):.3f}")

    ax.set_xlabel("Cosine Similarity (Query-Tool Description)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Cosine Similarities:\nLLM-Selected vs. Non-Selected Tools")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_figures(results_dir: str, figures_dir: str):
    """Generate all figures from experiment results."""
    os.makedirs(figures_dir, exist_ok=True)

    print("\n=== Generating Figures ===")

    # Figure 1: Semantic similarity accuracy
    sem_csv = f"{results_dir}/semantic_analysis/semantic_comparison.csv"
    if os.path.exists(sem_csv):
        plot_semantic_accuracy(sem_csv, f"{figures_dir}/fig1_semantic_accuracy.png")

    # Figure 2: Per-category accuracy
    cat_csv = f"{results_dir}/semantic_analysis/per_category_semantic.csv"
    if os.path.exists(cat_csv):
        plot_per_category_accuracy(cat_csv, f"{figures_dir}/fig2_per_category_accuracy.png")

    # Figure 3: Positional bias
    pos_stats = f"{results_dir}/positional_analysis/positional_bias_stats.json"
    if os.path.exists(pos_stats):
        plot_positional_bias(pos_stats, f"{figures_dir}/fig3_positional_bias.png")

    # Figure 4: Description perturbation
    pert_results = f"{results_dir}/perturbation_results.json"
    if os.path.exists(pert_results):
        plot_description_perturbation(pert_results, f"{figures_dir}/fig4_description_perturbation.png")

    # Figure 5: Layer-wise probing
    probe_results = f"{results_dir}/probing_analysis/probing_results.json"
    if os.path.exists(probe_results):
        plot_layer_probing(probe_results, f"{figures_dir}/fig5_layer_probing.png")

    # Figure 6: Similarity distribution
    sim_pred = f"{results_dir}/semantic_analysis/sim_predictions.json"
    if os.path.exists(sim_pred) and os.path.exists(sem_csv):
        plot_similarity_distribution(sim_pred, sem_csv, f"{figures_dir}/fig6_similarity_distribution.png")

    print(f"\nAll figures saved to {figures_dir}")


if __name__ == "__main__":
    generate_all_figures("results", "figures")
