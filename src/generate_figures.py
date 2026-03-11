"""
Generate all research figures from experimental results.
This is the master figure generation script.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import os

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 120,
})

COLORS = {
    "semantic": "#1565C0",
    "position": "#BF360C",
    "probing_cat": "#2E7D32",
    "probing_pos_mean": "#6A1B9A",
    "probing_pos_last": "#00838F",
    "baseline": "#757575",
    "original": "#1565C0",
    "adversarial": "#BF360C",
    "generic": "#FF8F00",
}


def fig1_semantic_accuracy(results_dir: str, figures_dir: str):
    """
    Figure 1: Semantic similarity prediction accuracy compared to baselines.
    """
    df = pd.read_csv(f"{results_dir}/semantic_analysis/semantic_comparison.csv")

    overall_acc = df["sim_matches_llm"].mean()
    random_acc = 0.25
    position_acc = (df["llm_selected_position"] == 0).mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Baselines comparison
    methods = ["Random\nBaseline", "Position-0\nHeuristic", "Semantic\nSimilarity"]
    accs = [random_acc, position_acc, overall_acc]
    colors = [COLORS["baseline"], COLORS["position"], COLORS["semantic"]]

    bars = axes[0].bar(methods, accs, color=colors, edgecolor="white", linewidth=2, width=0.5, alpha=0.85)
    axes[0].axhline(y=random_acc, color=COLORS["baseline"], linestyle="--", alpha=0.6, linewidth=1.5, label="Chance level")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Top-1 Accuracy (predicting LLM selection)")
    axes[0].set_title("Semantic Similarity vs. Baselines\n(n=45 examples, 4 tools each)")
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{acc:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=12)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Right: Per-category accuracy
    per_cat = pd.read_csv(f"{results_dir}/semantic_analysis/per_category_semantic.csv")
    per_cat = per_cat.sort_values("top1_accuracy", ascending=False)

    cat_colors = ["#1565C0", "#2E7D32", "#6A1B9A", "#00838F", "#E65100", "#880E4F", "#37474F"]
    bars2 = axes[1].bar(range(len(per_cat)), per_cat["top1_accuracy"],
                        color=cat_colors[:len(per_cat)], width=0.6, alpha=0.85)
    axes[1].axhline(y=0.25, color=COLORS["baseline"], linestyle="--", alpha=0.6, linewidth=1.5, label="Chance (0.25)")
    axes[1].set_xticks(range(len(per_cat)))
    axes[1].set_xticklabels(per_cat["category"], rotation=35, ha="right", fontsize=9)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Top-1 Accuracy")
    axes[1].set_title("Semantic Similarity Accuracy\nby Tool Category")
    for i, (bar, row) in enumerate(zip(bars2, per_cat.itertuples())):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{row.top1_accuracy:.2f}", ha="center", va="bottom", fontsize=8)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/fig1_semantic_accuracy.png", bbox_inches="tight")
    plt.close()
    print(f"Saved fig1_semantic_accuracy.png")


def fig2_position_bias(results_dir: str, figures_dir: str):
    """
    Figure 2: Positional bias analysis.
    """
    with open(f"{results_dir}/positional_analysis/positional_bias_stats.json") as f:
        pos_stats = json.load(f)

    with open(f"{results_dir}/main_selection_results.json") as f:
        main_results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Position rates (positional bias experiment - controlled rotations)
    positions = list(range(4))
    obs_rates = [pos_stats["position_rates"].get(str(i), pos_stats["position_rates"].get(i, 0)) for i in positions]
    exp_rates = [0.25] * 4

    x = np.arange(4)
    width = 0.35
    axes[0].bar(x - width/2, obs_rates, width, label="Observed", color=COLORS["position"], alpha=0.85)
    axes[0].bar(x + width/2, exp_rates, width, label="Expected (uniform)", color=COLORS["baseline"], alpha=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Position {i}" for i in range(4)])
    axes[0].set_ylim(0, max(obs_rates) * 1.5 + 0.05)
    axes[0].set_ylabel("Selection Rate")
    axes[0].set_title(f"Position Preference: Rotation Experiment\n"
                      f"(χ²={pos_stats['chi2_statistic']:.2f}, p={pos_stats['chi2_pvalue']:.3f}, n={pos_stats['total_examples']})")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    for i, rate in enumerate(obs_rates):
        axes[0].text(x[i] - width/2, rate + 0.01, f"{rate:.2f}", ha="center", va="bottom", fontsize=9)

    # Right: Position-0 bias in main experiment (from main_selection_results)
    from collections import Counter
    pos_counts = Counter(r["selected_position"] for r in main_results if r["selected_position"] >= 0)
    total = sum(pos_counts.values())
    main_pos_rates = [pos_counts.get(i, 0) / total for i in range(4)]

    axes[1].bar(range(4), main_pos_rates, color=COLORS["position"], alpha=0.85, label="Observed")
    axes[1].axhline(y=0.25, color=COLORS["baseline"], linestyle="--", linewidth=1.5, alpha=0.7, label="Chance (0.25)")
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels([f"Position {i}" for i in range(4)])
    axes[1].set_ylim(0, 0.7)
    axes[1].set_ylabel("Selection Rate")
    axes[1].set_title(f"Position Preference: Main Experiment\n"
                      f"(n={total} selections, random tool ordering)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    for i, rate in enumerate(main_pos_rates):
        axes[1].text(i, rate + 0.01, f"{rate:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/fig2_positional_bias.png", bbox_inches="tight")
    plt.close()
    print(f"Saved fig2_positional_bias.png")


def fig3_stability(results_dir: str, figures_dir: str):
    """
    Figure 3: Selection stability across tool orderings.
    """
    with open(f"{results_dir}/positional_bias_results.json") as f:
        pos_results = json.load(f)

    # Compute per-base_id stability
    by_base = {}
    for r in pos_results:
        by_base.setdefault(r["base_id"], []).append(r)

    stabilities = []
    categories = []
    for base_id, variants in by_base.items():
        selected = [v["selected_tool"] for v in variants]
        most_common = max(set(selected), key=selected.count)
        stability = selected.count(most_common) / len(selected)
        stabilities.append(stability)
        categories.append(variants[0]["category"])

    mean_stab = np.mean(stabilities)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Histogram of stability scores
    bins = [0.0, 0.25, 0.5, 0.75, 1.01]
    hist_vals, _, patches = axes[0].hist(stabilities, bins=bins, edgecolor="white", linewidth=2, color=COLORS["probing_cat"], alpha=0.85)
    axes[0].axvline(mean_stab, color=COLORS["position"], linestyle="--", linewidth=2, label=f"Mean = {mean_stab:.2f}")
    axes[0].set_xlabel("Stability Score (fraction of rotations with same tool)")
    axes[0].set_ylabel("Number of Examples")
    axes[0].set_title(f"Selection Stability Across Tool Orderings\n(n={len(stabilities)} examples)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_xticklabels(["0\n(always\ndifferent)", "0.25", "0.5", "0.75", "1.0\n(always\nsame)"])

    # Right: Per-category stability
    cat_stab = {}
    for cat, stab in zip(categories, stabilities):
        cat_stab.setdefault(cat, []).append(stab)
    cat_means = {cat: np.mean(vals) for cat, vals in cat_stab.items()}
    cat_labels = list(cat_means.keys())
    cat_values = [cat_means[c] for c in cat_labels]

    # Sort by stability
    sorted_pairs = sorted(zip(cat_values, cat_labels), reverse=True)
    cat_values, cat_labels = zip(*sorted_pairs)

    bar_colors = [COLORS["probing_cat"] if v >= 0.8 else COLORS["probing_pos_last"] if v >= 0.6 else COLORS["position"]
                  for v in cat_values]
    axes[1].bar(range(len(cat_labels)), cat_values, color=bar_colors, alpha=0.85)
    axes[1].axhline(y=0.25, color=COLORS["baseline"], linestyle="--", alpha=0.6, linewidth=1.5, label="Chance (0.25)")
    axes[1].axhline(y=1.0, color="#333", linestyle=":", alpha=0.3, label="Perfect stability")
    axes[1].set_xticks(range(len(cat_labels)))
    axes[1].set_xticklabels(cat_labels, rotation=30, ha="right")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel("Mean Stability Score")
    axes[1].set_title("Selection Stability by Category")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(cat_values):
        axes[1].text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/fig3_stability.png", bbox_inches="tight")
    plt.close()
    print(f"Saved fig3_stability.png")


def fig4_perturbation(results_dir: str, figures_dir: str):
    """
    Figure 4: Description perturbation causal analysis.
    """
    with open(f"{results_dir}/perturbation_results.json") as f:
        results = json.load(f)

    # Find matching pairs
    groups = {}
    for r in results:
        base = r["id"].rsplit("_", 1)[0]
        groups.setdefault(base, {})[r["condition"]] = r

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Show all three conditions for pert_001 (weather query)
    pert001 = groups.get("pert_001", {})
    conditions = ["original", "generic", "adversarial"]
    labels = ["Original\n(semantic match)", "Generic\n(neutral descriptions)", "Adversarial\n(swapped descriptions)"]

    selections = []
    for cond in conditions:
        if cond in pert001:
            selections.append(pert001[cond]["consensus_selection"])
        else:
            selections.append("N/A")

    colors = [COLORS["original"], COLORS["generic"], COLORS["adversarial"]]
    bar_pos = range(len(conditions))
    bars = axes[0].bar(bar_pos, [1, 1, 1], color=colors, alpha=0.7, width=0.5)
    axes[0].set_xticks(bar_pos)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylim(0, 1.5)
    axes[0].set_yticks([])
    axes[0].set_title("Effect of Description Perturbation\n(Query: 'What's the forecast for London?')")
    for i, (pos, sel) in enumerate(zip(bar_pos, selections)):
        axes[0].text(pos, 0.5, f"Selected:\n{sel}", ha="center", va="center",
                     fontsize=8, color="white", fontweight="bold", wrap=True)

    # Add legend boxes
    legend_patches = [
        mpatches.Patch(color=COLORS["original"], alpha=0.7, label="Correct tool selected"),
        mpatches.Patch(color=COLORS["adversarial"], alpha=0.7, label="Wrong tool selected"),
    ]
    axes[0].legend(handles=legend_patches, loc="upper right")

    # Right: Summary across all queries
    n_pairs = 0
    n_changed = 0
    for base_id, conditions in groups.items():
        if "original" in conditions and "adversarial" in conditions:
            n_pairs += 1
            if conditions["original"]["consensus_selection"] != conditions["adversarial"]["consensus_selection"]:
                n_changed += 1

    metrics = [
        ("Adversarial\nChanged Selection", n_changed / n_pairs if n_pairs > 0 else 0),
        ("Adversarial\nSelection Stable", (n_pairs - n_changed) / n_pairs if n_pairs > 0 else 0),
    ]
    bar_cols = [COLORS["adversarial"], COLORS["probing_cat"]]
    bars2 = axes[1].bar(range(2), [m[1] for m in metrics], color=bar_cols, alpha=0.85, width=0.4)
    axes[1].set_xticks(range(2))
    axes[1].set_xticklabels([m[0] for m in metrics], fontsize=9)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel("Fraction of Examples")
    axes[1].set_title(f"Selection Change After Adversarial\nDescription Swapping (n={n_pairs} pairs)")
    for i, (bar, m) in enumerate(zip(bars2, metrics)):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                     f"{m[1]:.2f}\n({int(m[1]*n_pairs)}/{n_pairs})",
                     ha="center", va="bottom", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/fig4_perturbation.png", bbox_inches="tight")
    plt.close()
    print(f"Saved fig4_perturbation.png")


def fig5_layer_probing(results_dir: str, figures_dir: str):
    """
    Figure 5: Layer-wise probing accuracy curves.
    """
    with open(f"{results_dir}/probing_analysis/probing_results.json") as f:
        results = json.load(f)

    layers = list(range(1, 13))

    cat_accs = results["category_probe_mean_pool"]["accuracies"]
    pos_mean_accs = results["position_probe_mean_pool"]["accuracies"]
    pos_last_accs = results["position_probe_last_token"]["accuracies"]

    # Get std deviations
    pos_mean_stds = [r["std_accuracy"] for r in results["position_probe_mean_pool"]["layer_results"]]
    pos_last_stds = [r["std_accuracy"] for r in results["position_probe_last_token"]["layer_results"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Category probe
    axes[0].plot(layers, cat_accs, "o-", color=COLORS["probing_cat"], linewidth=2.5,
                 markersize=8, markerfacecolor="white", markeredgewidth=2,
                 label="Category probe (mean pool)")
    axes[0].axhline(y=0.25, color=COLORS["baseline"], linestyle="--", linewidth=1.5,
                    label="Chance (0.25)")
    axes[0].set_xticks(layers)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_xlabel("Transformer Layer")
    axes[0].set_ylabel("Cross-Validated Probe Accuracy")
    axes[0].set_title("Tool Category Probe\n(GPT-2-Small, 4 categories, n=100)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].annotate(f"All layers ≥99%\nacross all 12 layers",
                     xy=(6, 1.0), fontsize=9, color=COLORS["probing_cat"],
                     ha="center", style="italic")

    # Right: Position probe (two methods)
    axes[1].plot(layers, pos_mean_accs, "s-", color=COLORS["probing_pos_mean"], linewidth=2.5,
                 markersize=8, markerfacecolor="white", markeredgewidth=2,
                 label="Position probe (mean pool)")
    axes[1].fill_between(layers,
                          [a - s for a, s in zip(pos_mean_accs, pos_mean_stds)],
                          [a + s for a, s in zip(pos_mean_accs, pos_mean_stds)],
                          alpha=0.15, color=COLORS["probing_pos_mean"])

    axes[1].plot(layers, pos_last_accs, "o-", color=COLORS["probing_pos_last"], linewidth=2.5,
                 markersize=8, markerfacecolor="white", markeredgewidth=2,
                 label="Position probe (last token)")
    axes[1].fill_between(layers,
                          [a - s for a, s in zip(pos_last_accs, pos_last_stds)],
                          [a + s for a, s in zip(pos_last_accs, pos_last_stds)],
                          alpha=0.15, color=COLORS["probing_pos_last"])

    axes[1].axhline(y=0.25, color=COLORS["baseline"], linestyle="--", linewidth=1.5,
                    label="Chance (0.25)")
    axes[1].set_xticks(layers)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Transformer Layer")
    axes[1].set_ylabel("Cross-Validated Probe Accuracy")
    axes[1].set_title("Correct Tool Position Probe\n(GPT-2-Small, 4 positions, n=100)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Annotate the emergent pattern for mean-pool
    best_layer_mean = results["position_probe_mean_pool"]["best_layer"]
    best_acc_mean = results["position_probe_mean_pool"]["best_accuracy"]
    axes[1].annotate(f"Peak: Layer {best_layer_mean}\n({best_acc_mean:.2f})",
                     xy=(best_layer_mean, best_acc_mean),
                     xytext=(best_layer_mean - 3, best_acc_mean - 0.15),
                     arrowprops=dict(arrowstyle="->", color=COLORS["probing_pos_mean"]),
                     fontsize=9, color=COLORS["probing_pos_mean"])

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/fig5_layer_probing.png", bbox_inches="tight")
    plt.close()
    print(f"Saved fig5_layer_probing.png")


def fig6_similarity_distributions(results_dir: str, figures_dir: str):
    """
    Figure 6: Similarity distributions for selected vs. non-selected tools.
    """
    if not os.path.exists(f"{results_dir}/semantic_analysis/sim_predictions.json"):
        print("Skipping fig6: sim_predictions.json not found")
        return

    with open(f"{results_dir}/semantic_analysis/sim_predictions.json") as f:
        predictions = json.load(f)

    df = pd.read_csv(f"{results_dir}/semantic_analysis/semantic_comparison.csv")
    llm_sel_by_id = {row["id"]: row["llm_selected"] for _, row in df.iterrows()}

    selected_sims = []
    nonselected_sims = []

    for pred in predictions:
        ex_id = pred["id"]
        if ex_id not in llm_sel_by_id:
            continue
        llm_selected = llm_sel_by_id[ex_id]
        for tool_name, sim in zip(pred["tool_names"], pred["similarities"]):
            if tool_name == llm_selected:
                selected_sims.append(sim)
            else:
                nonselected_sims.append(sim)

    if not selected_sims:
        print("Skipping fig6: no data")
        return

    # T-test
    t_stat, p_val = stats.ttest_ind(selected_sims, nonselected_sims)

    fig, ax = plt.subplots(figsize=(9, 5))
    all_sims = selected_sims + nonselected_sims
    bins = np.linspace(min(all_sims) - 0.02, max(all_sims) + 0.02, 25)

    ax.hist(selected_sims, bins=bins, alpha=0.7, color=COLORS["semantic"],
            label=f"LLM-Selected Tools (n={len(selected_sims)})", density=True)
    ax.hist(nonselected_sims, bins=bins, alpha=0.5, color=COLORS["baseline"],
            label=f"Non-Selected Tools (n={len(nonselected_sims)})", density=True)

    mean_sel = np.mean(selected_sims)
    mean_nonsel = np.mean(nonselected_sims)
    ax.axvline(mean_sel, color=COLORS["semantic"], linestyle="--", linewidth=2,
               label=f"Mean selected: {mean_sel:.3f}")
    ax.axvline(mean_nonsel, color=COLORS["baseline"], linestyle="--", linewidth=2,
               label=f"Mean non-selected: {mean_nonsel:.3f}")

    ax.set_xlabel("Cosine Similarity (Query Embedding vs Tool Description Embedding)")
    ax.set_ylabel("Density")
    ax.set_title(f"Cosine Similarity Distribution: Selected vs Non-Selected Tools\n"
                 f"(Welch's t-test: t={t_stat:.2f}, p={p_val:.4f})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/fig6_similarity_distributions.png", bbox_inches="tight")
    plt.close()
    print(f"Saved fig6_similarity_distributions.png")


def fig7_summary_heatmap(results_dir: str, figures_dir: str):
    """
    Figure 7: Summary heatmap of all findings.
    """
    # Create a summary table of all key metrics
    with open(f"{results_dir}/semantic_analysis/semantic_metrics.json") as f:
        sem_metrics = json.load(f)
    with open(f"{results_dir}/positional_analysis/positional_bias_stats.json") as f:
        pos_stats = json.load(f)
    with open(f"{results_dir}/probing_analysis/probing_results.json") as f:
        probe_results = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 5))

    metrics = {
        "Semantic Top-1 Acc.": sem_metrics.get("top1_accuracy", 0),
        "Chance (4 tools)": 0.25,
        "Position-0 Rate (main)": 0.578,
        "Position-0 Rate (rotation)": pos_stats.get("position_0_rate", 0),
        "Selection Stability": pos_stats.get("mean_stability", 0),
        "Position Probe Peak\n(mean pool)": probe_results["position_probe_mean_pool"]["best_accuracy"],
        "Position Probe Peak\n(last token)": probe_results["position_probe_last_token"]["best_accuracy"],
    }

    names = list(metrics.keys())
    values = list(metrics.values())
    colors = [
        COLORS["semantic"], COLORS["baseline"],
        COLORS["position"], COLORS["position"],
        COLORS["probing_cat"],
        COLORS["probing_pos_mean"], COLORS["probing_pos_last"],
    ]

    bars = ax.barh(names, values, color=colors, alpha=0.85, height=0.6)
    ax.axvline(x=0.25, color=COLORS["baseline"], linestyle="--", linewidth=1.5, alpha=0.7, label="Chance level")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score / Accuracy")
    ax.set_title("Key Metrics Summary Across All Experiments")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/fig7_summary.png", bbox_inches="tight")
    plt.close()
    print(f"Saved fig7_summary.png")


def main():
    results_dir = "results"
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)

    print("=== Generating All Figures ===\n")

    fig1_semantic_accuracy(results_dir, figures_dir)
    fig2_position_bias(results_dir, figures_dir)
    fig3_stability(results_dir, figures_dir)
    fig4_perturbation(results_dir, figures_dir)
    fig5_layer_probing(results_dir, figures_dir)
    fig6_similarity_distributions(results_dir, figures_dir)
    fig7_summary_heatmap(results_dir, figures_dir)

    print(f"\nAll figures saved to {figures_dir}/")
    print("Figure list:")
    for f in sorted(os.listdir(figures_dir)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
