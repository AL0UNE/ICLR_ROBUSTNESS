#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


OUTPUT_DIR = Path("figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

color_map = {True: "#2b5c8f", False: "#d95f02"}  # HPO: blue, No HPO: orange

def plot_and_save_elo_comparison(datasets, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=120, sharey=False)

    for ax, config in zip(axes, datasets):
        # 1. Parse and deduplicate
        df = pd.read_csv(config["filepath"])
        df = df.drop_duplicates(subset=["Model"]).copy()
        df = df[~(df["Model"] == config["anchor"] + "_hpo")].copy()

        # 2. Flag HPO vs non-HPO
        df["HPO"] = df["Model"].str.endswith("_hpo")

        # 3. Dynamic metric column names
        median_col = f"{config['metric_col']}_bootstrap_median"
        ci_lower_col = f"{config['metric_col']}_ci_lower"
        ci_upper_col = f"{config['metric_col']}_ci_upper"

        # 4. Sort ascending so highest values appear at the top
        df = df.sort_values(by=median_col, ascending=True).reset_index(drop=True)

        # 5. Asymmetric confidence intervals
        lower_err = df[median_col] - df[ci_lower_col]
        upper_err = df[ci_upper_col] - df[median_col]
        xerr = np.array([lower_err, upper_err])

        # 6. Plot bars
        colors = df["HPO"].map(color_map)
        y_positions = np.arange(len(df))

        ax.barh(
            y_positions,
            df[median_col],
            xerr=xerr,
            capsize=4,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
            error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
        )

        # 7. Styling and labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(df["Model"], fontsize=10)
        ax.set_xlabel(config["xlabel"], fontsize=11, fontweight="bold")
        ax.set_title(config["subtitle"], fontsize=13, fontweight="bold", pad=12)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        # Subplot legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color_map[True], edgecolor="black", label="With HPO"),
            plt.Rectangle((0, 0), 1, 1, facecolor=color_map[False], edgecolor="black", label="Without HPO"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", frameon=True)

    plt.tight_layout()

    # Save at high resolution (bbox_inches="tight" avoids cutting off the suptitle)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")

    plt.show()



# -------------------------------------------------------------------------
# 1. Global Elo Comparison
# -------------------------------------------------------------------------
datasets_global = [
    {
        "filepath": "../result/elo_table_classif.csv",
        "metric_col": "elo_global_auc",
        "subtitle": "Classification",
        "xlabel": "Global Robustness Elo",
        "anchor": "Logistic",
    },
    {
        "filepath": "../result/elo_table_reg.csv",
        "metric_col": "elo_global_mae",
        "subtitle": "Regression",
        "xlabel": "Global Robustness Elo",
        "anchor": "Linear",
    },
]

plot_and_save_elo_comparison(
    datasets=datasets_global,
    output_path="figures/elo_global_comparison.pdf",
)

# -------------------------------------------------------------------------
# 2. Clean Elo Comparison
# -------------------------------------------------------------------------
datasets_clean = [
    {
        "filepath": "../results/elo_table_classif.csv",
        "metric_col": "elo_clean_auc",
        "subtitle": "Classification",
        "xlabel": "Clean Performance Elo",
        "anchor": "Logistic",
    },
    {
        "filepath": "../results/elo_table_reg.csv",
        "metric_col": "elo_clean_mae",
        "subtitle": "Regression",
        "xlabel": "Clean Performance Elo",
        "anchor": "Linear",
    },
]


plot_and_save_elo_comparison(
    datasets=datasets_clean,
    output_path="figures/elo_clean_comparison.pdf",
)

