#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

MODELS_COLOR = {
    # Linear models
    "Logistic": "#0072B2",
    "Linear": "#009E73",
    "Ridge": "#56B4E9",
    "LASSO": "#7A68A6",
    # Tree-based models
    "Random Forest": "#E69F00",
    "Gradient Boosting": "#D55E00",
    "XGBoost": "#F0E442",
    "LightGBM": "#6A994E",
    "CatBoost": "#0096A6",
    # Deep learning
    "RealMLP": "#CC79A7",
    # Foundation models
    "TabDPT": "#C44E52",
    "TabPFNv3": "#8C564B",
    "TabICLv2": "#4C4C4C",
}

MODEL_ORDER = [
    "Ridge",
    "LASSO",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "RealMLP",
]

# Fixed typo where missing comma concatenated "TabPFNv3" and "TabICL"

EXCLUDE_MODELS = {
    "TabPFNv3",
    "TabICLv2",
    "TabDPT",
    "Logistic",
    "Linear",
}

CONTINUOUS_PERTURBATIONS = {
    "Input Noise",
    "Imbalance Data",
    "Feature Shuffle",
    "Training Size",
    "Missing Data",
}

DATASET_DISPLAY_NAMES = {
    "M3": "MIMIC-III",
    "M4": "MIMIC-IV",
    "eICU": "eICU",
}

# Dynamic external dataset assignment for each training set
EXTERNAL_DATASETS = {
    "M3": ("eICU", "M4"),
    "M4": ("M3", "eICU"),
    "eICU": ("M3", "M4"),
}

# Task-to-outcome mapping
TASK_OUTCOMES = {
    "Classif": ["AUC", "Brier score"],
    "Reg": ["MAE", "RMSE", "R2"],
}

ALL_TRAIN_DFS = ["M3", "M4", "eICU"]
OUTPUT_DIR = Path("figures")



# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


icu_mapping = {

    # MIMIC-III
    "CCU": "CCU",
    "CSRU": "CSRU",
    "SICU": "SICU",
    "MICU": "MICU",
    "TSICU": "TSICU",
    
    # MIMIC-IV
    "MICU": "MICU",
    "SICU": "SICU",
    "Med-Surg ICU": "MED_SURG_ICU",
    "Cardiac ICU": "CICU",
    "CCU-CTICU": "CCU-CTICU",
    "CTICU": "CTICU",
    "CSICU": "CSICU",
    "Neuro ICU": "NEURO_ICU",    
    
    # eICU
    "Medical/Surgical Intensive Care Unit (MICU/SICU)": "MICU-SICU",
    "Cardiac Vascular Intensive Care Unit (CVICU)": "CVICU",
    "Coronary Care Unit (CCU)": "CCU",
    "Trauma SICU (TSICU)": "TSICU",
    "Neuro Intermediate": "NEURO_INTERMEDIATE",
    "Medical Intensive Care Unit (MICU)": "MICU",
    "Surgical Intensive Care Unit (SICU)": "SICU",
    "Neuro Stepdown": "NEURO_STEPPDOWN",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)": "NEURO_ICU",

}

excluded_icu = [c for c in icu_mapping.values() if c not in ['MICU', 'SICU']]



def clean_subgroups(df):
    """Maps variable and ICU names, filtering out non-overlapping categories."""
    df = df.copy()
    df['Category'] = df['Category'].replace(icu_mapping)

    df = df.loc[~((df['Variable'] == 'ICU type') & (df['Category'].isin(excluded_icu)))]
    return df


def normalize_ptype(ptype_raw: str) -> str:
    """Safely normalizes perturbation names without mangling acronyms like eICU, M3, M4."""
    custom_names = {
        "eicu": "eICU",
        "Eicu": "eICU",
        "eICU": "eICU",
        "m3": "M3",
        "m4": "M4",
    }
    return custom_names.get(ptype_raw, ptype_raw.replace("_", " ").title())


def parse_metadata(df: pd.DataFrame, ptype_raw: str, filename: str) -> pd.DataFrame:
    """Standardizes perturbation metadata, mechanisms, sets, and noise scaling."""
    df = df.copy()
    ptype = normalize_ptype(ptype_raw)
    df["Perturbation type"] = ptype

    # Extract Set
    fname_upper = filename.upper()
    if "TRAIN" in fname_upper:
        df["Set"] = "Train"
    elif "VAL" in fname_upper:
        df["Set"] = "Validation"
    elif "ALL" in fname_upper:
        df["Set"] = "All"
    else:
        df["Set"] = "N/A"

    # Extract Mechanism
    parts = filename.replace(".csv", "").split("_")
    if ptype_raw == "missing_data":
        df["Mechanism"] = parts[0].upper()
    elif ptype_raw == "input_noise":
        if len(parts) >= 4 and parts[3] in ["CONT", "CAT"]:
            df["Mechanism"] = parts[3].replace("CONT", "Continuous").replace("CAT", "Categorical")
        else:
            df["Mechanism"] = "All"
    elif ptype_raw == "label_noise":
        label_type = parts[0].upper()
        label_map = {
            "01": "(0 → 1)",
            "10": "(1 → 0)",
            "AGE": "P(AGE)",
            "PROXY": "Proxy",
            "RANDOM": "Random",
        }
        df["Mechanism"] = label_map.get(label_type, label_type)
        if label_type == "RANDOM" and "Noise level" in df.columns:
            df = df.loc[df["Noise level"] <= 0.5]
    else:
        df["Mechanism"] = "N/A"

    # Handle Subgroups and domain shifts
    if ptype in {"Subgroups", "M4", "M3", "Eicu", "EICU", "eICU"}:
        if "Variable" in df.columns and "Category" in df.columns:
            df["Set"] = df["Variable"]
            df["Noise level"] = df["Category"]
            df = clean_subgroups(df)
    elif "Noise level" in df.columns and pd.api.types.is_numeric_dtype(df["Noise level"]):
        # Invert noise level for training size (smaller size = higher perturbation)
        if ptype == "Training Size":
            df["Noise level"] = df["Noise level"].max() - df["Noise level"]

        # Min-max scale continuous noise levels to [0, 1]
        n_min, n_max = df["Noise level"].min(), df["Noise level"].max()
        if n_max > n_min:
            df["Noise level"] = ((df["Noise level"] - n_min) / (n_max - n_min)).round(2)

    return df


def get_marker_size(noise_value: float) -> int:
    """Maps normalized continuous noise level (0-1) to marker size."""
    if noise_value <= 0.33:
        return 15
    if noise_value <= 0.66:
        return 50
    return 110


# -----------------------------------------------------------------------------
# Data Loading Function (Loaded once per TRAIN_DF and TASK pair)
# -----------------------------------------------------------------------------

def load_and_merge_data(train_df: str, task: str) -> pd.DataFrame:
    """Loads and merges HPO vs No-HPO data for a specific dataset and task."""
    dir_with_hpo = Path(f"../result/{task}_HPO/{train_df}")
    dir_without_hpo = Path(f"../result/{task}_no_HPO/{train_df}")

    merge_keys = ["Model", "Perturbation type", "Set", "Mechanism", "Noise level"]
    agg_cols = (
        {"AUC": "mean", "Brier score": "mean"}
        if task == "Classif"
        else {"MAE": "mean", "RMSE": "mean", "R2": "mean"}
    )

    if not dir_with_hpo.exists() or not dir_without_hpo.exists():
        print(f"Skipping {train_df} - {task}: Directory not found.")
        return pd.DataFrame()

    merged_dfs = []
    for ptype_dir in dir_with_hpo.iterdir():
        no_hpo_ptype_dir = dir_without_hpo / ptype_dir.name
        if not no_hpo_ptype_dir.exists():
            continue

        for file_path in ptype_dir.glob("*.csv"):
            no_hpo_file = no_hpo_ptype_dir / file_path.name
            if not no_hpo_file.exists():
                continue

            try:
                df1 = pd.read_csv(file_path)
                df2 = pd.read_csv(no_hpo_file)

                # Filter excluded models
                df1 = df1[~df1["Model"].isin(EXCLUDE_MODELS)]
                df2 = df2[~df2["Model"].isin(EXCLUDE_MODELS)]

                df1 = parse_metadata(df1, ptype_dir.name, file_path.name)
                df2 = parse_metadata(df2, ptype_dir.name, file_path.name)

                df1_agg = df1.groupby(merge_keys, as_index=False).agg(agg_cols)
                df2_agg = df2.groupby(merge_keys, as_index=False).agg(agg_cols)

                df_merged = pd.merge(
                    df1_agg,
                    df2_agg,
                    on=merge_keys,
                    suffixes=("", " without HPO"),
                    how="inner",
                )
                merged_dfs.append(df_merged)

            except Exception as e:
                print(f"Error processing {ptype_dir.name}/{file_path.name}: {e}")

    return pd.concat(merged_dfs, ignore_index=True) if merged_dfs else pd.DataFrame()


# -----------------------------------------------------------------------------
# Visualization Function
# -----------------------------------------------------------------------------

def plot_and_save_figure(
    df_diff: pd.DataFrame,
    train_df: str,
    task: str,
    outcome: str,
    output_dir: Path,
):
    """Plots and saves the comparison grid for a given dataset, task, and outcome."""
    ext_df_1, ext_df_2 = EXTERNAL_DATASETS[train_df]

    title_mapping = {
        "Missing Data": "Missing Data",
        "Input Noise": "Input Noise",
        "Feature Shuffle": "Feature Selection",
        "Label Noise": "Label Noise",
        "Imbalance Data": "Imbalanced Data",
        "Training Size": "Training Size",
        "Subgroups": f"{train_df} Subgroups",
        ext_df_1: DATASET_DISPLAY_NAMES.get(ext_df_1, ext_df_1),
        ext_df_2: DATASET_DISPLAY_NAMES.get(ext_df_2, ext_df_2),
    }

    df_plot = df_diff.copy()
    df_plot["Model"] = pd.Categorical(df_plot["Model"], categories=MODEL_ORDER, ordered=True)
    df_plot = df_plot.sort_values("Model")

    # Build unique perturbation list (exclude Imbalance Data if Regression)
    unique_perturbations = [
        "Missing Data",
        "Input Noise",
        "Feature Shuffle",
        "Label Noise",
        "Imbalance Data" if task == "Classif" else None,
        "Training Size",
        "Subgroups",
        ext_df_1,
        ext_df_2,
    ]
    unique_perturbations = [p for p in unique_perturbations if p is not None]

    n_cols = 3
    n_rows = int(np.ceil(len(unique_perturbations) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten()

    for idx, ptype in enumerate(unique_perturbations):
        ax = axes[idx]
        df_ptype = df_plot[df_plot["Perturbation type"] == ptype]

        if df_ptype.empty:
            ax.set_title(title_mapping.get(ptype, ptype), fontsize=16)
            continue

        is_label_noise = ptype == "Label Noise"
        is_continuous_type = ptype in CONTINUOUS_PERTURBATIONS

        for model in MODEL_ORDER:
            df_m = df_ptype[df_ptype["Model"] == model]
            if df_m.empty:
                continue

            color = MODELS_COLOR.get(model, "gray")

            if is_label_noise:
                numeric_mask = pd.to_numeric(df_m["Noise level"], errors="coerce").notna()
                df_cont, df_cat = df_m[numeric_mask], df_m[~numeric_mask]
            elif is_continuous_type:
                df_cont, df_cat = df_m, pd.DataFrame()
            else:
                df_cont, df_cat = pd.DataFrame(), df_m

            # Plot continuous points
            if not df_cont.empty:
                sizes = df_cont["Noise level"].astype(float).map(get_marker_size)
                ax.scatter(
                    df_cont[outcome],
                    df_cont[f"{outcome} without HPO"],
                    s=sizes,
                    c=color,
                    alpha=0.7,
                    linewidths=0.2,
                )

            # Plot categorical points
            if not df_cat.empty:
                ax.scatter(
                    df_cat[outcome],
                    df_cat[f"{outcome} without HPO"],
                    s=75,
                    marker="X",
                    c=color,
                    alpha=0.7,
                    linewidths=0.2,
                )

        # Reference line (y = x)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        min_val = min(xlim[0], ylim[0])
        max_val = max(xlim[1], ylim[1])
        ax.plot([min_val, max_val], [min_val, max_val], ls="--", c=".3", zorder=0)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

        # Formatting
        ax.set_title(title_mapping.get(ptype, ptype), fontsize=18)
        ax.set_xlabel(f"{outcome} with HPO", fontsize=13)
        if idx % n_cols == 0:
            ax.set_ylabel(f"{outcome} without HPO", fontsize=13)

        ax.set_aspect("equal", adjustable="box")
        fmt = "%.3f" if outcome == "Brier score" else "%.2f"
        ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))

        # Annotations on the first subplot
        if idx == 0:
            # Metrics where higher is better vs lower is better
            if outcome in ["AUC", "R2"]:
                deg_coords = (0.05, 0.95, 0.25, 0.80)  # Top-Left: degraded
                imp_coords = (0.95, 0.05, 0.75, 0.20)  # Bottom-Right: improved
            else:  # Brier score, MAE, RMSE
                deg_coords = (0.95, 0.05, 0.75, 0.20)  # Bottom-Right: degraded
                imp_coords = (0.05, 0.95, 0.25, 0.80)  # Top-Left: improved

            ax.annotate(
                f"HPO degrades {outcome}",
                xy=(deg_coords[0], deg_coords[1]),
                xytext=(deg_coords[2], deg_coords[3]),
                xycoords="axes fraction",
                textcoords="axes fraction",
                fontsize=11,
                ha="center",
                va="center",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
            )

            ax.annotate(
                f"HPO improves {outcome}",
                xy=(imp_coords[0], imp_coords[1]),
                xytext=(imp_coords[2], imp_coords[3]),
                xycoords="axes fraction",
                textcoords="axes fraction",
                fontsize=11,
                ha="center",
                va="center",
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5),
            )

# Hide unused axes
    for idx in range(len(unique_perturbations), len(axes)):
        axes[idx].set_visible(False)

    # Legend handles
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=MODELS_COLOR.get(m, "gray"),
            markeredgecolor="black",
            markersize=10,
            label=m,
        )
        for m in MODEL_ORDER
    ]
    size_legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=np.sqrt(30),
            label="Low",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=np.sqrt(80),
            label="Medium",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=np.sqrt(150),
            label="High",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=np.sqrt(80),
            label="Categorical / NA",
        ),
    ]

    # Subplot spacing to leave clean room beneath the plots
    plt.subplots_adjust(
        left=0.08, right=0.95, bottom=0.10, top=0.95, wspace=0.10, hspace=0.28
    )

    # Line 1: Models in a single horizontal row
    legend1 = fig.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=len(model_handles),
        fontsize=10.5,
        title="Model",
        title_fontsize=11.5,
        frameon=True,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    # Line 2: Perturbation levels in a single horizontal row directly below
    legend2 = fig.legend(
        handles=size_legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.028),
        ncol=len(size_legend_elements),
        fontsize=10.5,
        title="Perturbation Level",
        title_fontsize=11.5,
        frameon=True,
        columnspacing=1.8,
        handletextpad=0.5,
    )

    # Retain the first legend when the second is rendered
    fig.add_artist(legend1)

    filename_metric = outcome.replace(" ", "_").lower()
    save_path = output_dir / f"HPO_{train_df}_{task}_{filename_metric}.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# -----------------------------------------------------------------------------
# Main Execution Loop
# -----------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for train_df in ALL_TRAIN_DFS:
        for task, outcomes in TASK_OUTCOMES.items():
            print(f"\nProcessing {train_df} | Task: {task}...")
            df_diff = load_and_merge_data(train_df=train_df, task=task)

            if df_diff.empty:
                print(f"No valid data found for {train_df} ({task}). Skipping plots.")
                continue

            for outcome in outcomes:
                print(f"  Generating plot for outcome: {outcome}...")
                plot_and_save_figure(
                    df_diff=df_diff,
                    train_df=train_df,
                    task=task,
                    outcome=outcome,
                    output_dir=OUTPUT_DIR,
                )

    print("\nAll figures generated and saved successfully.")


if __name__ == "__main__":
    main()

