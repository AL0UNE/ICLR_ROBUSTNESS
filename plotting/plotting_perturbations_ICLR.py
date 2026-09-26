#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import seaborn as sns

# ==============================================================================
# Model & Palette Configurations
# ==============================================================================

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

MODEL_FAMILY = {
    "Logistic": "Linear",
    "Linear": "Linear",
    "Ridge": "Linear",
    "LASSO": "Linear",
    "Random Forest": "Tree-based",
    "Gradient Boosting": "Tree-based",
    "XGBoost": "Tree-based",
    "LightGBM": "Tree-based",
    "CatBoost": "Tree-based",
    "RealMLP": "Deep learning",
    "TabDPT": "Foundation",
    "TabPFNv3": "Foundation",
    "TabICLv2": "Foundation",
}

FOUNDATION_MODELS = {m for m, fam in MODEL_FAMILY.items() if fam == "Foundation"}

FAMILY_MARKERS = {
    "Linear": "o",          # Circle
    "Tree-based": "s",      # Square
    "Deep learning": "^",   # Triangle
    "Foundation": "D",      # Diamond
}

FAMILY_ORDER = ["Linear", "Tree-based", "Deep learning", "Foundation"]
LINE_WIDTH = 3.0
CI_LINE_WIDTH = 2.0
CI_ALPHA = 0.10

SUBGROUPS_MAPPING = {
    "age_group": "Age group",
    "gender": "Gender",
    "ICU_unit": "ICU type",
    "anchor_year_group": "Year group",
    "region": "Hospital region",
}

ICU_MAPPING = {
    # MIMIC-III
    "CCU": "CCU",
    "CSRU": "CSRU",
    "SICU": "SICU",
    "MICU": "MICU",
    "TSICU": "TSICU",
    
    # MIMIC-IV
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

EXCLUDED_ICU = [c for c in ICU_MAPPING.values() if c not in {"MICU", "SICU"}]

MODEL_SUBSETS = {
    "complete": [
        "Linear", "Logistic", "LASSO", "Ridge",
        "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost",
        "RealMLP", "TabICLv2", "TabDPT", "TabPFNv3",
    ],
}

# ==============================================================================
# Helpers & Validations
# ==============================================================================

def get_task_models(subset_models, task=None, outcome=None):
    """Filters model subset depending on whether the task is classification or regression."""
    task_type = None
    if task:
        task_type = task.lower()
    elif outcome:
        out_lower = str(outcome).lower()
        if any(metric in out_lower for metric in ["auc", "brier", "acc", "f1", "logloss"]):
            task_type = "classification"
        elif any(metric in out_lower for metric in ["rmse", "mae", "r2", "mse"]):
            task_type = "regression"

    if task_type == "classification":
        return [m for m in subset_models if m != "Linear"]
    elif task_type == "regression":
        return [m for m in subset_models if m != "Logistic"]
    return list(subset_models)


def validate_model_configuration(subset_models):
    missing_colors = [m for m in subset_models if m not in MODELS_COLOR]
    missing_families = [m for m in subset_models if m not in MODEL_FAMILY]
    if missing_colors:
        raise ValueError(f"Models missing from MODELS_COLOR: {missing_colors}")
    if missing_families:
        raise ValueError(f"Models missing from MODEL_FAMILY: {missing_families}")


def add_model_family(df):
    if df.empty:
        return df
    df = df.copy()
    df["Model family"] = df["Model"].map(MODEL_FAMILY)
    missing = df.loc[df["Model family"].isna(), "Model"].dropna().unique()
    if len(missing) > 0:
        raise ValueError(f"Missing from MODEL_FAMILY: {list(missing)}")
    return df


def clean_subgroups(df):
    df = df.copy()
    df["Variable"] = df["Variable"].replace(SUBGROUPS_MAPPING)
    df["Category"] = df["Category"].replace(ICU_MAPPING)
    return df[~((df["Variable"] == "ICU type") & df["Category"].isin(EXCLUDED_ICU))]


def resolve_dataset_folder(parent_dir, dataset_name):
    if not parent_dir:
        return None
    alias_map = {
        "eicu": ["eICU", "eicu"],
        "mimic-iv": ["MIMIC-IV", "m4", "mimic4", "MIMIC4", "mimic-iv"],
        "mimic4": ["MIMIC-IV", "m4", "mimic4", "MIMIC4", "mimic-iv"],
        "m4": ["m4", "mimic4", "MIMIC-IV", "MIMIC4", "mimic-iv"],
        "mimic-iii": ["MIMIC-III", "m3", "mimic3", "MIMIC3", "mimic-iii"],
        "mimic3": ["MIMIC-III", "m3", "mimic3", "MIMIC3", "mimic-iii"],
        "m3": ["m3", "mimic3", "MIMIC-III", "MIMIC3", "mimic-iii"],
    }
    parent = Path(parent_dir)
    if not parent.exists():
        return None

    candidates = alias_map.get(dataset_name.lower(), [dataset_name])
    existing = [p for p in parent.iterdir() if p.is_dir()]
    name_to_path = {p.name: p for p in existing}
    lower_to_path = {p.name.lower(): p for p in existing}

    for cand in candidates:
        if cand in name_to_path:
            return str(name_to_path[cand])
    for cand in candidates:
        if cand.lower() in lower_to_path:
            return str(lower_to_path[cand.lower()])
    return None


def get_age_order(dataset_name, available_categories=None):
    d_lower = dataset_name.lower()
    if any(k in d_lower for k in ("m3", "mimic-iii", "mimic3")):
        default_order = ["(15, 30]", "(30, 45]", "(45, 60]", "(60, 75]", "(75, 90]"]
    elif "eicu" in d_lower:
        default_order = ["(15, 30]", "(30, 45]", "(45, 60]", "(60, 75]", "(75, 92]"]
    else:
        default_order = ["(17, 30]", "(30, 45]", "(45, 60]", "(60, 75]", "(75, 103]"]

    if available_categories is not None:
        order = [c for c in default_order if c in available_categories]
        remaining = sorted([c for c in available_categories if c not in order])
        return order + remaining
    return default_order


def get_target_subgroups(dataset_name, available_vars):
    d_lower = dataset_name.lower()
    if "eicu" in d_lower:
        preferred = ["Age group", "Gender", "ICU type", "Hospital region"]
    elif any(k in d_lower for k in ("m4", "mimic4", "mimic-iv")):
        preferred = ["Age group", "Gender", "ICU type", "Year group"]
    else:
        preferred = ["Age group", "Gender", "ICU type"]

    result = [v for v in preferred if v in available_vars]
    raw_subgroup_cols = {"anchor_year_group", "region", "age_group", "gender", "ICU_unit"}
    result.extend([v for v in available_vars if v not in result and v not in raw_subgroup_cols])
    return result


def filter_subset(df, subset_models, set_c=None, sort_col=None):
    if df.empty:
        return df
    mask = df["Model"].isin(subset_models)
    if set_c is not None and "Set" in df.columns:
        mask &= df["Set"] == set_c
    filtered = df[mask]
    return filtered.sort_values(by=sort_col) if sort_col and sort_col in filtered.columns else filtered


# ==============================================================================
# Data Merging Helpers
# ==============================================================================

def merge_hpo_and_no_hpo_dfs(df_hpo, df_no_hpo):
    """Combines non-foundation models from HPO and foundation models from no_HPO."""
    dfs = []
    if df_hpo is not None and not df_hpo.empty:
        if "Model" in df_hpo.columns:
            dfs.append(df_hpo[~df_hpo["Model"].isin(FOUNDATION_MODELS)])
        else:
            dfs.append(df_hpo)
            
    if df_no_hpo is not None and not df_no_hpo.empty:
        if "Model" in df_no_hpo.columns:
            dfs.append(df_no_hpo[df_no_hpo["Model"].isin(FOUNDATION_MODELS)])
        else:
            dfs.append(df_no_hpo)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def merge_perturbation_data(data_hpo, data_no_hpo):
    """Merges dictionaries of perturbation datasets."""
    merged = {}
    all_keys = set(data_hpo.keys()).union(data_no_hpo.keys())
    for k in all_keys:
        df_hpo = data_hpo.get(k, pd.DataFrame())
        df_no_hpo = data_no_hpo.get(k, pd.DataFrame())
        merged[k] = merge_hpo_and_no_hpo_dfs(df_hpo, df_no_hpo)
    return merged


# ==============================================================================
# Data Loaders
# ==============================================================================

def load_perturbation_data(dev_dir):
    """Loads all perturbation files for the development dataset."""
    data = {
        'missing': pd.DataFrame(),
        'input_noise': pd.DataFrame(),
        'feature_shuffle': pd.DataFrame(),
        'training_size': pd.DataFrame(),
        'imbalanced': pd.DataFrame(),
        'label_noise': pd.DataFrame(),
        'subgroups': pd.DataFrame(),
    }
    if not dev_dir or not os.path.exists(dev_dir):
        return data

    # 1. Missing Data
    df_missing = []
    dir_missing = os.path.join(dev_dir, "missing_data")
    if os.path.exists(dir_missing):
        for file in os.listdir(dir_missing):
            if not file.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(dir_missing, file))
            parts = file.split('.')[0].split('_')
            df['Mechanism'] = parts[0]
            df['sort_order'] = 2 if parts[0] == 'MAR' else 3 if parts[0] == 'MNAR' else 1
            if 'MAR' in parts:
                df['Noise level'] = df['Noise level'] / 2
            df['Noise level'] = (df['Noise level'] * 100).astype(int)
            df['Set'] = parts[1]
            df_missing.append(df)
    data['missing'] = pd.concat(df_missing, ignore_index=True) if df_missing else pd.DataFrame()

    # 2. Input Noise
    df_input = []
    dir_input = os.path.join(dev_dir, "input_noise")
    if os.path.exists(dir_input):
        for file in os.listdir(dir_input):
            if not file.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(dir_input, file))
            parts = file.split('.')[0].split('_')
            df['Set'] = parts[2]
            df['sort_order'] = 1 if 'CONT' in parts else 2 if 'CAT' in parts else 3
            df['Feature type'] = (
                'Continuous' if 'CONT' in parts
                else 'Categorical' if 'CAT' in parts
                else 'ALL'
            )
            df_input.append(df)
    data['input_noise'] = pd.concat(df_input, ignore_index=True) if df_input else pd.DataFrame()

    # 3. Feature Shuffle
    df_fs = []
    dir_fs = os.path.join(dev_dir, "feature_shuffle")
    if os.path.exists(dir_fs):
        for file in os.listdir(dir_fs):
            if not file.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(dir_fs, file))
            parts = file.split('_')
            df['Set'] = parts[1]
            df['Noise level'] = (df['Noise level'] * 100).astype(int)
            df_fs.append(df)
    data['feature_shuffle'] = pd.concat(df_fs, ignore_index=True) if df_fs else pd.DataFrame()

    # 4. Training Size
    df_training = []
    dir_train = os.path.join(dev_dir, "training_size")
    if os.path.exists(dir_train):
        for file in os.listdir(dir_train):
            if not file.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(dir_train, file))
            df['Noise level'] = (df['Noise level'] * 100).astype(int)
            df_training.append(df)
    data['training_size'] = pd.concat(df_training, ignore_index=True) if df_training else pd.DataFrame()

    # 5. Imbalanced Data
    df_imb = []
    dir_imb = os.path.join(dev_dir, "imbalance_data")
    if os.path.exists(dir_imb):
        for file in os.listdir(dir_imb):
            if not file.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(dir_imb, file))
            parts = file.split('_')
            df['Set'] = parts[1]
            df['Noise level'] = (df['Noise level'].round(1) * 100).astype(int)
            df_imb.append(df)
    data['imbalanced'] = pd.concat(df_imb, ignore_index=True) if df_imb else pd.DataFrame()

    # 6. Label Noise
    df_label = []
    dir_label = os.path.join(dev_dir, "label_noise")
    if os.path.exists(dir_label):
        for file in os.listdir(dir_label):
            if not file.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(dir_label, file))
            parts = file.split('_')
            if parts[0] == '01':
                df['Type'] = r"(0 $\Longrightarrow$ 1)"
                df['sort_order'] = 2
            elif parts[0] == '10':
                df['Type'] = r"(1 $\Longrightarrow$ 0)"
                df['sort_order'] = 1
            elif parts[0] == 'AGE':
                df['Type'] = "P(AGE)"
                df['sort_order'] = 3
            elif parts[0] == 'PROXY':
                df['Type'] = "Proxy"
                df['Noise level'] = df['Noise level'].replace({
                    'hospital_death': 'Hospital',
                    'icu_death': 'ICU mortality',
                    'overall_death': 'All-cause mortality'
                })
                df = df.loc[df['Noise level'] != 'Hospital']
                df['sort_order'] = 4
            elif parts[0] == 'RANDOM':
                df['Type'] = r"(0 $\Longleftrightarrow$ 1)"
                df = df.loc[df['Noise level'] <= 0.5]
                df['sort_order'] = 0

            if parts[0] != 'PROXY':
                df['Noise level'] = (df['Noise level'] * 100).astype(int)
            df_label.append(df)
    data['label_noise'] = pd.concat(df_label, ignore_index=True) if df_label else pd.DataFrame()

    # 7. Development Subgroups
    df_sub = []
    dir_sub = os.path.join(dev_dir, "subgroups")
    if os.path.exists(dir_sub):
        for file in os.listdir(dir_sub):
            if not file.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(dir_sub, file))
            df = clean_subgroups(df)
            df_sub.append(df)
    data['subgroups'] = pd.concat(df_sub, ignore_index=True) if df_sub else pd.DataFrame()

    return data


def load_external_subgroups(ext_dir):
    """Loads all external subgroup files, skipping summary files."""
    df_ext = []
    if ext_dir and os.path.exists(ext_dir):
        for file in os.listdir(ext_dir):
            if not file.endswith('.csv'):
                continue
            if file in ['m3.csv', 'm4.csv', 'eICU.csv', 'mimic3.csv', 'mimic4.csv', 'eicu.csv']:
                continue
            df = pd.read_csv(os.path.join(ext_dir, file))
            df = clean_subgroups(df)
            df_ext.append(df)
        if df_ext:
            return pd.concat(df_ext, ignore_index=True)
    return pd.DataFrame()


# ==============================================================================
# Visualization Helpers
# ==============================================================================

def remove_legend(ax):
    if ax.legend_ is not None:
        ax.legend_.remove()
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def remove_blank_axes(fig):
    """Removes all axes from the figure that have no plotted data."""
    for ax in list(fig.axes):
        if not ax.has_data():
            fig.delaxes(ax)


def add_group_box(fig, axs_group, text, pad=None, text_pad=None):
    """Draws a boundary box around a group of axes and displays the group label.
    Adapts to axes spanning multiple rows by drawing separate boxes per row to prevent overlaps."""
    valid_axs = [ax for ax in axs_group if ax in fig.axes and ax.get_visible() and ax.has_data()]
    if not valid_axs:
        return

    fig_h = fig.get_figheight()

    if pad is None:
        pad = 50.0 / (fig_h * 72.0)
    if text_pad is None:
        text_pad = 16.0 / (fig_h * 72.0)

    # Group axes by their y0 coordinate to detect if they are wrapped across different rows
    row_groups = defaultdict(list)
    for ax in valid_axs:
        # y0 is identical/very close for axes located in the same horizontal row
        y0_rounded = round(ax.get_position().y0, 3)
        row_groups[y0_rounded].append(ax)

    # Sort rows from top to bottom (higher y0 is higher geographically on the figure)
    sorted_rows = sorted(row_groups.keys(), reverse=True)

    for i, y0 in enumerate(sorted_rows):
        group = row_groups[y0]
        bboxes = [ax.get_position() for ax in group]
        full_bbox = bboxes[0].union(bboxes)

        box_top = full_bbox.y1 + pad

        rect = Rectangle(
            (full_bbox.x0, full_bbox.y0),
            full_bbox.width,
            full_bbox.height + pad,
            edgecolor="dodgerblue",
            facecolor="none",
            alpha=0.8,
            linewidth=4.0,
            transform=fig.transFigure,
            clip_on=False,
        )
        fig.add_artist(rect)
        
        # Add text label ONLY to the top-most bounding box of the group
        if i == 0:
            fig.text(
                full_bbox.x0,
                box_top + text_pad,
                text,
                ha="left",
                va="bottom",
                fontsize=40,
                fontweight="bold",
                transform=fig.transFigure,
            )


def create_global_legend(
    fig,
    subset_models,
    models_color,
    y_pos=0.02,
    model_y_offset=None,
    fontsize=28,
):
    """
    Creates a two-tiered global legend (Models on top, Model family below).
    Dynamically computes model_y_offset to prevent overlapping between the
    two legend groups and the figure axes regardless of figure height.
    """
    fig_h = fig.get_figheight()

    if model_y_offset is None:
        # Allocate ~135 pt vertical clearance between baseline anchors
        model_y_offset = max(135.0 / (fig_h * 72.0), 0.058)

    model_handles = [
        Line2D(
            [0], [0],
            color=models_color[m],
            marker=FAMILY_MARKERS[MODEL_FAMILY[m]],
            markersize=14,
            linestyle="-",
            linewidth=3,
            label=m,
        )
        for m in subset_models if m in models_color
    ]

    family_handles = [
        Line2D(
            [0], [0],
            color="black",
            marker=FAMILY_MARKERS[fam],
            markersize=14,
            linestyle="-",
            linewidth=3,
            label=fam,
        )
        for fam in FAMILY_ORDER
    ]

    # Upper Legend: Models
    leg_models = fig.legend(
        handles=model_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y_pos + model_y_offset),
        ncol=6 if len(model_handles) <= 12 else 5,
        fontsize=fontsize,
        title="Models",
        title_fontsize=fontsize + 4,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.5,
    )
    fig.add_artist(leg_models)

    # Lower Legend: Model family
    fig.legend(
        handles=family_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y_pos),
        ncol=4,
        fontsize=fontsize,
        title="Model family",
        title_fontsize=fontsize + 4,
        frameon=False,
        handlelength=2.5,
        columnspacing=2.0,
    )


def format_axes(axs, outcome, fontsize=36):
    y_fmt = "%.3f" if "brier" in outcome.lower() else "%.2f"

    for row in axs:
        first_col_set = False
        for ax in row:
            if not ax.has_data():
                continue
            ax.tick_params(axis="both", labelsize=24)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_linewidth(1.0)
            ax.yaxis.set_major_formatter(FormatStrFormatter(y_fmt))

            if not first_col_set:
                ax.set_ylabel(outcome, fontsize=fontsize * 1.3)
                first_col_set = True
            else:
                ax.set_ylabel("")


def plot_model_lines(data, x, y, subset_models, ax):
    if data.empty:
        return

    sns.lineplot(
        data=add_model_family(data),
        x=x,
        y=y,
        hue="Model",
        hue_order=subset_models,
        palette=MODELS_COLOR,
        style="Model family",
        style_order=FAMILY_ORDER,
        markers=FAMILY_MARKERS,
        dashes=False,
        markersize=9,
        linewidth=LINE_WIDTH,
        errorbar="sd",
        ax=ax,
    )

    # When the metric is RMSE, convert shaded confidence bands into unfilled dotted borders
    if str(y).strip().upper() == "RMSE":
        for coll in ax.collections:
            fc = coll.get_facecolor()
            if len(fc) > 0:
                coll.set_edgecolor(fc[:, :3])
            coll.set_facecolor("none")
            coll.set_linestyle(":")
            coll.set_linewidth(CI_LINE_WIDTH)
            coll.set_alpha(1.0)

    remove_legend(ax)


# ==============================================================================
# Figures
# ==============================================================================

def save_fig_val_all_perturbations(
    data_perturb,
    dev_dataset="eICU",
    models_name="complete",
    outcome="AUC",
    save=True,
    show=False,
    legend=True,
    task=None,
):
    """
    Plots the perturbations for VAL and ALL sets on the same figure with 2 aligned rows.
    Row 0: VAL set (perturbations added to validation set only)
    Row 1: ALL set (perturbations added both to the training and validation set)
    """
    subset_models = get_task_models(MODEL_SUBSETS[models_name], task=task, outcome=outcome)
    validate_model_configuration(subset_models)

    # Replace spaces with underscores in outcome for clean filenames (e.g. Brier score -> Brier_score)
    clean_outcome = outcome.replace(" ", "_")
    Path("figures").mkdir(exist_ok=True)
    filename = f"figures/dev_{dev_dataset}_VAL_ALL_{clean_outcome}_perturbations"

    # Separate data for VAL and ALL
    data_by_set = {
        "VAL": {
            "missing": filter_subset(data_perturb["missing"], subset_models, "VAL", "sort_order"),
            "input": filter_subset(data_perturb["input_noise"], subset_models, "VAL", "sort_order"),
            "feature": filter_subset(data_perturb["feature_shuffle"], subset_models, "VAL"),
            "subtitle": "Perturbations added to Validation set",
        },
        "ALL": {
            "missing": filter_subset(data_perturb["missing"], subset_models, "ALL", "sort_order"),
            "input": filter_subset(data_perturb["input_noise"], subset_models, "ALL", "sort_order"),
            "feature": filter_subset(data_perturb["feature_shuffle"], subset_models, "ALL"),
            "subtitle": "Perturbations added to both the training and validation set",
        },
    }

    # Identify ordered mechanisms and feature types across both sets to align columns
    mechanisms = []
    for s in ["VAL", "ALL"]:
        df_m = data_by_set[s]["missing"]
        if not df_m.empty and "Mechanism" in df_m.columns:
            for m in df_m.sort_values(by="sort_order")["Mechanism"].unique():
                if m not in mechanisms:
                    mechanisms.append(m)
    if not mechanisms:
        mechanisms = ["MCAR", "MAR", "MNAR"]

    feature_types = []
    for s in ["VAL", "ALL"]:
        df_i = data_by_set[s]["input"]
        if not df_i.empty and "Feature type" in df_i.columns:
            for ft in df_i.sort_values(by="sort_order")["Feature type"].unique():
                if ft not in feature_types:
                    feature_types.append(ft)
    if not feature_types:
        feature_types = ["Continuous", "Categorical"]

    has_feature_shuffle = any(not data_by_set[s]["feature"].empty for s in ["VAL", "ALL"])

    ncols = len(mechanisms) + len(feature_types) + (1 if has_feature_shuffle else 0)
    nrows = 2
    fontsize = 34

    figsize = (max(ncols * 9.5, 52), 30)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if axs.ndim == 1:
        axs = axs.reshape(nrows, ncols)

    ftype_labels = {
        "Continuous": r"Noise level ($\times 2\sigma_{feature}$)",
        "Categorical": "Misclassification rate",
    }

    group_axes_by_row = []

    # Plot both sets
    for r, set_key in enumerate(["VAL", "ALL"]):
        df_m = data_by_set[set_key]["missing"]
        df_i = data_by_set[set_key]["input"]
        df_f = data_by_set[set_key]["feature"]

        col = 0
        missing_axes = []
        input_axes = []
        feature_axes = []

        # 1. Missing Data Columns
        for mech in mechanisms:
            ax = axs[r, col]
            sub = df_m[df_m["Mechanism"] == mech] if not df_m.empty and "Mechanism" in df_m.columns else pd.DataFrame()
            if not sub.empty:
                plot_model_lines(sub, "Noise level", outcome, subset_models, ax)
            ax.set_title(mech, fontsize=fontsize)
            ax.set_xlabel("Missing (%)", fontsize=fontsize)
            missing_axes.append(ax)
            col += 1

        # 2. Input Noise Columns
        for ftype in feature_types:
            ax = axs[r, col]
            sub = df_i[df_i["Feature type"] == ftype] if not df_i.empty and "Feature type" in df_i.columns else pd.DataFrame()
            if not sub.empty:
                plot_model_lines(sub, "Noise level", outcome, subset_models, ax)
            ax.set_title(f"{ftype} features", fontsize=fontsize)
            ax.set_xlabel(ftype_labels.get(ftype, "Noise level"), fontsize=fontsize)
            input_axes.append(ax)
            col += 1

        # 3. Feature Shuffle Column
        if has_feature_shuffle:
            ax = axs[r, col]
            if not df_f.empty:
                plot_model_lines(df_f, "Noise level", outcome, subset_models, ax)
            ax.set_title("Feature shuffle", fontsize=fontsize)
            ax.set_xlabel("Irrelevant features (%)", fontsize=fontsize)
            feature_axes.append(ax)
            col += 1

        group_axes_by_row.append((set_key, missing_axes, input_axes, feature_axes))

    format_axes(axs, outcome, fontsize=fontsize)
    remove_blank_axes(fig)

    # rect=[0, 0.185, 1, 0.88] reserves 18.5% (~5.5 inches) of space at bottom for the stacked legend
    plt.tight_layout(h_pad=18, w_pad=2, rect=[0, 0.185, 1, 0.88])

    fig_h = fig.get_figheight()

    # Draw group boxes and section subtitles for each row
    for r, (set_key, missing_axes, input_axes, feature_axes) in enumerate(group_axes_by_row):
        active_missing = [ax for ax in missing_axes if ax in fig.axes and ax.has_data()]
        active_input = [ax for ax in input_axes if ax in fig.axes and ax.has_data()]
        active_feature = [ax for ax in feature_axes if ax in fig.axes and ax.has_data()]

        if active_missing:
            add_group_box(fig, active_missing, "Missing Data")
        if active_input:
            add_group_box(fig, active_input, "Input Noise")
        if active_feature:
            add_group_box(fig, active_feature, "Feature Selection")

        # Subtitle positioning directly above the group boxes of each respective row
        row_axes = [ax for ax in axs[r] if ax in fig.axes and ax.has_data()]
        if row_axes:
            bboxes = [ax.get_position() for ax in row_axes]
            full_row_bbox = bboxes[0].union(bboxes)
            sub_y = full_row_bbox.y1 + (140.0 / (fig_h * 72.0))
            fig.text(
                full_row_bbox.x0,
                sub_y,
                data_by_set[set_key]["subtitle"],
                ha="left",
                va="bottom",
                fontsize=46,
                fontweight="bold",
                color="#0B3C5D",
                transform=fig.transFigure,
            )

    if legend:
        create_global_legend(fig, subset_models, MODELS_COLOR, y_pos=0.02)
    if save:
        plt.savefig(f"{filename}.pdf", format="pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def save_fig_perturbations(
    data_perturb,
    dev_dataset="eICU",
    models_name="complete",
    outcome="AUC",
    set_c="TRAIN",
    save=True,
    show=False,
    legend=True,
    task=None,
    df_ext1=None,
    df_ext2=None,
    ext1_name=None,
    ext2_name=None,
):
    """
    Plots perturbation figures. When set_c is 'VAL', 'ALL', or 'VAL_ALL', routes automatically
    to save_fig_val_all_perturbations to present both sets combined on one figure.
    When set_c is 'TRAIN', plots the comprehensive 4-row figure with integrated external subgroups.
    """
    if str(set_c).upper() in {"VAL", "ALL", "VAL_ALL", "COMBINED"}:
        return save_fig_val_all_perturbations(
            data_perturb=data_perturb,
            dev_dataset=dev_dataset,
            models_name=models_name,
            outcome=outcome,
            save=save,
            show=show,
            legend=legend,
            task=task,
        )

    subset_models = get_task_models(MODEL_SUBSETS[models_name], task=task, outcome=outcome)
    validate_model_configuration(subset_models)
    subset_markers = [FAMILY_MARKERS[MODEL_FAMILY[m]] for m in subset_models]

    # Replace spaces with underscores in outcome for clean filenames (e.g. Brier score -> Brier_score)
    clean_outcome = outcome.replace(" ", "_")
    Path("figures").mkdir(exist_ok=True)
    filename = f"figures/dev_{dev_dataset}_{set_c}_{clean_outcome}_perturbations"

    is_train = str(set_c).upper() == "TRAIN"
    is_mimic = "mimic" in dev_dataset.lower()

    df_missing = filter_subset(data_perturb["missing"], subset_models, set_c, "sort_order")
    df_input = filter_subset(data_perturb["input_noise"], subset_models, set_c, "sort_order")
    df_feature = filter_subset(data_perturb["feature_shuffle"], subset_models, set_c)
    df_label = filter_subset(data_perturb["label_noise"], subset_models, sort_col="sort_order")
    df_train = filter_subset(data_perturb["training_size"], subset_models)
    df_imb = filter_subset(data_perturb["imbalanced"], subset_models)
    df_sub = filter_subset(data_perturb["subgroups"], subset_models)

    fontsize = 36
    nrows = 4 if is_train else 3
    
    # Provide 7 columns so all extra categories for the External Datasets fit properly during TRAIN
    if is_train:
        ncols = 7
        figsize = (72, 46)
    else:
        ncols = 6
        figsize = (62, 34)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if axs.ndim == 1:
        axs = axs.reshape(nrows, ncols)

    training_axes = []
    imbalance_axes = []

    # Inject layout mapping for MIMIC during TRAIN
    if is_train and is_mimic:
        if not df_train.empty:
            ax = axs[0, ncols - 1]
            plot_model_lines(
                df_train.reset_index(),
                "Noise level",
                outcome,
                subset_models,
                ax,
            )
            ax.set_title("Training size", fontsize=fontsize)
            ax.set_xlim(ax.get_xlim()[::-1])
            ax.set_xlabel("Training subset (%)", fontsize=fontsize)
            training_axes.append(ax)

        if not df_imb.empty:
            ax = axs[1, ncols - 1]
            plot_model_lines(
                df_imb,
                "Noise level",
                outcome,
                subset_models,
                ax,
            )
            ax.set_title("Class imbalance", fontsize=fontsize)
            ax.set_xlim(ax.get_xlim()[::-1])
            ax.set_xlabel("Positive samples kept (%)", fontsize=fontsize)
            imbalance_axes.append(ax)

    # Protect the last column from standard processing when used by custom layout
    max_col_row0 = (ncols - 1) if (is_train and is_mimic) else ncols
    max_col_row1 = (ncols - 1) if (is_train and is_mimic) else ncols

    # 1. Row 0: Missing Data & Input Noise
    for i, mech in enumerate(df_missing["Mechanism"].unique()[:3] if not df_missing.empty else []):
        if i >= max_col_row0:
            break
        plot_model_lines(df_missing[df_missing["Mechanism"] == mech], "Noise level", outcome, subset_models, axs[0, i])
        axs[0, i].set_title(mech, fontsize=fontsize)
        axs[0, i].set_xlabel("Missing (%)", fontsize=fontsize)

    ftype_labels = {
        "Continuous": r"Noise level ($\times 2\sigma_{feature}$)",
        "Categorical": "Misclassification rate",
    }
    for i, ftype in enumerate(df_input["Feature type"].unique()[:3] if not df_input.empty else []):
        col = i + 3
        if col >= max_col_row0:
            break
        plot_model_lines(df_input[df_input["Feature type"] == ftype], "Noise level", outcome, subset_models, axs[0, col])
        axs[0, col].set_title(f"{ftype} features", fontsize=fontsize)
        axs[0, col].set_xlabel(ftype_labels.get(ftype, "Noise level"), fontsize=fontsize)

    # Row 1: Label Noise + Feature Selection
    next_col = 0
    label_noise_axes = []
    feature_selection_axes = []

    if not df_label.empty:
        label_types = [
            ltype
            for ltype in df_label["Type"].unique()
            if ltype not in {"P(AGE)", "Proxy"}
        ]

        for ltype in label_types:
            if next_col >= max_col_row1:
                break

            ax = axs[1, next_col]
            plot_model_lines(
                df_label[df_label["Type"] == ltype],
                "Noise level",
                outcome,
                subset_models,
                ax,
            )
            ax.set_title(f"{ltype}", fontsize=fontsize)
            ax.set_xlabel("Label noise (%)", fontsize=fontsize)
            label_noise_axes.append(ax)
            next_col += 1

        if "P(AGE)" in df_label["Type"].values and next_col < max_col_row1:
            ax = axs[1, next_col]
            plot_model_lines(
                df_label[df_label["Type"] == "P(AGE)"],
                "Noise level",
                outcome,
                subset_models,
                ax,
            )
            ax.set_title(r"Label noise $\propto$ Age", fontsize=fontsize)
            ax.set_xlabel("Label noise (%)", fontsize=fontsize)
            label_noise_axes.append(ax)
            next_col += 1

        if "Proxy" in df_label["Type"].values and next_col < max_col_row1:
            ax = axs[1, next_col]
            proxy_df = df_label[df_label["Type"] == "Proxy"]

            if "eicu" in dev_dataset.lower():
                proxy_order = ["ICU mortality"]
                proxy_df = proxy_df[proxy_df["Noise level"] == "ICU mortality"]
            else:
                proxy_order = ["ICU mortality", "All-cause mortality"]

            sns.pointplot(
                data=proxy_df,
                x="Noise level",
                y=outcome,
                hue="Model",
                order=proxy_order,
                palette=MODELS_COLOR,
                hue_order=subset_models,
                markers=subset_markers,
                dodge=0.2,
                ax=ax,
            )
            ax.set_title("Proxy label", fontsize=fontsize)
            ax.set_xlabel("Proxy type", fontsize=fontsize)
            remove_legend(ax)
            label_noise_axes.append(ax)
            next_col += 1

    if not df_feature.empty and next_col < max_col_row1:
        ax = axs[1, next_col]
        plot_model_lines(
            df_feature,
            "Noise level",
            outcome,
            subset_models,
            ax,
        )
        ax.set_title("Feature shuffle", fontsize=fontsize)
        ax.set_xlabel("Irrelevant features (%)", fontsize=fontsize)
        feature_selection_axes.append(ax)
        next_col += 1

    # Row 2: Training Data + Imbalance + Subgroups
    next_col = 0
    subgroup_axes = []

    # If the setup wasn't moved dynamically to Rows 0 & 1, print on Row 2
    if not (is_train and is_mimic):
        if not df_train.empty and next_col < ncols:
            ax = axs[2, next_col]
            plot_model_lines(
                df_train.reset_index(),
                "Noise level",
                outcome,
                subset_models,
                ax,
            )
            ax.set_title("Training size", fontsize=fontsize)
            ax.set_xlim(ax.get_xlim()[::-1])
            ax.set_xlabel("Training subset (%)", fontsize=fontsize)
            training_axes.append(ax)
            next_col += 1

        if not df_imb.empty and next_col < ncols:
            ax = axs[2, next_col]
            plot_model_lines(
                df_imb,
                "Noise level",
                outcome,
                subset_models,
                ax,
            )
            ax.set_title("Class imbalance", fontsize=fontsize)
            ax.set_xlim(ax.get_xlim()[::-1])
            ax.set_xlabel("Positive samples kept (%)", fontsize=fontsize)
            imbalance_axes.append(ax)
            next_col += 1

    if not df_sub.empty:
        gender_df = df_sub[df_sub["Variable"] == "Gender"]
        if not gender_df.empty and next_col < ncols:
            ax = axs[2, next_col]
            sns.boxplot(
                data=gender_df,
                x="Category",
                y=outcome,
                hue="Model",
                palette=MODELS_COLOR,
                hue_order=subset_models,
                ax=ax,
            )
            ax.set_title("Gender", fontsize=fontsize)
            ax.set_xlabel("")
            remove_legend(ax)
            subgroup_axes.append(ax)
            next_col += 1

        age_df = df_sub[df_sub["Variable"] == "Age group"]
        if not age_df.empty and next_col < ncols:
            ax = axs[2, next_col]
            sns.pointplot(
                data=age_df,
                x="Category",
                y=outcome,
                hue="Model",
                palette=MODELS_COLOR,
                hue_order=subset_models,
                markers=subset_markers,
                order=get_age_order(dev_dataset, age_df["Category"].unique()),
                dodge=0.5,
                ax=ax,
            )
            ax.set_title("Age group", fontsize=fontsize)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=20)
            remove_legend(ax)
            subgroup_axes.append(ax)
            next_col += 1

        icu_df = df_sub[df_sub["Variable"] == "ICU type"]
        if not icu_df.empty and next_col < ncols:
            ax = axs[2, next_col]
            sns.boxplot(
                data=icu_df,
                x="Category",
                y=outcome,
                hue="Model",
                palette=MODELS_COLOR,
                hue_order=subset_models,
                ax=ax,
            )
            ax.set_title("ICU type", fontsize=fontsize)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=20)
            remove_legend(ax)
            subgroup_axes.append(ax)
            next_col += 1

    # Row 3: External Datasets (Plotted when set is TRAIN)
    ext1_axes = []
    ext2_axes = []

    if is_train:
        ext_configs = [
            (df_ext1, ext1_name, ext1_axes),
            (df_ext2, ext2_name, ext2_axes),
        ]

        # Gather ALL remaining empty axes in the figure
        available_axes = []
        if is_mimic:
            # Top-down tracking starting from Row 2 to ensure Ext1 populates alongside Dev Subgroups
            for r in range(2, nrows):
                for c in range(ncols):
                    ax = axs[r, c]
                    if not ax.has_data():
                        available_axes.append(ax)
        else:
            # Default tracking bottom-up for eICU
            for r in range(nrows - 1, -1, -1):
                for c in range(ncols):
                    ax = axs[r, c]
                    if not ax.has_data():
                        available_axes.append(ax)

        current_ax_idx = 0
        for df_ext, name, axes_list in ext_configs:
            if df_ext is not None and not df_ext.empty and name:
                
                # --- OVERRIDE FOR eICU ---
                # Guarantee that eICU is pushed strictly to the final row when Dev is MIMIC-IV
                if "mimic-iv" in dev_dataset.lower() and "eicu" in name.lower():
                    while current_ax_idx < len(available_axes) and available_axes[current_ax_idx] not in axs[-1]:
                        current_ax_idx += 1
                # -------------------------

                df_filtered = filter_subset(df_ext, subset_models)
                avail_vars = (
                    df_filtered["Variable"].unique()
                    if "Variable" in df_filtered.columns
                    else []
                )
                target_cats = get_target_subgroups(name, avail_vars)

                # Dynamically assign available missing axes sequentially to plot all targeted subgroups
                for cat in target_cats:
                    if current_ax_idx >= len(available_axes):
                        break
                    
                    sub_data = df_filtered[df_filtered["Variable"] == cat]
                    if sub_data.empty:
                        continue

                    ax = available_axes[current_ax_idx]

                    unique_cats = sub_data["Category"].dropna().unique()
                    order, rotation, dodge = None, 0, 0.2

                    if cat == "Age group":
                        order, rotation, dodge = get_age_order(name, unique_cats), 20, 0.3
                    elif cat == "Year group":
                        years = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"]
                        order, rotation, dodge = [y for y in years if y in unique_cats] or None, 25, 0.2
                    elif cat in {"Hospital region", "ICU type"}:
                        order, rotation, dodge = sorted(unique_cats), 25, 0.3

                    sns.pointplot(
                        data=sub_data,
                        x="Category",
                        y=outcome,
                        hue="Model",
                        palette=MODELS_COLOR,
                        hue_order=subset_models,
                        markers=subset_markers,
                        order=order,
                        dodge=dodge,
                        ax=ax,
                    )
                    if rotation:
                        ax.tick_params(axis="x", rotation=rotation)
                    ax.set_title(cat, fontsize=fontsize)
                    ax.set_xlabel("")
                    remove_legend(ax)
                    axes_list.append(ax)
                    current_ax_idx += 1

    format_axes(axs, outcome, fontsize=fontsize)
    remove_blank_axes(fig)

    rect = [0, 0.11, 1, 0.96] if is_train else [0, 0.17, 1, 0.95]
    plt.tight_layout(h_pad=8, w_pad=2, rect=rect)

    add_group_box(fig, axs[0, 0:3], "Missing Data")
    add_group_box(fig, axs[0, 3:6], "Input Noise")

    if label_noise_axes:
        add_group_box(fig, label_noise_axes, "Label Noise")
    if feature_selection_axes:
        add_group_box(fig, feature_selection_axes, "Feature Selection")
    if training_axes:
        add_group_box(fig, training_axes, "Training Data")
    if imbalance_axes:
        add_group_box(fig, imbalance_axes, "Imbalanced Data")
    if subgroup_axes:
        add_group_box(fig, subgroup_axes, f"{dev_dataset} Subgroups")

    if is_train:
        if ext1_axes and ext1_name:
            add_group_box(fig, ext1_axes, f"External Dataset : {ext1_name}")
        if ext2_axes and ext2_name:
            add_group_box(fig, ext2_axes, f"External Dataset : {ext2_name}")

    if legend:
        create_global_legend(fig, subset_models, MODELS_COLOR, y_pos=0.015)
    if save:
        plt.savefig(f"{filename}.pdf", format="pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ==============================================================================
# Pipeline Runner
# ==============================================================================

def run_all_configurations(
    experiments=None,
    configurations=None,
    model_set_name="complete",
    set_cs=None,
    save=True,
    show=False,
    legend=True,
):
    if set_cs is None:
        set_cs = ["TRAIN", "VAL", "ALL"]

    if experiments is None:
        experiments = [
            {
                "name": "classification",
                "hpo_dir": "../result/Classif_HPO/",
                "no_hpo_dir": "../result/Classif_no_HPO/",
                "outcomes": ["AUC", "Brier score"],
            },
            {
                "name": "regression",
                "hpo_dir": "../result/Reg_HPO/",
                "no_hpo_dir": "../result/Reg_no_HPO/",
                "outcomes": ["RMSE", "MAE", "R2"],
            },
        ]

    if configurations is None:
        configurations = [
            {"dev": "eICU", "ext1": "MIMIC-IV", "ext2": "MIMIC-III"},
            {"dev": "MIMIC-IV", "ext1": "MIMIC-III", "ext2": "eICU"},
            {"dev": "MIMIC-III", "ext1": "MIMIC-IV", "ext2": "eICU"},
        ]

    validate_model_configuration(MODEL_SUBSETS[model_set_name])

    has_train = any(str(sc).upper() == "TRAIN" for sc in set_cs)
    has_val_or_all = any(str(sc).upper() in {"VAL", "ALL", "VAL_ALL"} for sc in set_cs)

    for exp in experiments:
        exp_name = exp["name"]
        hpo_dir = exp.get("hpo_dir") or exp.get("base_dir")
        no_hpo_dir = exp.get("no_hpo_dir") or (hpo_dir.replace("_HPO", "_no_HPO") if hpo_dir else None)
        outcomes = exp["outcomes"]

        print(f"\n{'#'*80}\n# {exp_name.upper()}\n# HPO dir: {hpo_dir}\n# No-HPO dir: {no_hpo_dir}\n# Outcomes: {', '.join(outcomes)}\n{'#'*80}")

        for config in configurations:
            dev_name, ext1_name, ext2_name = config["dev"], config["ext1"], config["ext2"]
            print(f"\n{'='*70}\nExecuting: {exp_name.upper()} | Dev={dev_name} | Ext1={ext1_name} | Ext2={ext2_name}\n{'='*70}")

            # Resolve development folders in both HPO and no_HPO roots
            dev_dir_hpo = resolve_dataset_folder(hpo_dir, dev_name)
            dev_dir_no_hpo = resolve_dataset_folder(no_hpo_dir, dev_name)

            if not dev_dir_hpo and not dev_dir_no_hpo:
                print(f"[Warning] Development folder for {dev_name} not found in {hpo_dir} or {no_hpo_dir}. Skipping.")
                continue

            # Load and merge perturbation data
            data_perturb_hpo = load_perturbation_data(dev_dir_hpo) if dev_dir_hpo else {}
            data_perturb_no_hpo = load_perturbation_data(dev_dir_no_hpo) if dev_dir_no_hpo else {}
            data_perturb = merge_perturbation_data(data_perturb_hpo, data_perturb_no_hpo)

            df_ext1 = df_ext2 = pd.DataFrame()
            if has_train:
                # Resolve external folders for HPO
                ext1_hpo = resolve_dataset_folder(dev_dir_hpo, ext1_name) if dev_dir_hpo else None
                ext2_hpo = resolve_dataset_folder(dev_dir_hpo, ext2_name) if dev_dir_hpo else None
                df_ext1_hpo = load_external_subgroups(ext1_hpo)
                df_ext2_hpo = load_external_subgroups(ext2_hpo)

                # Resolve external folders for no_HPO
                ext1_no_hpo = resolve_dataset_folder(dev_dir_no_hpo, ext1_name) if dev_dir_no_hpo else None
                ext2_no_hpo = resolve_dataset_folder(dev_dir_no_hpo, ext2_name) if dev_dir_no_hpo else None
                df_ext1_no_hpo = load_external_subgroups(ext1_no_hpo)
                df_ext2_no_hpo = load_external_subgroups(ext2_no_hpo)

                # Merge external data
                df_ext1 = merge_hpo_and_no_hpo_dfs(df_ext1_hpo, df_ext1_no_hpo)
                df_ext2 = merge_hpo_and_no_hpo_dfs(df_ext2_hpo, df_ext2_no_hpo)

            for outcome in outcomes:
                # 1. Plot TRAIN figure (with integrated external subgroups in Row 3)
                if has_train:
                    print(f"\n[{exp_name.upper()}] Dev={dev_name} | Set=TRAIN | Outcome={outcome}")
                    save_fig_perturbations(
                        data_perturb=data_perturb,
                        dev_dataset=dev_name,
                        models_name=model_set_name,
                        outcome=outcome,
                        set_c="TRAIN",
                        save=save,
                        show=show,
                        legend=legend,
                        task=exp_name,
                        df_ext1=df_ext1,
                        df_ext2=df_ext2,
                        ext1_name=ext1_name,
                        ext2_name=ext2_name,
                    )

                # 2. Plot combined VAL & ALL figure on the same figure with explanatory subtitles
                if has_val_or_all:
                    print(f"\n[{exp_name.upper()}] Dev={dev_name} | Set=VAL & ALL (Combined) | Outcome={outcome}")
                    save_fig_val_all_perturbations(
                        data_perturb=data_perturb,
                        dev_dataset=dev_name,
                        models_name=model_set_name,
                        outcome=outcome,
                        save=save,
                        show=show,
                        legend=legend,
                        task=exp_name,
                    )


if __name__ == "__main__":
    run_all_configurations(
        model_set_name="complete",
        set_cs=["TRAIN", "VAL"],
        save=True,
        show=False,
        legend=True,
    )


# In[ ]:




