#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None


# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

CLASS_COLORS = {
    "Linear": "#0072B2",
    "Tree": "#009E73",
    "MLP": "#D55E00",
    "Transformer": "#CC79A7",
    "Other": "#7F7F7F",
}

MODEL_CLASSES = {
    "Logistic": "Linear",
    "Linear": "Linear",
    "Ridge": "Linear",
    "LASSO": "Linear",
    "Random Forest": "Tree",
    "Gradient Boosting": "Tree",
    "XGBoost": "Tree",
    "LightGBM": "Tree",
    "CatBoost": "Tree",
    "RealMLP": "MLP",
    "TabDPT": "Transformer",
    "TabPFNv3": "Transformer",
    "TabICLv2": "Transformer",
}

ALL_TRAIN_DFS = [
    "M3",
    "M4",
    "eICU",
]

TASK_CONFIG = {
    "Classif": {
        "metric": "AUC",
        "baseline": "Logistic",
        "higher_is_better": True,
    },
    # "Reg": {
    #     "metric": "MAE",
    #     "baseline": "Linear",
    #     "higher_is_better": False,
    # },
}

OUTPUT_DIR = Path("figures/")


# -----------------------------------------------------------------------------
# ICU Mapping / Subgroup Cleaning
# -----------------------------------------------------------------------------

ICU_MAPPING = {
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

# ICU services available across the three datasets.
OVERLAPPING_ICU_SERVICES = {
    "MICU",
    "SICU",
}

EXCLUDED_ICU = sorted(
    {mapped for mapped in ICU_MAPPING.values() if mapped not in OVERLAPPING_ICU_SERVICES}
)


def clean_subgroups(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize ICU subgroup categories across datasets.

    Only rows corresponding to the ICU subgroup are modified.
    ICU categories are mapped onto a common vocabulary and only:
        MICU, SICU
    are retained. All non-ICU subgroup variables are preserved unchanged.
    """
    df = df.copy()

    if "Variable" not in df.columns or "Category" not in df.columns:
        return df

    # Identify ICU subgroup rows (e.g. ICU_unit, ICU, ICU unit, ICU service)
    variable_normalized = df["Variable"].astype(str).str.strip().str.lower()
    is_icu = variable_normalized.str.contains("icu", regex=False, na=False)

    if not is_icu.any():
        return df

    # Map only ICU categories
    mapped_categories = (
        df.loc[is_icu, "Category"]
        .astype(str)
        .str.strip()
        .replace(ICU_MAPPING)
    )
    df.loc[is_icu, "Category"] = mapped_categories

    # Keep only ICU categories shared across datasets; preserve non-ICU rows
    keep = ~is_icu | df["Category"].isin(OVERLAPPING_ICU_SERVICES)
    return df.loc[keep].copy()


# -----------------------------------------------------------------------------
# Scenario Metadata
# -----------------------------------------------------------------------------

def parse_scenario_metadata(
    df: pd.DataFrame,
    ptype_raw: str,
    filename: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Standardize scenario metadata.

    Statistical hierarchy:
        Family -> perturbation scenario -> perturbation level/category -> raw observations

    For subgroups (e.g., scenario: ICU_unit; levels: MICU, SICU),
    Category is a LEVEL, not a separate scenario.
    """
    df = df.copy()
    stem = Path(filename).stem
    parts = stem.split("_")
    ptype_lower = ptype_raw.lower()

    # Family determination
    if ptype_lower in {"m4", "m3", "eicu"}:
        dataset = ptype_raw
        df["Perturbation_category"] = "Domain Shift"
        df["Family"] = f"Domain Shift ({dataset})"
        df["Perturbation"] = f"Domain Shift ({dataset})"
    elif ptype_lower == "subgroups":
        df["Perturbation_category"] = "Subgroups"
        df["Family"] = "Subgroups"
        df["Perturbation"] = "Subgroup Analysis"
    else:
        family = ptype_raw.replace("_", " ").title()
        df["Perturbation_category"] = family
        df["Family"] = family
        df["Perturbation"] = family

    # Default train / validation / all split
    fname_upper = filename.upper()
    if "TRAIN" in fname_upper:
        df["Set"] = "Train"
    elif "VAL" in fname_upper:
        df["Set"] = "Validation"
    else:
        df["Set"] = "All"

    # Missing data
    if ptype_lower == "missing_data":
        mech = parts[0].upper()
        df["Mechanism"] = mech
        df["Perturbation"] = f"Missing ({mech})"

    # Input noise
    elif ptype_lower == "input_noise":
        if len(parts) >= 4 and parts[3] in {"CONT", "CAT"}:
            ftype = parts[3].replace("CONT", "Continuous").replace("CAT", "Categorical")
        else:
            ftype = "All"
        df["Feature_type"] = ftype
        df["Perturbation"] = f"Input Noise ({ftype})"

    # Label noise
    elif ptype_lower == "label_noise":
        ltype = parts[0].upper()
        label_map = {
            "01": "(0 → 1)",
            "10": "(1 → 0)",
            "AGE": "P(AGE)",
            "PROXY": "Proxy",
            "RANDOM": "Random",
        }
        mapped_label = label_map.get(ltype, ltype)
        df["Label_type"] = mapped_label
        df["Perturbation"] = f"Label Noise ({mapped_label})"

        if ltype == "RANDOM" and "Noise level" in df.columns:
            df = df[df["Noise level"] <= 0.5].copy()

    # Subgroups / external datasets
    if ptype_lower in {"subgroups", "m4", "m3", "eicu"}:
        if "Variable" in df.columns and "Category" in df.columns:
            df = clean_subgroups(df)
            if df.empty:
                return df, []

            df["Set"] = df["Variable"]
            df["Noise level"] = df["Category"]
            df["Perturbation"] = (
                df["Perturbation"].astype(str) + " - " + df["Variable"].astype(str)
            )

    group_cols = [
        "Model",
        "Family",
        "Perturbation_category",
        "Perturbation",
        "Set",
        "Mechanism",
        "Feature_type",
        "Label_type",
        "Variable",
        "Category",
        "Noise level",
    ]
    valid_group_cols = [col for col in dict.fromkeys(group_cols) if col in df.columns]

    return df, valid_group_cols


# -----------------------------------------------------------------------------
# Relative Performance
# -----------------------------------------------------------------------------

def compute_relative_performance(
    baseline_performance: pd.Series,
    model_performance: pd.Series,
    higher_is_better: bool,
    ideal_performance: float = 1.0,
) -> pd.Series:
    """Compute scenario-level performance relative to the matching baseline."""
    baseline = pd.to_numeric(baseline_performance, errors="coerce")
    model = pd.to_numeric(model_performance, errors="coerce")
    result = pd.Series(np.nan, index=model.index, dtype=float)

    valid = (
        baseline.notna()
        & model.notna()
        & np.isfinite(baseline)
        & np.isfinite(model)
    )

    if not valid.any():
        return result

    with np.errstate(divide="ignore", invalid="ignore"):
        if higher_is_better:
            baseline_error = ideal_performance - baseline.loc[valid]
            model_error = ideal_performance - model.loc[valid]
            result.loc[valid] = baseline_error / model_error
        else:
            result.loc[valid] = baseline.loc[valid] / model.loc[valid]

    return result.replace([np.inf, -np.inf], np.nan)


# -----------------------------------------------------------------------------
# Scenario-Level Loading and Aggregation
# -----------------------------------------------------------------------------

def load_and_aggregate_scenarios(
    directory: Path,
    train_df: str,
    performance_col: str,
    baseline_model: str,
    higher_is_better: bool = True,
    compute_relative: bool = True,
    prediction_time_col: str = "Test pred time",
    train_time_col: str = "Train fit time",
) -> pd.DataFrame:
    """Load raw results and apply the hierarchical aggregation."""
    if not directory.exists():
        print(f"Directory does not exist: {directory}")
        return pd.DataFrame()

    all_scenarios = []

    for ptype_dir in sorted(directory.iterdir()):
        if not ptype_dir.is_dir():
            continue

        for file_path in sorted(ptype_dir.glob("*.csv")):
            if file_path.stem.lower() == train_df.lower():
                continue

            try:
                raw_df = pd.read_csv(file_path)

                if performance_col not in raw_df.columns:
                    print(f"Skipping {file_path}: missing {performance_col}")
                    continue

                df, group_cols = parse_scenario_metadata(
                    raw_df, ptype_dir.name, file_path.name
                )
                if df.empty or not group_cols:
                    continue

                # LEVEL 1: Mean within perturbation level/category
                level_agg_dict = {
                    "Level_Performance": (performance_col, "mean"),
                }
                if prediction_time_col in df.columns:
                    level_agg_dict["Level_Pred_Time"] = (prediction_time_col, "mean")
                if train_time_col in df.columns:
                    level_agg_dict["Level_Train_Time"] = (train_time_col, "median")

                level_df = (
                    df.groupby(group_cols, as_index=False, dropna=False)
                    .agg(**level_agg_dict)
                )

                # LEVEL 2: Aggregate levels across scenario
                level_columns = {"Noise level", "Category"}
                scenario_group_cols = [
                    col for col in group_cols if col not in level_columns
                ]

                scenario_agg_dict = {
                    "Scenario_Performance": ("Level_Performance", "mean"),
                    "Scenario_Performance_SD": ("Level_Performance", "std"),
                    "N_Levels": ("Level_Performance", "count"),
                }
                if "Level_Pred_Time" in level_df.columns:
                    scenario_agg_dict["Scenario_Pred_Time"] = ("Level_Pred_Time", "mean")
                if "Level_Train_Time" in level_df.columns:
                    scenario_agg_dict["Scenario_Train_Time"] = ("Level_Train_Time", "median")

                scenario_agg = (
                    level_df.groupby(scenario_group_cols, as_index=False, dropna=False)
                    .agg(**scenario_agg_dict)
                )

                # Scenario identifier
                scenario_keys = [col for col in scenario_group_cols if col != "Model"]
                scenario_agg["Scenario_ID"] = (
                    scenario_agg[scenario_keys].astype(str).agg(" | ".join, axis=1)
                )

                # Relative performance
                if compute_relative:
                    baseline_rows = scenario_agg[
                        scenario_agg["Model"] == baseline_model
                    ].copy()

                    if baseline_rows.empty:
                        scenario_agg["Baseline_Performance"] = np.nan
                        scenario_agg["Relative_Performance"] = np.nan
                    else:
                        baseline_map = baseline_rows[
                            scenario_keys + ["Scenario_Performance"]
                        ].rename(
                            columns={"Scenario_Performance": "Baseline_Performance"}
                        )

                        scenario_agg = scenario_agg.merge(
                            baseline_map,
                            on=scenario_keys,
                            how="left",
                            validate="many_to_one",
                        )
                        scenario_agg["Relative_Performance"] = compute_relative_performance(
                            baseline_performance=scenario_agg["Baseline_Performance"],
                            model_performance=scenario_agg["Scenario_Performance"],
                            higher_is_better=higher_is_better,
                            ideal_performance=1.0,
                        )
                else:
                    scenario_agg["Baseline_Performance"] = np.nan
                    scenario_agg["Relative_Performance"] = np.nan

                all_scenarios.append(scenario_agg)

            except Exception as exc:
                print(f"Error reading {file_path}: {exc}")

    if not all_scenarios:
        return pd.DataFrame()

    return pd.concat(all_scenarios, ignore_index=True)


# -----------------------------------------------------------------------------
# Geometric Mean
# -----------------------------------------------------------------------------

def gmean_safe(values: pd.Series) -> float:
    """Numerically stable geometric mean over finite positive values."""
    values = pd.to_numeric(values, errors="coerce")
    values = values[values.notna() & np.isfinite(values) & (values > 0)]

    if values.empty:
        return np.nan

    return float(np.exp(np.log(values.to_numpy()).mean()))


# -----------------------------------------------------------------------------
# Hierarchical Aggregation
# -----------------------------------------------------------------------------

def compute_hierarchical_model_summary(
    scenario_df: pd.DataFrame,
    baseline_model: str,
    use_relative: bool = True,
    min_family_coverage: float = 1.0,
    require_all_families: bool = True,
) -> pd.DataFrame:
    """Hierarchical robustness aggregation."""
    if scenario_df.empty:
        return pd.DataFrame()

    required_columns = {"Model", "Family", "Scenario_ID"}
    missing = required_columns - set(scenario_df.columns)
    if missing:
        raise ValueError(f"scenario_df is missing required columns: {sorted(missing)}")

    perf_col = "Relative_Performance" if use_relative else "Scenario_Performance"
    if perf_col not in scenario_df.columns:
        raise ValueError(f"scenario_df must contain '{perf_col}'.")

    # Expected scenario coverage
    baseline_scenarios = scenario_df[scenario_df["Model"] == baseline_model].copy()
    if baseline_scenarios.empty:
        raise ValueError(f"No rows found for baseline model '{baseline_model}'.")

    expected_scenarios = baseline_scenarios.groupby("Family")["Scenario_ID"].nunique()
    total_families = len(expected_scenarios)

    # Family score: arithmetic mean across unique scenarios
    family_summary = (
        scenario_df.groupby(["Model", "Family"], as_index=False, dropna=False)
        .agg(
            Family_Score=(perf_col, "mean"),
            N_Valid_Scenarios=(perf_col, "count"),
            N_Observed_Scenarios=("Scenario_ID", "nunique"),
        )
    )

    family_summary["Expected_Scenarios"] = family_summary["Family"].map(expected_scenarios)
    family_summary["Family_Coverage"] = np.where(
        family_summary["Expected_Scenarios"] > 0,
        family_summary["N_Valid_Scenarios"] / family_summary["Expected_Scenarios"],
        np.nan,
    )
    family_summary.loc[
        family_summary["Family_Coverage"] < min_family_coverage, "Family_Score"
    ] = np.nan

    # Overall mRS: Geometric (or arithmetic) mean across family scores
    agg_fn = gmean_safe if use_relative else "mean"
    overall_summary = (
        family_summary.groupby("Model", as_index=False)
        .agg(
            mRS=("Family_Score", agg_fn),
            N_families_used=("Family_Score", lambda v: int(v.notna().sum())),
        )
    )
    overall_summary["N_families_total"] = total_families

    if require_all_families:
        incomplete = (
            overall_summary["N_families_used"] < overall_summary["N_families_total"]
        )
        overall_summary.loc[incomplete, "mRS"] = np.nan

    # Prediction time median
    if "Scenario_Pred_Time" in scenario_df.columns:
        time_summary = (
            scenario_df.groupby("Model", as_index=False)
            .agg(Overall_Pred_Time_median=("Scenario_Pred_Time", "median"))
        )
        overall_summary = overall_summary.merge(time_summary, on="Model", how="left")

    # Train time median
    if "Scenario_Train_Time" in scenario_df.columns:
        train_time_summary = (
            scenario_df.groupby("Model", as_index=False)
            .agg(Overall_Train_Time_median=("Scenario_Train_Time", "median"))
        )
        overall_summary = overall_summary.merge(train_time_summary, on="Model", how="left")

    # Diagnostic arithmetic mean across family scores
    arithmetic_family_mean = (
        family_summary.groupby("Model")["Family_Score"]
        .mean()
        .rename("Relative_mean")
    )
    overall_summary = overall_summary.merge(arithmetic_family_mean, on="Model", how="left")

    return overall_summary


# -----------------------------------------------------------------------------
# Pareto Frontier
# -----------------------------------------------------------------------------

def compute_pareto_frontier(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> List[Tuple[float, float]]:
    """Pareto frontier: minimize x, maximize y."""
    valid = (
        df[[x_col, y_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values(by=[x_col, y_col], ascending=[True, False])
    )

    pareto_pts = []
    best_y = -np.inf

    for _, row in valid.iterrows():
        x = float(row[x_col])
        y = float(row[y_col])
        if y > best_y:
            pareto_pts.append((x, y))
            best_y = y

    return pareto_pts


# -----------------------------------------------------------------------------
# Pareto Plot
# -----------------------------------------------------------------------------

def plot_and_save_pareto(
    scenario_df_hpo: pd.DataFrame,
    scenario_df_no_hpo: Optional[pd.DataFrame],
    train_df: str,
    task: str,
    metric: str,
    baseline_model: str,
    output_dir: Path,
    x_metric: str = "Overall_Pred_Time_median",
    y_metric: str = "mRS",
    exclude_models: Optional[List[str]] = None,
    show_frontier: bool = True,
    show_connections: bool = True,
    use_relative: bool = True,
    min_family_coverage: float = 1.0,
    require_all_families: bool = True,
    figsize: Tuple[int, int] = (12, 12),
    show_linear_inset: bool = True,
):
    """Create Pareto plot."""
    exclude = set(exclude_models or [])

    # HPO summary
    sum_hpo = compute_hierarchical_model_summary(
        scenario_df_hpo,
        baseline_model=baseline_model,
        use_relative=use_relative,
        min_family_coverage=min_family_coverage,
        require_all_families=require_all_families,
    )
    if not sum_hpo.empty:
        sum_hpo = sum_hpo[~sum_hpo["Model"].isin(exclude)].copy()

    # No-HPO summary
    sum_no_hpo = pd.DataFrame()
    if scenario_df_no_hpo is not None and not scenario_df_no_hpo.empty:
        sum_no_hpo = compute_hierarchical_model_summary(
            scenario_df_no_hpo,
            baseline_model=baseline_model,
            use_relative=use_relative,
            min_family_coverage=min_family_coverage,
            require_all_families=require_all_families,
        )
        if not sum_no_hpo.empty:
            sum_no_hpo = sum_no_hpo[~sum_no_hpo["Model"].isin(exclude)].copy()

    # Remove HPO Logistic
    if not sum_hpo.empty:
        sum_hpo = sum_hpo[sum_hpo["Model"] != "Logistic"].copy()

    # Validation
    required_columns = {x_metric, y_metric, "Model"}
    missing = required_columns - set(sum_hpo.columns)
    if missing:
        raise ValueError(f"Missing columns in summary: {sorted(missing)}")

    # Plot helpers
    def row_is_valid(row):
        return (
            pd.notna(row[x_metric])
            and pd.notna(row[y_metric])
            and np.isfinite(row[x_metric])
            and np.isfinite(row[y_metric])
            and row[x_metric] > 0
        )

    def scatter_hpo(target_ax, data, size=90):
        for _, row in data.iterrows():
            if not row_is_valid(row):
                continue
            model = row["Model"]
            model_class = MODEL_CLASSES.get(model, "Other")
            target_ax.scatter(
                row[x_metric],
                row[y_metric],
                color=CLASS_COLORS.get(model_class, "gray"),
                s=size,
                alpha=0.9,
                zorder=5,
            )

    def scatter_no_hpo(target_ax, data, size=90):
        if data.empty:
            return
        for _, row in data.iterrows():
            if not row_is_valid(row):
                continue
            model = row["Model"]
            model_class = MODEL_CLASSES.get(model, "Other")
            target_ax.scatter(
                row[x_metric],
                row[y_metric],
                facecolors="white",
                edgecolors=CLASS_COLORS.get(model_class, "gray"),
                linewidth=1.5,
                s=size,
                alpha=0.95,
                zorder=4,
            )

    def draw_connections(target_ax, hpo_data, no_hpo_data, line_width=1.2, alpha=0.6):
        if not show_connections or hpo_data.empty or no_hpo_data.empty:
            return

        common = pd.merge(
            hpo_data[["Model", x_metric, y_metric]],
            no_hpo_data[["Model", x_metric, y_metric]],
            on="Model",
            suffixes=("_hpo", "_no_hpo"),
        )

        for _, row in common.iterrows():
            values = [
                row[f"{x_metric}_hpo"],
                row[f"{y_metric}_hpo"],
                row[f"{x_metric}_no_hpo"],
                row[f"{y_metric}_no_hpo"],
            ]
            if any(pd.isna(v) for v in values) or not all(np.isfinite(v) for v in values):
                continue

            model_class = MODEL_CLASSES.get(row["Model"], "Other")
            target_ax.annotate(
                "",
                xy=(row[f"{x_metric}_hpo"], row[f"{y_metric}_hpo"]),
                xytext=(row[f"{x_metric}_no_hpo"], row[f"{y_metric}_no_hpo"]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=CLASS_COLORS.get(model_class, "gray"),
                    lw=line_width,
                    alpha=alpha,
                ),
                zorder=2,
            )

    # Main figure setup
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale("log")

    scatter_hpo(ax, sum_hpo, size=90)
    scatter_no_hpo(ax, sum_no_hpo, size=90)
    draw_connections(ax, sum_hpo, sum_no_hpo, line_width=1.2, alpha=0.6)

    # Main labels (suppress linear models if inset is displayed)
    texts = []
    candidate_rows = (
        pd.concat(
            [
                sum_hpo.assign(_plot_priority=0),
                sum_no_hpo.assign(_plot_priority=1),
            ],
            ignore_index=True,
        )
        .sort_values("_plot_priority")
        .drop_duplicates(subset=["Model"], keep="first")
    )

    for _, row in candidate_rows.iterrows():
        if not row_is_valid(row):
            continue
        model = row["Model"]
        model_class = MODEL_CLASSES.get(model, "Other")
        if show_linear_inset and model_class == "Linear":
            continue

        texts.append(ax.text(row[x_metric], row[y_metric], model, fontsize=9))

    if adjust_text and texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

    # Pareto frontier
    if show_frontier:
        frontier_df = pd.concat([sum_hpo, sum_no_hpo], ignore_index=True)
        frontier_df = (
            frontier_df[["Model", x_metric, y_metric]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        frontier_df = frontier_df[frontier_df[x_metric] > 0]
        pts = compute_pareto_frontier(frontier_df, x_col=x_metric, y_col=y_metric)

        if pts:
            px, py = zip(*pts)
            ax.step(
                px,
                py,
                where="post",
                linestyle="--",
                alpha=0.7,
                linewidth=1.8,
                color="black",
                zorder=3,
            )

    # Baseline reference line
    if use_relative:
        ax.axhline(1.0, color="gray", linestyle=":", lw=1.5, alpha=0.6)

    # Linear-model inset
    if show_linear_inset:
        linear_hpo = sum_hpo[
            sum_hpo["Model"].map(MODEL_CLASSES) == "Linear"
        ].copy()

        if not sum_no_hpo.empty and "Model" in sum_no_hpo.columns:
            linear_no_hpo = sum_no_hpo[
                sum_no_hpo["Model"].map(MODEL_CLASSES) == "Linear"
            ].copy()
        else:
            linear_no_hpo = pd.DataFrame(columns=sum_hpo.columns)

        linear_all = pd.concat([linear_hpo, linear_no_hpo], ignore_index=True)
        linear_all = (
            linear_all[["Model", x_metric, y_metric]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        linear_all = linear_all[linear_all[x_metric] > 0]

        if not linear_all.empty:
            axins = ax.inset_axes([0.61, 0.12, 0.27, 0.25])
            axins.set_xscale("log")

            scatter_hpo(axins, linear_hpo, size=42)
            scatter_no_hpo(axins, linear_no_hpo, size=42)
            draw_connections(axins, linear_hpo, linear_no_hpo, line_width=0.7, alpha=0.5)

            if use_relative:
                axins.axhline(1.0, color="gray", linestyle=":", lw=0.8, alpha=0.5)

            x_values = linear_all[x_metric].astype(float).to_numpy()
            y_values = linear_all[y_metric].astype(float).to_numpy()

            log_x = np.log10(x_values)
            log_x_min, log_x_max = log_x.min(), log_x.max()
            log_x_padding = (
                0.15 if np.isclose(log_x_min, log_x_max) else 0.15 * (log_x_max - log_x_min)
            )
            x_lower = 10 ** (log_x_min - log_x_padding)
            x_upper = 10 ** (log_x_max + log_x_padding)

            y_min, y_max = y_values.min(), y_values.max()
            if np.isclose(y_min, y_max):
                y_padding = max(0.02, abs(y_min) * 0.05)
            else:
                y_padding = 0.15 * (y_max - y_min)

            axins.set_xlim(x_lower, x_upper)
            axins.set_ylim(y_min - y_padding, y_max + y_padding)

            # Inset text labels
            linear_label_rows = (
                pd.concat(
                    [
                        linear_hpo.assign(_plot_priority=0),
                        linear_no_hpo.assign(_plot_priority=1),
                    ],
                    ignore_index=True,
                )
                .sort_values("_plot_priority")
                .drop_duplicates(subset=["Model"], keep="first")
            )

            inset_texts = []
            for _, row in linear_label_rows.iterrows():
                if not row_is_valid(row):
                    continue
                inset_texts.append(
                    axins.text(row[x_metric], row[y_metric], row["Model"], fontsize=6.5)
                )

            if adjust_text and inset_texts:
                adjust_text(
                    inset_texts,
                    ax=axins,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.35),
                )

            axins.set_title("Linear models", fontsize=8, pad=2)
            axins.set_xticks([])
            axins.set_yticks([])
            axins.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )
            axins.set_xlabel("")
            axins.set_ylabel("")
            axins.grid(False)

            ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.55)

    # Axis formatting
    def time_fmt(x, _):
        if x < 1:
            return f"{x:.2g}s"
        if x < 60:
            return f"{x:.1f}s"
        return f"{x / 60:.1f}m"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(time_fmt))
    ax.set_xlabel("Median prediction time across scenarios", fontsize=13, labelpad=8)
    if use_relative:
        ax.set_ylabel(
            f"Mean Robustness Score", fontsize=13
        )
    else:
        ax.set_ylabel(f"Aggregate {metric}", fontsize=13)

    # Direction annotation
    ax.annotate(
        "Faster & more robust",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        xytext=(0.22, 0.80),
        textcoords="axes fraction",
        arrowprops=dict(
            facecolor="black",
            shrink=0.05,
            width=1,
            headwidth=8,
            alpha=0.7,
        ),
        fontsize=11,
        ha="center",
        va="center",
    )

    # Legends
    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            label=model_class,
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=8,
        )
        for model_class, color in CLASS_COLORS.items()
        if model_class != "Other"
    ]

    style_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="gray",
            markeredgecolor="gray",
            markersize=8,
            label="With HPO",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="gray",
            markeredgewidth=1.5,
            markersize=8,
            label="Without HPO",
        ),
    ]

    if show_frontier:
        style_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.8,
                label="Pareto frontier",
            )
        )


    legend_models = ax.legend(
        handles=class_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),  # Sits right below the x-axis label
        bbox_transform=ax.transAxes,
        ncol=len(class_handles),
        fontsize=9,
        frameon=False,
        columnspacing=1.8,
        handletextpad=0.5,
    )

    legend_styles = ax.legend(
        handles=style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),  # Sits right below legend_models
        bbox_transform=ax.transAxes,
        ncol=len(style_handles),
        fontsize=9,
        frameon=False,
        columnspacing=1.8,
        handletextpad=0.5,
    )

    ax.add_artist(legend_models)

    ax.grid(True, which="major", linestyle="--", alpha=0.4)


    save_path = output_dir / f"Pareto_{train_df}_{task}_{metric}.pdf"

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved hierarchical Pareto plot: {save_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for train_df in ALL_TRAIN_DFS:
        for task, cfg in TASK_CONFIG.items():
            metric = cfg["metric"]
            baseline = cfg["baseline"]
            higher_is_better = cfg["higher_is_better"]

            dir_with_hpo = Path(f"../result/{task}_HPO/{train_df}")
            dir_without_hpo = Path(f"../result/{task}_no_HPO/{train_df}")

            print(f"\nProcessing {train_df} | Task: {task} | Metric: {metric}")

            # Load with HPO
            scenario_results_hpo = load_and_aggregate_scenarios(
                directory=dir_with_hpo,
                train_df=train_df,
                performance_col=metric,
                baseline_model=baseline,
                higher_is_better=higher_is_better,
                compute_relative=True,
                prediction_time_col="Test pred time",
                train_time_col="Train fit time",
            )

            # Load without HPO
            scenario_results_no_hpo = load_and_aggregate_scenarios(
                directory=dir_without_hpo,
                train_df=train_df,
                performance_col=metric,
                baseline_model=baseline,
                higher_is_better=higher_is_better,
                compute_relative=True,
                prediction_time_col="Test pred time",
                train_time_col="Train fit time",
            )

            if scenario_results_hpo.empty:
                print(f"Skipping {train_df} ({task}): no valid HPO scenarios.")
                continue

            print("HPO model counts:")
            print(scenario_results_hpo["Model"].value_counts())

            if not scenario_results_no_hpo.empty:
                print("No-HPO model counts:")
                print(scenario_results_no_hpo["Model"].value_counts())

            # Diagnostic summary
            diagnostic_summary = compute_hierarchical_model_summary(
                scenario_results_hpo,
                baseline_model=baseline,
                use_relative=True,
                min_family_coverage=1.0,
                require_all_families=True,
            )

            if not diagnostic_summary.empty:
                print("\nHPO hierarchical summary:")
                diagnostic_columns = [
                    "Model",
                    "mRS",
                    "Relative_mean",
                    "N_families_used",
                    "N_families_total",
                ]
                if "Overall_Pred_Time_median" in diagnostic_summary.columns:
                    diagnostic_columns.append("Overall_Pred_Time_median")

                print(
                    diagnostic_summary[diagnostic_columns]
                    .sort_values("mRS", ascending=False)
                    .to_string(index=False)
                )

            # Plot
            plot_and_save_pareto(
                scenario_df_hpo=scenario_results_hpo,
                scenario_df_no_hpo=scenario_results_no_hpo,
                train_df=train_df,
                task=task,
                metric=metric,
                baseline_model=baseline,
                output_dir=OUTPUT_DIR,
                x_metric="Overall_Pred_Time_median",
                y_metric="mRS",
                exclude_models=[],
                show_frontier=True,
                show_connections=True,
                show_linear_inset=True,
                use_relative=True,
                min_family_coverage=1.0,
                require_all_families=True,
            )

            all_results[(train_df, task)] = {
                "scenario_hpo": scenario_results_hpo,
                "scenario_no_hpo": scenario_results_no_hpo,
                "summary_hpo": diagnostic_summary,
            }

    print("\nAll hierarchical Pareto frontier plots generated and saved successfully.")
    return all_results


if __name__ == "__main__":
    all_results = main()

