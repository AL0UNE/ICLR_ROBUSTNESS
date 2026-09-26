#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Multi-perturbation Elo ratings with bootstrap confidence intervals (regression).

Regression sibling of `elo_bootstrap.py`. Scored on MAE only (orientation -1,
i.e. lower is better -- same convention as Brier score in the classification
script), anchored on the "Linear" model instead of "Logistic".

Methodology
-----------
Phase 1:
    Construct pairwise duels at the finest context grain
    c = (run, development set, composite, fold).

    The paired-significance tie rule is evaluated at the coarser
    (run, development set, composite) level across folds.

    Model failures (NaN metric / missing fold row) are scored as decisive
    losses against models that successfully completed the same context.
    Two failures in the same context are treated as a tie.

Phase 2:
    Compute clean-performance Elo from unperturbed baseline duels.

Phase 3:
    Compute composite-perturbation Elo from all perturbed-composite duels
    pooled with uniform weights.

Phase 4:
    Estimate uncertainty by bootstrap resampling.

    - Composite Elo:
      Resample composite perturbations with replacement within each run.
      All folds and development datasets associated with a sampled composite
      are retained together.

    - Clean Elo:
      Resample cross-validation folds with replacement within each
      (run, development dataset, clean context), retaining all models evaluated
      on a sampled fold.

Both Elo views use an L2-regularised Bradley-Terry model and the mapping

    R = 400 / ln(10) * theta + R0

with a reference model pinned to a fixed Elo rating.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

# =============================================================================
# Constants
# =============================================================================

ELO_SCALE = 400.0 / math.log(10.0)

# metric key -> (results column, orientation)
# +1 = higher is better, -1 = lower is better
METRICS = {
    "mae": ("MAE", -1),
}

CLEAN_TEST_NAME = "CLEAN_BASELINE"
MULTI_TEST_NAME = "MULTI_PERTURBATION_COMPOSITE"


# =============================================================================
# Load results
# =============================================================================

def load_run_results(res_dir: str) -> pd.DataFrame:
    """Load one res_multi_* directory into a tidy per-fold dataframe.

    Returns
    -------
    DataFrame with columns:
        run, dev_set, composite, is_clean, fold, Model, MAE

    Notes
    -----
    Fold indices are reconstructed positionally because the benchmark runner
    writes rows model-outer / fold-inner within each composite.

    Regression result files are saved as `{test_name}_{n_training_sample}.csv`
    (see `helper/helper.py:save_results`, `n_tr=cfg.n_training_sample`) rather
    than the plain `{test_name}.csv` the classification benchmark writes, so
    the file is located with a glob instead of a fixed name.
    """
    run = os.path.basename(os.path.normpath(res_dir))
    frames = []

    for dev_key in sorted(os.listdir(res_dir)):
        dev_dir = os.path.join(res_dir, dev_key)
        if not os.path.isdir(dev_dir):
            continue

        for subdir, test_name, is_clean in (
            ("clean", CLEAN_TEST_NAME, True),
            ("multi_perturbation", MULTI_TEST_NAME, False),
        ):
            matches = sorted(glob.glob(os.path.join(dev_dir, subdir, f"{test_name}*.csv")))
            if not matches:
                continue
            if len(matches) > 1:
                raise FileNotFoundError(
                    f"Ambiguous {test_name} results under {os.path.join(dev_dir, subdir)}: "
                    f"found {len(matches)} matching files {matches}; expected exactly one."
                )
            csv_path = matches[0]

            df = pd.read_csv(
                csv_path,
                usecols=["Model", "Noise level", "MAE"],
            ).rename(columns={"Noise level": "composite"})

            df["run"] = run
            df["dev_set"] = dev_key
            df["is_clean"] = is_clean

            # Reconstruct fold ID from output ordering
            df["fold"] = df.groupby(["composite", "Model"]).cumcount()
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No {CLEAN_TEST_NAME}*.csv or {MULTI_TEST_NAME}*.csv files found under {res_dir}"
        )

    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Tie rule
# =============================================================================

def _significant(diffs, alpha: float, test: str) -> bool:
    """Paired significance test for orientation-adjusted fold differences.

    Parameters
    ----------
    diffs : array-like
        Per-fold differences after orienting the metric so that larger values
        always indicate better performance.
    alpha : float
        Significance level.
    test : {"ttest", "wilcoxon"}
        Paired comparison procedure.

    Returns
    -------
    bool
        True if the difference is considered statistically significant.
    """
    diffs = np.asarray(diffs, dtype=float)

    if len(diffs) < 2 or np.all(diffs == 0.0):
        return False

    # All folds favor one model by exactly the same non-zero amount.
    # Classical variance-based tests are undefined here, but direction is unanimous.
    if np.ptp(diffs) == 0.0:
        return True

    if test == "wilcoxon":
        try:
            pvalue = stats.wilcoxon(diffs).pvalue
        except ValueError:
            return False
    elif test == "ttest":
        pvalue = stats.ttest_1samp(diffs, 0.0).pvalue
    else:
        raise ValueError(
            f"Unknown paired test {test!r}. Expected 'ttest' or 'wilcoxon'."
        )

    if not np.isfinite(pvalue):
        return False

    return bool(pvalue < alpha)


# =============================================================================
# Build pairwise duels
# =============================================================================

def build_duels(
    df: pd.DataFrame,
    metric_col: str,
    orientation: int,
    alpha: float = 0.05,
    test: str = "ttest",
) -> pd.DataFrame:
    """Build fold-level pairwise duel outcomes.

    Duel scoring:
        y = 1.0   model_i wins
        y = 0.5   tie
        y = 0.0   model_i loses

    Statistical significance is evaluated once for each model pair within a
    (run, dev_set, composite) group using the paired fold-level differences.
    If not significant, all fold-level comparisons for that pair are ties.

    Failure handling:
        - i succeeds, j fails -> i wins
        - i fails, j succeeds -> i loses
        - both fail           -> tie
    """
    duels = []
    group_cols = ["run", "dev_set", "composite"]

    for _, group in df.groupby(group_cols, sort=False):
        per_model = {
            model: dict(zip(g["fold"], pd.to_numeric(g[metric_col], errors="coerce")))
            for model, g in group.groupby("Model", sort=False)
        }
        folds = sorted(set(group["fold"]))

        for model_i, model_j in combinations(sorted(per_model), 2):
            fi = per_model[model_i]
            fj = per_model[model_j]

            ok_i = {f for f in folds if np.isfinite(fi.get(f, np.nan))}
            ok_j = {f for f in folds if np.isfinite(fj.get(f, np.nan))}
            common = sorted(ok_i & ok_j)

            differences = [orientation * (fi[f] - fj[f]) for f in common]
            significant = _significant(differences, alpha=alpha, test=test)

            for f in folds:
                # Both models succeeded
                if f in ok_i and f in ok_j:
                    if not significant:
                        y = 0.5
                    else:
                        d = orientation * (fi[f] - fj[f])
                        if d > 0:
                            y = 1.0
                        elif d < 0:
                            y = 0.0
                        else:
                            y = 0.5
                # i succeeds, j fails
                elif f in ok_i:
                    y = 1.0
                # j succeeds, i fails
                elif f in ok_j:
                    y = 0.0
                # Both fail
                else:
                    y = 0.5

                duels.append((model_i, model_j, y))

    return pd.DataFrame(duels, columns=["model_i", "model_j", "y"])


# =============================================================================
# Bradley-Terry model
# =============================================================================

def fit_bradley_terry(
    duels: pd.DataFrame,
    model_names: list[str],
    C: float = 1.0,
) -> pd.Series:
    """Fit an L2-regularised Bradley-Terry model.

    Parameters
    ----------
    duels : DataFrame
        Columns: model_i, model_j, y
    model_names : sequence of str
        Models included in the rating.
    C : float
        Inverse regularisation strength (lambda = 1 / C).

    Returns
    -------
    pd.Series
        Estimated latent Bradley-Terry strengths theta.
    """
    if duels.empty:
        raise ValueError("Cannot fit Bradley-Terry model: duel table is empty.")

    model_names = list(model_names)
    index = {model: k for k, model in enumerate(model_names)}

    i = duels["model_i"].map(index).to_numpy()
    j = duels["model_j"].map(index).to_numpy()
    y = duels["y"].to_numpy(dtype=float)

    if np.any(pd.isna(i)) or np.any(pd.isna(j)):
        raise ValueError("Duel table contains a model absent from model_names.")

    i = i.astype(int)
    j = j.astype(int)
    lam = 1.0 / C

    def nll_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Stable penalised negative log-likelihood and gradient."""
        delta = theta[i] - theta[j]
        p = special.expit(delta)

        # Logistic cross-entropy: log(1 + exp(delta)) - y * delta
        nll_data = np.sum(np.logaddexp(0.0, delta) - y * delta)
        penalty = 0.5 * lam * np.dot(theta, theta)
        nll = nll_data + penalty

        residual = p - y
        grad = np.zeros(len(model_names))
        np.add.at(grad, i, residual)
        np.add.at(grad, j, -residual)
        grad += lam * theta

        return nll, grad

    result = optimize.minimize(
        nll_grad,
        np.zeros(len(model_names)),
        jac=True,
        method="L-BFGS-B",
    )

    if not result.success:
        warnings.warn(f"Bradley-Terry fit did not fully converge: {result.message}")

    return pd.Series(result.x, index=model_names)


# =============================================================================
# Convert Bradley-Terry strength to Elo
# =============================================================================

def scale_to_elo(
    theta: pd.Series,
    anchor_model: str | None = None,
    anchor_rating: float = 1000.0,
) -> pd.Series:
    """Map Bradley-Terry strengths onto the conventional Elo scale.

        Elo = 400 / ln(10) * theta + offset

    If anchor_model is available, its rating is fixed at anchor_rating.
    Otherwise, the mean rating is pinned to anchor_rating.
    """
    elo = ELO_SCALE * theta

    if anchor_model is not None and anchor_model in elo.index:
        offset = anchor_rating - elo[anchor_model]
    else:
        if anchor_model is not None:
            warnings.warn(
                f"Anchor model {anchor_model!r} is not among the rated models. "
                "Pinning the mean rating instead."
            )
        offset = anchor_rating - elo.mean()

    return elo + offset


# =============================================================================
# Helper: fit one Elo view
# =============================================================================

def fit_elo_view(
    df: pd.DataFrame,
    metric_col: str,
    orientation: int,
    alpha: float = 0.05,
    test: str = "ttest",
    C: float = 1.0,
    anchor_model: str = "Linear",
    anchor_rating: float = 1000.0,
    model_names: list[str] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Build duels and fit one Elo model."""
    if model_names is None:
        model_names = sorted(df["Model"].unique())

    duels = build_duels(
        df,
        metric_col=metric_col,
        orientation=orientation,
        alpha=alpha,
        test=test,
    )
    theta = fit_bradley_terry(duels, model_names=model_names, C=C)
    elo = scale_to_elo(
        theta,
        anchor_model=anchor_model,
        anchor_rating=anchor_rating,
    )

    return elo, duels


# =============================================================================
# Bootstrap resampling
# =============================================================================

def _bootstrap_global_sample(
    sub: pd.DataFrame,
    rng: np.random.Generator,
    bootstrap_id: int,
) -> pd.DataFrame:
    """Generate one bootstrap sample for composite-perturbation Elo.

    Composite perturbations are resampled with replacement within each run.
    All dev sets and folds associated with a sampled composite move together,
    preserving the hierarchy: run -> composite -> dev_set -> fold.
    """
    sampled_parts = []

    for _, run_df in sub.groupby("run", sort=False):
        composites = run_df["composite"].drop_duplicates().to_numpy()
        sampled_composites = rng.choice(composites, size=len(composites), replace=True)

        for sample_position, composite in enumerate(sampled_composites):
            block = run_df[run_df["composite"] == composite].copy()
            # Assign distinct cluster label for duplicate draws
            block["composite"] = f"boot{bootstrap_id}_{sample_position}_{composite}"
            sampled_parts.append(block)

    if not sampled_parts:
        raise ValueError("No perturbed observations available for global bootstrap.")

    return pd.concat(sampled_parts, ignore_index=True)


def _bootstrap_clean_sample(
    sub: pd.DataFrame,
    rng: np.random.Generator,
    bootstrap_id: int,
) -> pd.DataFrame:
    """Generate one bootstrap sample for clean Elo.

    Folds are resampled with replacement within each
    (run, development dataset, clean composite), retaining all models together.
    """
    del bootstrap_id  # Unused, retained for signature parity
    sampled_parts = []
    group_cols = ["run", "dev_set", "composite"]

    for _, group in sub.groupby(group_cols, sort=False):
        folds = group["fold"].drop_duplicates().to_numpy()
        sampled_folds = rng.choice(folds, size=len(folds), replace=True)

        for sample_position, original_fold in enumerate(sampled_folds):
            block = group[group["fold"] == original_fold].copy()
            block["fold"] = sample_position
            sampled_parts.append(block)

    if not sampled_parts:
        raise ValueError("No clean observations available for clean bootstrap.")

    return pd.concat(sampled_parts, ignore_index=True)


def _bootstrap_seed(random_state: int, is_clean: bool, metric_idx: int) -> int:
    """Deterministic per-(view, metric) bootstrap seed.

    Kept separate from the base random_state so the clean and global views,
    and the different metrics within each view, draw independent bootstrap
    resamples.
    """
    return random_state + 10000 * int(is_clean) + metric_idx


def _replicate_rng(random_state: int, bootstrap_id: int) -> np.random.Generator:
    """Independent, reproducible RNG for a single bootstrap replicate.
    """
    return np.random.default_rng([random_state, bootstrap_id])


def _default_n_jobs() -> int:
    """Pick a worker count that respects a SLURM allocation when present."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return os.cpu_count() or 1


def _fit_bootstrap_replicate(
    boot_df: pd.DataFrame,
    *,
    metric_col: str,
    orientation: int,
    alpha: float,
    test: str,
    C: float,
    anchor_model: str,
    anchor_rating: float,
    model_names: list[str],
) -> pd.Series:
    """Fit one bootstrap replicate. Top-level so it can be pickled for
    ProcessPoolExecutor."""
    elo, _ = fit_elo_view(
        boot_df,
        metric_col=metric_col,
        orientation=orientation,
        alpha=alpha,
        test=test,
        C=C,
        anchor_model=anchor_model,
        anchor_rating=anchor_rating,
        model_names=model_names,
    )
    return elo


def _run_bootstrap_replicates(
    sub: pd.DataFrame,
    is_clean: bool,
    metric_col: str,
    orientation: int,
    model_names: list[str],
    replicate_ids,
    alpha: float = 0.05,
    test: str = "ttest",
    C: float = 1.0,
    anchor_model: str = "Linear",
    anchor_rating: float = 1000.0,
    random_state: int = 42,
    n_jobs: int | None = None,
    n_boot_total: int | None = None,
    log_every: int = 500,
    log_prefix: str = "",
) -> pd.DataFrame:
    """Fit an arbitrary subset of bootstrap replicates.

    `replicate_ids` may be any subset of range(n_boot) since each replicate draws
    its own sample from an independent RNG (`_replicate_rng`). `n_boot_total`
    is only used to make progress logs read as "done/n_boot" when computing
    one block of a bigger run.

    Returns a DataFrame of Elo draws indexed by replicate id (failed
    replicates are dropped, with a warning).
    """
    replicate_ids = list(replicate_ids)
    n_boot_total = n_boot_total or len(replicate_ids)
    sampler = _bootstrap_clean_sample if is_clean else _bootstrap_global_sample

    boot_dfs = {
        b: sampler(sub, rng=_replicate_rng(random_state, b), bootstrap_id=b)
        for b in replicate_ids
    }

    fit_replicate = partial(
        _fit_bootstrap_replicate,
        metric_col=metric_col,
        orientation=orientation,
        alpha=alpha,
        test=test,
        C=C,
        anchor_model=anchor_model,
        anchor_rating=anchor_rating,
        model_names=model_names,
    )

    if n_jobs is None:
        n_jobs = _default_n_jobs()
    n_jobs = max(1, min(n_jobs, len(replicate_ids)))

    results_by_id: dict[int, pd.Series] = {}
    failures = 0
    done = 0

    def log_progress() -> None:
        nonlocal done
        done += 1
        if log_every > 0 and (done % log_every == 0 or done == len(replicate_ids)):
            print(
                f"{log_prefix}Bootstrap replicate {done}/{len(replicate_ids)} "
                f"in this batch done (n_boot={n_boot_total})"
            )

    if n_jobs <= 1:
        for b in replicate_ids:
            try:
                results_by_id[b] = fit_replicate(boot_dfs[b])
            except Exception as exc:
                failures += 1
                warnings.warn(f"Bootstrap replicate {b} failed: {exc}")
            log_progress()
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(fit_replicate, boot_dfs[b]): b for b in replicate_ids
            }
            for future in as_completed(futures):
                b = futures[future]
                try:
                    results_by_id[b] = future.result()
                except Exception as exc:
                    failures += 1
                    warnings.warn(f"Bootstrap replicate {b} failed: {exc}")
                log_progress()

    if failures > 0:
        warnings.warn(f"{failures}/{len(replicate_ids)} bootstrap replicates failed in this batch.")

    if not results_by_id:
        raise RuntimeError("All bootstrap Elo fits failed.")

    ordered_ids = sorted(results_by_id)
    bootstrap_df = pd.DataFrame(
        [results_by_id[b] for b in ordered_ids], index=ordered_ids
    ).reindex(columns=model_names)
    bootstrap_df.index.name = "replicate"
    return bootstrap_df


def summarize_bootstrap(bootstrap_df: pd.DataFrame, ci_level: float = 0.95) -> pd.DataFrame:
    """Median/CI/success-count summary from a table of bootstrap Elo draws."""
    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be between 0 and 1.")

    q_low = (1.0 - ci_level) / 2.0
    q_high = (1.0 + ci_level) / 2.0

    return pd.DataFrame({
        "bootstrap_median": bootstrap_df.median(axis=0),
        "ci_lower": bootstrap_df.quantile(q_low, axis=0),
        "ci_upper": bootstrap_df.quantile(q_high, axis=0),
        "n_boot_success": bootstrap_df.notna().sum(axis=0),
    })


def bootstrap_elo_view(
    sub: pd.DataFrame,
    is_clean: bool,
    metric_col: str,
    orientation: int,
    model_names: list[str],
    n_boot: int = 1000,
    ci_level: float = 0.95,
    alpha: float = 0.05,
    test: str = "ttest",
    C: float = 1.0,
    anchor_model: str = "Linear",
    anchor_rating: float = 1000.0,
    random_state: int = 42,
    n_jobs: int | None = None,
    log_every: int = 500,
    log_prefix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap one Elo view on a single machine (all n_boot replicates).

    For splitting n_boot across a SLURM array instead, see
    `run_bootstrap_block` / `merge_bootstrap_blocks`.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1.")

    bootstrap_df = _run_bootstrap_replicates(
        sub,
        is_clean,
        metric_col,
        orientation,
        model_names,
        replicate_ids=range(n_boot),
        alpha=alpha,
        test=test,
        C=C,
        anchor_model=anchor_model,
        anchor_rating=anchor_rating,
        random_state=random_state,
        n_jobs=n_jobs,
        n_boot_total=n_boot,
        log_every=log_every,
        log_prefix=log_prefix,
    )

    n_failed = n_boot - len(bootstrap_df)
    if n_failed > 0:
        warnings.warn(f"{n_failed}/{n_boot} bootstrap replicates failed.")

    summary = summarize_bootstrap(bootstrap_df, ci_level=ci_level)
    return bootstrap_df, summary


# =============================================================================
# Main function
# =============================================================================

def compute_elo_table(
    res_dirs: list[str],
    alpha: float = 0.05,
    C: float = 1.0,
    anchor_model: str = "Linear",
    anchor_rating: float = 1000.0,
    test: str = "ttest",
    n_boot: int = 10000,
    ci_level: float = 0.95,
    random_state: int = 42,
    return_bootstrap: bool = False,
    n_jobs: int | None = None,
    log_every: int = 500,
) -> tuple[pd.DataFrame, dict] | tuple[pd.DataFrame, dict, dict]:
    """Compute clean and composite-perturbation Elo ratings with bootstrap CIs."""
    # -----------------------------------------------------------------
    # Load all benchmark runs
    # -----------------------------------------------------------------
    df = pd.concat([load_run_results(d) for d in res_dirs], ignore_index=True)
    print(f"Loaded results: {df.shape}")

    ratings = {}
    n_duels = {}
    bootstrap_draws = {}

    # -----------------------------------------------------------------
    # Clean and composite views
    # -----------------------------------------------------------------
    for view, is_clean in (("clean", True), ("global", False)):
        sub = df[df["is_clean"] == is_clean].copy()
        if sub.empty:
            label = "clean baseline" if is_clean else "perturbed"
            warnings.warn(f"No {label} results found. Skipping {view} Elo.")
            continue

        models = sorted(sub["Model"].unique())

        # -------------------------------------------------------------
        # Each metric
        # -------------------------------------------------------------
        for metric_idx, (metric_key, (metric_col, orientation)) in enumerate(
            METRICS.items()
        ):
            print(f"Computing {view} Elo for {metric_key}...")

            # Point estimate
            elo, duels = fit_elo_view(
                sub,
                metric_col=metric_col,
                orientation=orientation,
                alpha=alpha,
                test=test,
                C=C,
                anchor_model=anchor_model,
                anchor_rating=anchor_rating,
                model_names=models,
            )

            base_col = f"elo_{view}_{metric_key}"
            ratings[base_col] = elo
            n_duels[(view, metric_key)] = len(duels)

            # Bootstrap
            if n_boot > 0:
                print(f"  Bootstrap: {n_boot} replicates")
                bootstrap_seed = _bootstrap_seed(random_state, is_clean, metric_idx)

                boot_df, boot_summary = bootstrap_elo_view(
                    sub,
                    is_clean=is_clean,
                    metric_col=metric_col,
                    orientation=orientation,
                    model_names=models,
                    n_boot=n_boot,
                    ci_level=ci_level,
                    alpha=alpha,
                    test=test,
                    C=C,
                    anchor_model=anchor_model,
                    anchor_rating=anchor_rating,
                    random_state=bootstrap_seed,
                    n_jobs=n_jobs,
                    log_every=log_every,
                    log_prefix=f"[{view}/{metric_key}] ",
                )

                bootstrap_draws[(view, metric_key)] = boot_df
                ratings[f"{base_col}_bootstrap_median"] = boot_summary["bootstrap_median"]
                ratings[f"{base_col}_ci_lower"] = boot_summary["ci_lower"]
                ratings[f"{base_col}_ci_upper"] = boot_summary["ci_upper"]
                ratings[f"{base_col}_n_boot_success"] = boot_summary["n_boot_success"]

    # -----------------------------------------------------------------
    # Build final table
    # -----------------------------------------------------------------
    if not ratings:
        raise ValueError(
            "No Elo ratings could be computed from the supplied result directories."
        )

    table = pd.DataFrame(ratings)
    table.index.name = "Model"

    sort_col = (
        "elo_global_mae"
        if "elo_global_mae" in table.columns
        else table.columns[0]
    )
    table = table.sort_values(sort_col, ascending=False)

    if return_bootstrap:
        return table, n_duels, bootstrap_draws

    return table, n_duels


def run_bootstrap_block(
    res_dirs: list[str],
    out_dir: str,
    block_index: int,
    block_size: int = 2000,
    n_boot: int = 10000,
    alpha: float = 0.05,
    C: float = 1.0,
    anchor_model: str = "Linear",
    anchor_rating: float = 1000.0,
    test: str = "ttest",
    random_state: int = 42,
    n_jobs: int | None = None,
    log_every: int = 500,
) -> None:
    """Compute one block of bootstrap replicates for every view/metric and
    save each as a CSV under `out_dir`.

    With the defaults (block_size=2000, n_boot=10000) a SLURM array of 5
    tasks (`--array=0-4`, one `block_index` per task) covers the whole
    bootstrap. Blocks can run in any order, on any node, and be retried
    individually without affecting the others.
    """
    os.makedirs(out_dir, exist_ok=True)

    start = block_index * block_size
    if start >= n_boot:
        print(f"Block {block_index} starts at replicate {start} >= n_boot={n_boot}; nothing to do.")
        return
    end = min(start + block_size, n_boot)
    replicate_ids = range(start, end)

    df = pd.concat([load_run_results(d) for d in res_dirs], ignore_index=True)
    print(f"Loaded results: {df.shape}")

    for view, is_clean in (("clean", True), ("global", False)):
        sub = df[df["is_clean"] == is_clean].copy()
        if sub.empty:
            label = "clean baseline" if is_clean else "perturbed"
            warnings.warn(f"No {label} results found. Skipping {view} block.")
            continue

        models = sorted(sub["Model"].unique())

        for metric_idx, (metric_key, (metric_col, orientation)) in enumerate(
            METRICS.items()
        ):
            bootstrap_seed = _bootstrap_seed(random_state, is_clean, metric_idx)
            print(
                f"[block {block_index}] {view}/{metric_key}: "
                f"replicates {start}-{end - 1} of {n_boot}"
            )

            bootstrap_df = _run_bootstrap_replicates(
                sub,
                is_clean,
                metric_col,
                orientation,
                models,
                replicate_ids=replicate_ids,
                alpha=alpha,
                test=test,
                C=C,
                anchor_model=anchor_model,
                anchor_rating=anchor_rating,
                random_state=bootstrap_seed,
                n_jobs=n_jobs,
                n_boot_total=n_boot,
                log_every=log_every,
                log_prefix=f"[block {block_index}] [{view}/{metric_key}] ",
            )

            out_path = os.path.join(
                out_dir, f"{view}_{metric_key}_block{block_index:04d}.csv"
            )
            bootstrap_df.to_csv(out_path, index_label="replicate")
            print(f"  Saved {len(bootstrap_df)} replicates -> {out_path}")


def merge_bootstrap_blocks(
    res_dirs: list[str],
    out_dir: str,
    n_boot: int = 10000,
    alpha: float = 0.05,
    C: float = 1.0,
    anchor_model: str = "Linear",
    anchor_rating: float = 1000.0,
    test: str = "ttest",
    ci_level: float = 0.95,
) -> tuple[pd.DataFrame, dict]:
    """Combine block CSVs written by `run_bootstrap_block` into the final
    Elo table (same shape/columns as `compute_elo_table`).
    """
    df = pd.concat([load_run_results(d) for d in res_dirs], ignore_index=True)
    print(f"Loaded results: {df.shape}")

    ratings = {}
    n_duels = {}

    for view, is_clean in (("clean", True), ("global", False)):
        sub = df[df["is_clean"] == is_clean].copy()
        if sub.empty:
            label = "clean baseline" if is_clean else "perturbed"
            warnings.warn(f"No {label} results found. Skipping {view} Elo.")
            continue

        models = sorted(sub["Model"].unique())

        for metric_key, (metric_col, orientation) in METRICS.items():
            elo, duels = fit_elo_view(
                sub,
                metric_col=metric_col,
                orientation=orientation,
                alpha=alpha,
                test=test,
                C=C,
                anchor_model=anchor_model,
                anchor_rating=anchor_rating,
                model_names=models,
            )
            base_col = f"elo_{view}_{metric_key}"
            ratings[base_col] = elo
            n_duels[(view, metric_key)] = len(duels)

            pattern = os.path.join(out_dir, f"{view}_{metric_key}_block*.csv")
            block_paths = sorted(glob.glob(pattern))
            if not block_paths:
                raise FileNotFoundError(
                    f"No bootstrap block files found for {view}/{metric_key} under "
                    f"{out_dir!r} (pattern: {pattern!r}). Run run_bootstrap_block first."
                )

            block_frames = [pd.read_csv(p, index_col="replicate") for p in block_paths]
            bootstrap_df = pd.concat(block_frames).sort_index()
            bootstrap_df = bootstrap_df[~bootstrap_df.index.duplicated(keep="first")]

            n_found = len(bootstrap_df)
            if n_found < n_boot:
                warnings.warn(
                    f"{view}/{metric_key}: found {n_found}/{n_boot} bootstrap replicates "
                    f"across {len(block_paths)} block file(s); merging what is available."
                )

            boot_summary = summarize_bootstrap(bootstrap_df, ci_level=ci_level)
            ratings[f"{base_col}_bootstrap_median"] = boot_summary["bootstrap_median"]
            ratings[f"{base_col}_ci_lower"] = boot_summary["ci_lower"]
            ratings[f"{base_col}_ci_upper"] = boot_summary["ci_upper"]
            ratings[f"{base_col}_n_boot_success"] = boot_summary["n_boot_success"]

    if not ratings:
        raise ValueError(
            "No Elo ratings could be computed from the supplied result directories."
        )

    table = pd.DataFrame(ratings)
    table.index.name = "Model"

    sort_col = (
        "elo_global_mae"
        if "elo_global_mae" in table.columns
        else table.columns[0]
    )
    table = table.sort_values(sort_col, ascending=False)

    return table, n_duels


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute clean/global regression Elo ratings (MAE) with bootstrap "
            "CIs. Runs everything locally by default. Pass --block-index (or "
            "run under SLURM, which sets $SLURM_ARRAY_TASK_ID automatically) "
            "to compute just one block of the bootstrap, then --merge once "
            "every block has finished."
        )
    )
    parser.add_argument("--res-dirs", nargs="+", default=["../result/multi_reg_all"])
    parser.add_argument(
        "--out-dir", default="bootstrap_blocks_reg",
        help="Directory for per-block bootstrap CSVs (block and --merge modes).",
    )
    parser.add_argument(
        "--out-csv", default="elo_table_reg.csv",
        help="Where to write the final Elo table (local run and --merge modes).",
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--block-size", type=int, default=2000)
    parser.add_argument(
        "--block-index", type=int, default=None,
        help=(
            "Compute only replicates "
            "[block_index * block_size, (block_index + 1) * block_size) and "
            "exit. Defaults to $SLURM_ARRAY_TASK_ID when that is set."
        ),
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge block CSVs from --out-dir into the final Elo table.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--anchor-model", default="Linear")
    parser.add_argument("--anchor-rating", type=float, default=1000.0)
    parser.add_argument("--test", default="ttest", choices=["ttest", "wilcoxon"])
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--n-jobs", type=int, default=None,
        help=(
            "Worker processes for this task. Defaults to "
            "$SLURM_CPUS_PER_TASK, then all available cores."
        ),
    )
    parser.add_argument("--log-every", type=int, default=500)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    if args.merge:
        table, nduels = merge_bootstrap_blocks(
            res_dirs=args.res_dirs,
            out_dir=args.out_dir,
            n_boot=args.n_boot,
            alpha=args.alpha,
            C=args.C,
            anchor_model=args.anchor_model,
            anchor_rating=args.anchor_rating,
            test=args.test,
            ci_level=args.ci_level,
        )
        print("nduel:", nduels)
        print(table)
        table.to_csv(args.out_csv)

    else:
        block_index = args.block_index
        if block_index is None and "SLURM_ARRAY_TASK_ID" in os.environ:
            block_index = int(os.environ["SLURM_ARRAY_TASK_ID"])

        if block_index is not None:
            run_bootstrap_block(
                res_dirs=args.res_dirs,
                out_dir=args.out_dir,
                block_index=block_index,
                block_size=args.block_size,
                n_boot=args.n_boot,
                alpha=args.alpha,
                C=args.C,
                anchor_model=args.anchor_model,
                anchor_rating=args.anchor_rating,
                test=args.test,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                log_every=args.log_every,
            )
        else:
            table, nduels = compute_elo_table(
                args.res_dirs,
                alpha=args.alpha,
                C=args.C,
                anchor_model=args.anchor_model,
                anchor_rating=args.anchor_rating,
                test=args.test,
                n_boot=args.n_boot,
                ci_level=args.ci_level,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                log_every=args.log_every,
            )
            print("nduel:", nduels)
            print(table)
            table.to_csv(args.out_csv)
