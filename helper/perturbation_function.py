"""Per-task perturbation functions for the classification benchmarks."""

import copy
import time

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from . import m3_handling

from .composite_perturbations import perturb_dic  # noqa: F401
from .helper import predict_proba_batched, set_random_seed
from .m3_handling import (
    cat_features,
    cont_features,
    features,
)
from .perturbations import (
    add_measurement_noise as _add_measurement_noise,
)
from .perturbations import (
    apply_missingness,
    apply_training_size,
    shuffle_features,
)


def _require_callbacks(
    tune_model_fn=None,
    calibrate_model_fn=None,
    compute_calibration_metrics_fn=None,
):
    if tune_model_fn is None:
        raise ValueError("tune_model_fn must be provided.")
    if calibrate_model_fn is None:
        raise ValueError("calibrate_model_fn must be provided.")
    if compute_calibration_metrics_fn is None:
        raise ValueError("compute_calibration_metrics_fn must be provided.")
    return tune_model_fn, calibrate_model_fn, compute_calibration_metrics_fn


def _clone_model(model):
    try:
        return clone(model)
    except Exception:
        return copy.deepcopy(model)


def apply_label_noise(
    y_train, X_train, kind, level, rng, proxy_icu=None, proxy_overall=None
):
    """Returns a label-noised copy of ``y_train``.

    Shared by the single and composite classification benchmarks.

    ``kind`` is one of ``random``/``0to1``/``1to0``/``conditional`` or
    ``proxy_hospital_death``/``proxy_icu_death``/``proxy_overall_death``.
    ``level`` is the numeric noise level (ignored for proxy variants).
    ``proxy_icu``/``proxy_overall`` must already be aligned to ``y_train``'s
    rows.
    """
    y_noisy = y_train.copy()
    if kind == "random":
        probs = rng.random(len(y_noisy))
        idx_change_outcome = probs < level
        y_noisy.iloc[idx_change_outcome] = 1 - y_noisy.iloc[idx_change_outcome]
    elif kind == "0to1":
        mask = y_noisy == 0
        probs = rng.random(mask.sum())
        idx_change = probs < level
        idxs = np.where(mask)[0][idx_change]
        y_noisy.iloc[idxs] = 1
    elif kind == "1to0":
        mask = y_noisy == 1
        probs = rng.random(mask.sum())
        idx_change = probs < level
        idxs = np.where(mask)[0][idx_change]
        y_noisy.iloc[idxs] = 0
    elif kind == "conditional":
        age_train_perc = X_train.age.rank(pct=True)
        # In the composite benchmark a prior missing-data step can introduce
        # NaNs into ``age``; rank(pct=True) then yields NaN for those rows,
        # which would make rng.binomial raise "p contains NaNs". Treat
        # unknown-age rows as "no swap" (proba 0) and clip to guard against
        # out-of-range levels.
        swap_proba = (age_train_perc * level).fillna(0.0).clip(0.0, 1.0)
        swap_by_age = rng.binomial(1, p=swap_proba, size=len(age_train_perc))
        y_noisy = y_noisy * (1 - swap_by_age) + (1 - y_noisy) * swap_by_age
    elif kind == "proxy_icu_death":
        y_noisy = proxy_icu
    elif kind == "proxy_overall_death":
        y_noisy = proxy_overall
    elif kind == "proxy_hospital_death":
        pass 
    return y_noisy


def apply_imbalance_old(X_train, y_train, ratio, seed):
    """Downsamples the negative class to ``ratio`` of its size.

    Shared by the single and composite drivers.
    """
    negative_mask = y_train == 0
    if negative_mask.sum() == 0:
        return X_train, y_train
    X_train_negative = X_train.loc[negative_mask].sample(
        frac=ratio, random_state=seed
    )
    y_train_negative = y_train.loc[X_train_negative.index]
    X_balanced = pd.concat([X_train_negative, X_train.loc[y_train == 1]])
    y_balanced = pd.concat([y_train_negative, y_train.loc[y_train == 1]])
    return X_balanced, y_balanced


def apply_imbalance(X_train, y_train, ratio, seed):
    """Downsamples the positive class to ``ratio`` of its size.

    Shared by the single and composite drivers.
    """
    positive_mask = y_train == 1
    if positive_mask.sum() == 0:
        return X_train, y_train
    X_train_positive = X_train.loc[positive_mask].sample(
        frac=ratio, random_state=seed
    )
    y_train_positive = y_train.loc[X_train_positive.index]
    X_balanced = pd.concat([X_train_positive, X_train.loc[y_train == 0]])
    y_balanced = pd.concat([y_train_positive, y_train.loc[y_train == 0]])
    return X_balanced, y_balanced


def _run_perturbation(
    model_name,
    model,
    x_value,
    seed,
    X_train,
    y_train,
    X_val,
    y_val,
    preset_test_name,
    fold_idx,
    return_trained,
    tune_model_fn,
    calibrate_model_fn,
    compute_calibration_metrics_fn,
    predict_n_jobs,
):
    """Shared tail for the CV perturbation benchmarks.

    The flow is clone -> tune -> calibrate -> (return trained bundle | predict
    and score). The per-perturbation functions only build ``(X_train,
    y_train, X_val, y_val)`` and the ``seed``, then delegate here.
    """
    model = _clone_model(model)

    start = time.time()
    tuned_model, best_params, hpo_done = tune_model_fn(
        model_name,
        model,
        X_train,
        y_train,
        random_state=seed,
        preset_context={
            "test_name": preset_test_name,
            "noise_level": x_value,
            "fold_idx": fold_idx,
        },
    )
    train_fit_time = time.time() - start

    tuned_model, X_val, y_val = calibrate_model_fn(
        tuned_model,
        X_val,
        y_val,
        calibration_size=0.1,
        random_state=seed,
    )

    if return_trained:
        return {
            "model": tuned_model,
            "X_val": X_val,
            "y_val": y_val,
            "model_name": model_name,
            "x_value": x_value,
            "train_fit_time": train_fit_time,
            "best_params": best_params,
            "hpo_done": hpo_done,
        }

    start = time.time()
    y_pred = predict_proba_batched(tuned_model, X_val, n_jobs=predict_n_jobs)
    test_pred_time = time.time() - start

    intercept, slope = compute_calibration_metrics_fn(y_val, y_pred, model_name)

    del tuned_model

    return (
        model_name,
        x_value,
        _safe_auc(y_val, y_pred),
        brier_score_loss(y_val, y_pred),
        intercept,
        slope,
        train_fit_time,
        test_pred_time,
        best_params,
        hpo_done,
    )


def _safe_auc(y_true, y_pred):
    """Returns ROC AUC, or NaN when a stratum has a single outcome class.

    Per-subgroup/per-stratum scoring can hit strata with only one class (e.g.
    rare ICU units or age bins). There ``roc_auc_score`` either raises
    ``ValueError`` (older scikit-learn) or warns and returns NaN (newer); the
    explicit pre-check returns NaN cleanly in both cases, so a single-class
    subgroup neither aborts the stratified run nor floods warnings. Multi-class
    strata are scored normally.
    """
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    return roc_auc_score(y_score=y_pred, y_true=y_true)


def evaluate_standard_bundle(
    bundle, compute_calibration_metrics_fn=None, predict_n_jobs=1
):
    """Scores a trained bundle on the held-out validation fold.

    Args:
        bundle: Trained bundle returned by a perturbation function with
            ``return_trained=True``.
        compute_calibration_metrics_fn: Callback returning calibration intercept
            and slope.
        predict_n_jobs: Number of jobs used for prediction.

    Returns:
        The evaluation result row(s) for the bundle.
    """
    if compute_calibration_metrics_fn is None:
        raise ValueError("compute_calibration_metrics_fn must be provided.")

    model = bundle["model"]
    X_val = bundle["X_val"]
    y_val = bundle["y_val"]
    model_name = bundle["model_name"]

    start = time.time()
    y_pred = predict_proba_batched(model, X_val, n_jobs=predict_n_jobs)
    end = time.time()
    test_pred_time = end - start

    intercept, slope = compute_calibration_metrics_fn(y_val, y_pred, model_name)

    return (
        model_name,
        bundle["x_value"],
        _safe_auc(y_val, y_pred),
        brier_score_loss(y_val, y_pred),
        intercept,
        slope,
        bundle["train_fit_time"],
        test_pred_time,
        bundle["best_params"],
        bundle["hpo_done"],
    )


def evaluate_subgroup_bundle(
    bundle, compute_calibration_metrics_fn=None, predict_n_jobs=1
):
    """Scores a trained bundle on the validation fold, stratified by subgroup.

    Args:
        bundle: Trained bundle returned by a perturbation function with
            ``return_trained=True``.
        compute_calibration_metrics_fn: Callback returning calibration intercept
            and slope.
        predict_n_jobs: Number of jobs used for prediction.

    Returns:
        The evaluation result row(s) for the bundle.
    """
    if compute_calibration_metrics_fn is None:
        raise ValueError("compute_calibration_metrics_fn must be provided.")

    model = bundle["model"]
    X_val = bundle["X_val"]
    y_val = bundle["y_val"]
    stratify_on = bundle["stratify_on"]
    model_name = bundle["model_name"]

    mimic_3 = m3_handling.get_dataset(
        bundle.get("dataset_key", "m3")
    ).full_frame
    stratified_perf = []
    stratify_values = mimic_3.loc[X_val.index, stratify_on]

    for cat in mimic_3[stratify_on].dropna().unique():
        subgroup_mask = stratify_values == cat
        subgroup_index = stratify_values.index[subgroup_mask]
        X_val_strat = X_val.loc[subgroup_index]
        y_val_strat = y_val.loc[subgroup_index]

        if len(X_val_strat) == 0:
            continue

        start = time.time()
        y_pred = predict_proba_batched(
            model, X_val_strat, n_jobs=predict_n_jobs
        )
        end = time.time()
        test_pred_time = end - start

        intercept, slope = compute_calibration_metrics_fn(
            y_val_strat,
            y_pred,
            model_name,
        )

        stratified_perf.append(
            [
                model_name,
                stratify_on,
                cat,
                _safe_auc(y_val_strat, y_pred),
                brier_score_loss(y_val_strat, y_pred),
                intercept,
                slope,
                bundle["train_fit_time"],
                test_pred_time,
                bundle["best_params"],
                bundle["hpo_done"],
            ]
        )

    return stratified_perf


def evaluate_temporal_bundle(
    bundle, compute_calibration_metrics_fn=None, predict_n_jobs=1
):
    """Scores a trained bundle on the temporal test set.

    Args:
        bundle: Trained bundle returned by a perturbation function with
            ``return_trained=True``.
        compute_calibration_metrics_fn: Callback returning calibration intercept
            and slope.
        predict_n_jobs: Number of jobs used for prediction.

    Returns:
        The evaluation result row(s) for the bundle.
    """
    if compute_calibration_metrics_fn is None:
        raise ValueError("compute_calibration_metrics_fn must be provided.")

    model = bundle["model"]
    X_test = bundle["X_test"]
    y_test = bundle["y_test"]
    stratify_on = bundle["stratify_on"]
    model_name = bundle["model_name"]
    cols = bundle["train_columns"]

    if stratify_on is not None:
        stratified_perf = []
        for cat in X_test[stratify_on].dropna().unique():
            subgroup_mask = X_test[stratify_on] == cat
            subgroup_index = X_test.index[subgroup_mask]
            X_eval = X_test.loc[subgroup_index, cols]
            y_eval = y_test.loc[subgroup_index]

            if len(X_eval) == 0:
                continue

            start = time.time()
            y_pred = predict_proba_batched(model, X_eval, n_jobs=predict_n_jobs)
            end = time.time()
            test_pred_time = end - start

            intercept, slope = compute_calibration_metrics_fn(
                y_eval,
                y_pred,
                model_name,
            )

            stratified_perf.append(
                [
                    model_name,
                    stratify_on,
                    cat,
                    _safe_auc(y_eval, y_pred),
                    brier_score_loss(y_eval, y_pred),
                    intercept,
                    slope,
                    bundle["train_fit_time"],
                    test_pred_time,
                    bundle["best_params"],
                    bundle["hpo_done"],
                ]
            )

        return stratified_perf

    start = time.time()
    y_pred = predict_proba_batched(model, X_test[cols], n_jobs=predict_n_jobs)
    end = time.time()
    test_pred_time = end - start

    intercept, slope = compute_calibration_metrics_fn(
        y_test, y_pred, model_name
    )

    return (
        model_name,
        _safe_auc(y_test, y_pred),
        brier_score_loss(y_test, y_pred),
        intercept,
        slope,
        bundle["train_fit_time"],
        test_pred_time,
        bundle["hpo_done"],
    )


def label_noise(
    model_name,
    model,
    noise_level,
    train_idx,
    val_idx,
    fold_idx,
    noise_type="random",
    dataset_key="m3",
    preset_test_name=None,
    return_trained=False,
    tune_model_fn=None,
    calibrate_model_fn=None,
    compute_calibration_metrics_fn=None,
    predict_n_jobs=1,
    base_seed=42,
):
    """Runs one perturbation task for a single model, fold and level.

    Args:
        model_name: Name of the model (key of the models dict).
        model: Unfitted estimator; it is cloned before fitting.
        noise_level: Perturbation level (its meaning depends on the
            perturbation).
        train_idx: Positional indices of the training rows.
        val_idx: Positional indices of the validation rows.
        fold_idx: Cross-validation fold number (used for seeding).
        noise_type: Label-noise variant (e.g. ``random`` or a ``proxy_*``
            variant).
        dataset_key: Development set to load.
        preset_test_name: Test name used to look up preset hyperparameters.
        return_trained: If True, return the trained bundle instead of scores.
        tune_model_fn: Callback that tunes and fits the model.
        calibrate_model_fn: Callback that calibrates the fitted model.
        compute_calibration_metrics_fn: Callback returning calibration intercept
            and slope.
        predict_n_jobs: Number of jobs used for prediction.
        base_seed: Base seed for the perturbation and the model.

    Returns:
        The trained bundle if ``return_trained`` is True, otherwise the
        result row (model, level, AUC, Brier score, calibration, timings,
        best parameters and HPO flag).
    """
    tune_model_fn, calibrate_model_fn, compute_calibration_metrics_fn = (
        _require_callbacks(
            tune_model_fn,
            calibrate_model_fn,
            compute_calibration_metrics_fn,
        )
    )

    data = m3_handling.get_dataset(dataset_key)
    X, y = data.X, data.y
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    seed = set_random_seed(
        noise_type, noise_level, fold_idx, base_seed=base_seed
    )
    rng = np.random.default_rng(seed)
    kind = f"proxy_{noise_level}" if noise_type == "proxy" else noise_type
    proxy_icu = (
        data.y_proxy_death_icu.iloc[train_idx]
        if data.y_proxy_death_icu is not None
        else None
    )
    proxy_overall = (
        data.y_proxy_death_overall.iloc[train_idx]
        if data.y_proxy_death_overall is not None
        else None
    )
    y_train = apply_label_noise(
        y_train,
        X_train,
        kind,
        noise_level,
        rng,
        proxy_icu=proxy_icu,
        proxy_overall=proxy_overall,
    )

    return _run_perturbation(
        model_name,
        model,
        noise_level,
        seed,
        X_train,
        y_train,
        X_val,
        y_val,
        preset_test_name,
        fold_idx,
        return_trained,
        tune_model_fn,
        calibrate_model_fn,
        compute_calibration_metrics_fn,
        predict_n_jobs,
    )


def add_measurement_noise(
    X_frame, noise_level, feature_type="cont & cat", rng=None
):
    """Classification-bound ``perturbations.add_measurement_noise``."""
    return _add_measurement_noise(
        X_frame,
        noise_level,
        cont_features,
        cat_features,
        feature_type=feature_type,
        rng=rng,
    )


def input_noise(
    model_name,
    model,
    noise_level,
    train_idx,
    val_idx,
    fold_idx,
    which_set="Train",
    feature_type="cont & cat",
    dataset_key="m3",
    preset_test_name=None,
    return_trained=False,
    tune_model_fn=None,
    calibrate_model_fn=None,
    compute_calibration_metrics_fn=None,
    predict_n_jobs=1,
    base_seed=42,
):
    """Runs one perturbation task for a single model, fold and level.

    Args:
        model_name: Name of the model (key of the models dict).
        model: Unfitted estimator; it is cloned before fitting.
        noise_level: Perturbation level (its meaning depends on the
            perturbation).
        train_idx: Positional indices of the training rows.
        val_idx: Positional indices of the validation rows.
        fold_idx: Cross-validation fold number (used for seeding).
        which_set: Which split(s) to perturb: ``Train``, ``Val`` or
            ``Train_Val``.
        feature_type: Which features to perturb (e.g. ``cont & cat``).
        dataset_key: Development set to load.
        preset_test_name: Test name used to look up preset hyperparameters.
        return_trained: If True, return the trained bundle instead of scores.
        tune_model_fn: Callback that tunes and fits the model.
        calibrate_model_fn: Callback that calibrates the fitted model.
        compute_calibration_metrics_fn: Callback returning calibration intercept
            and slope.
        predict_n_jobs: Number of jobs used for prediction.
        base_seed: Base seed for the perturbation and the model.

    Returns:
        The trained bundle if ``return_trained`` is True, otherwise the
        result row (model, level, AUC, Brier score, calibration, timings,
        best parameters and HPO flag).
    """
    tune_model_fn, calibrate_model_fn, compute_calibration_metrics_fn = (
        _require_callbacks(
            tune_model_fn,
            calibrate_model_fn,
            compute_calibration_metrics_fn,
        )
    )

    data = m3_handling.get_dataset(dataset_key)
    X, y = data.X, data.y
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    seed = set_random_seed(
        f"{which_set}|{feature_type}",
        noise_level,
        fold_idx,
        base_seed=base_seed,
    )
    rng = np.random.default_rng(seed)

    if which_set == "Train":
        X_train = add_measurement_noise(
            X_train, noise_level, feature_type, rng=rng
        )
    if which_set == "Val":
        X_val = add_measurement_noise(X_val, noise_level, feature_type, rng=rng)
    if which_set == "Train_Val":
        X_train = add_measurement_noise(
            X_train, noise_level, feature_type, rng=rng
        )
        X_val = add_measurement_noise(X_val, noise_level, feature_type, rng=rng)

    return _run_perturbation(
        model_name,
        model,
        noise_level,
        seed,
        X_train,
        y_train,
        X_val,
        y_val,
        preset_test_name,
        fold_idx,
        return_trained,
        tune_model_fn,
        calibrate_model_fn,
        compute_calibration_metrics_fn,
        predict_n_jobs,
    )


def imbalance_data(
    model_name,
    model,
    imbalance_ratio,
    train_idx,
    val_idx,
    fold_idx,
    dataset_key="m3",
    preset_test_name=None,
    return_trained=False,
    tune_model_fn=None,
    calibrate_model_fn=None,
    compute_calibration_metrics_fn=None,
    predict_n_jobs=1,
    base_seed=42,
):
    """Runs one perturbation task for a single model, fold and level.

    Args:
        model_name: Name of the model (key of the models dict).
        model: Unfitted estimator; it is cloned before fitting.
        imbalance_ratio: Ratio the majority/minority class is downsampled to.
        train_idx: Positional indices of the training rows.
        val_idx: Positional indices of the validation rows.
        fold_idx: Cross-validation fold number (used for seeding).
        dataset_key: Development set to load.
        preset_test_name: Test name used to look up preset hyperparameters.
        return_trained: If True, return the trained bundle instead of scores.
        tune_model_fn: Callback that tunes and fits the model.
        calibrate_model_fn: Callback that calibrates the fitted model.
        compute_calibration_metrics_fn: Callback returning calibration intercept
            and slope.
        predict_n_jobs: Number of jobs used for prediction.
        base_seed: Base seed for the perturbation and the model.

    Returns:
        The trained bundle if ``return_trained`` is True, otherwise the
        result row (model, level, AUC, Brier score, calibration, timings,
        best parameters and HPO flag).
    """
    tune_model_fn, calibrate_model_fn, compute_calibration_metrics_fn = (
        _require_callbacks(
            tune_model_fn,
            calibrate_model_fn,
            compute_calibration_metrics_fn,
        )
    )

    data = m3_handling.get_dataset(dataset_key)
    X, y = data.X, data.y
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    seed = set_random_seed(
        "imbalance", imbalance_ratio, fold_idx, base_seed=base_seed
    )
    X_train, y_train = apply_imbalance(X_train, y_train, imbalance_ratio, seed)

    return _run_perturbation(
        model_name,
        model,
        imbalance_ratio,
        seed,
        X_train,
        y_train,
        X_val,
        y_val,
        preset_test_name,
        fold_idx,
        return_trained,
        tune_model_fn,
        calibrate_model_fn,
        compute_calibration_metrics_fn,
        predict_n_jobs,
    )


def training_data_regime(
    model_name,
    model,
    training_size,
    train_idx,
    val_idx,
    fold_idx,
    dataset_key="m3",
    preset_test_name=None,
    return_trained=False,
    tune_model_fn=None,
    calibrate_model_fn=None,
    compute_calibration_metrics_fn=None,
    predict_n_jobs=1,
    base_seed=42,
):
    """Runs one perturbation task for a single model, fold and level.

    Args:
        model_name: Name of the model (key of the models dict).
        model: Unfitted estimator; it is cloned before fitting.
        training_size: Fraction of the training rows to keep.
        train_idx: Positional indices of the training rows.
        val_idx: Positional indices of the validation rows.
        fold_idx: Cross-validation fold number (used for seeding).
        dataset_key: Development set to load.
        preset_test_name: Test name used to look up preset hyperparameters.
        return_trained: If True, return the trained bundle instead of scores.
        tune_model_fn: Callback that tunes and fits the model.
        calibrate_model_fn: Callback that calibrates the fitted model.
        compute_calibration_metrics_fn: Callback returning calibration intercept
            and slope.
        predict_n_jobs: Number of jobs used for prediction.
        base_seed: Base seed for the perturbation and the model.

    Returns:
        The trained bundle if ``return_trained`` is True, otherwise the
        result row (model, level, AUC, Brier score, calibration, timings,
        best parameters and HPO flag).
    """
    tune_model_fn, calibrate_model_fn, compute_calibration_metrics_fn = (
        _require_callbacks(
            tune_model_fn,
            calibrate_model_fn,
            compute_calibration_metrics_fn,
        )
    )

    data = m3_handling.get_dataset(dataset_key)
    X, y = data.X, data.y
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    seed = set_random_seed(
        "training", training_size, fold_idx, base_seed=base_seed
    )
    X_train, y_train = apply_training_size(
        X_train, y_train, training_size, seed
    )

    return _run_perturbation(
        model_name,
        model,
        training_size,
        seed,
        X_train,
        y_train,
        X_val,
        y_val,
        preset_test_name,
        fold_idx,
        return_trained,
        tune_model_fn,
        calibrate_model_fn,
        compute_calibration_metrics_fn,
        predict_n_jobs,
    )


def permutation_features(
    model_name,
    model,
    noise_level,
    train_idx,
    val_idx,
    fold_idx,
    which_set="Train",
    dataset_key="m3",
    preset_test_name=None,
    return_trained=False,
    tune_model_fn=None,
    calibrate_model_fn=None,
    compute_calibration_metrics_fn=None,
    predict_n_jobs=1,
    base_seed=42,
):
    """Runs one perturbation task for a single model, fold and level.

    Args:
        model_name: Name of the model (key of the models dict).
        model: Unfitted estimator; it is cloned before fitting.
        noise_level: Perturbation level (its meaning depends on the
            perturbation).
        train_idx: Positional indices of the training rows.
        val_idx: Positional indices of the validation rows.
        fold_idx: Cross-validation fold number (used for seeding).
        which_set: Which split(s) to perturb: ``Train``, ``Val`` or
            ``Train_Val``.
        dataset_key: Development set to load.
        preset_test_name: Test name used to look up preset hyperparameters.
        return_trained: If True, return the trained bundle instead of scores.
        tune_model_fn: Callback that tunes and fits the model.
        calibrate_model_fn: Callback that calibrates the fitted model.
        compute_calibration_metrics_fn: Callback returning calibration intercept
            and slope.
        predict_n_jobs: Number of jobs used for prediction.
        base_seed: Base seed for the perturbation and the model.

    Returns:
        The trained bundle if ``return_trained`` is True, otherwise the
        result row (model, level, AUC, Brier score, calibration, timings,
        best parameters and HPO flag).
    """
    tune_model_fn, calibrate_model_fn, compute_calibration_metrics_fn = (
        _require_callbacks(
            tune_model_fn,
            calibrate_model_fn,
            compute_calibration_metrics_fn,
        )
    )

    data = m3_handling.get_dataset(dataset_key)
    X, y = data.X, data.y
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    seed = set_random_seed(
        which_set, noise_level, fold_idx, base_seed=base_seed
    )
    rng = np.random.default_rng(seed)

    if which_set == "Train":
        X_train, _ = shuffle_features(X_train, noise_level, rng=rng)
    if which_set == "Val":
        X_val, _ = shuffle_features(X_val, noise_level, rng=rng)
    if which_set == "Train_Val":
        X_train, feat_to_shuffle = shuffle_features(
            X_train, noise_level, rng=rng
        )
        X_val, _ = shuffle_features(
            X_val, noise_level, feat_to_shuffle=feat_to_shuffle, rng=rng
        )

    return _run_perturbation(
        model_name,
        model,
        noise_level,
        seed,
        X_train,
        y_train,
        X_val,
        y_val,
        preset_test_name,
        fold_idx,
        return_trained,
        tune_model_fn,
        calibrate_model_fn,
        compute_calibration_metrics_fn,
        predict_n_jobs,
    )


def missing_data(
    model_name,
    model,
    noise_level,
    train_idx,
    val_idx,
    fold_idx,
    which_set="Train",
    mechanism="MNAR",
    dataset_key="m3",
    preset_test_name=None,
    return_trained=False,
    tune_model_fn=None,
    calibrate_model_fn=None,
    compute_calibration_metrics_fn=None,
    predict_n_jobs=1,
    base_seed=42,
):
    """Runs one perturbation task for a single model, fold and level.

    Args:
        model_name: Name of the model (key of the models dict).
        model: Unfitted estimator; it is cloned before fitting.
        noise_level: Perturbation level (its meaning depends on the
            perturbation).
        train_idx: Positional indices of the training rows.
        val_idx: Positional indices of the validation rows.
        fold_idx: Cross-validation fold number (used for seeding).
        which_set: Which split(s) to perturb: ``Train``, ``Val`` or
            ``Train_Val``.
        mechanism: Missing-data mechanism (``MCAR``, ``MAR`` or ``MNAR``).
        dataset_key: Development set to load.
        preset_test_name: Test name used to look up preset hyperparameters.
        return_trained: If True, return the trained bundle instead of scores.
        tune_model_fn: Callback that tunes and fits the model.
        calibrate_model_fn: Callback that calibrates the fitted model.
        compute_calibration_metrics_fn: Callback returning calibration intercept
            and slope.
        predict_n_jobs: Number of jobs used for prediction.
        base_seed: Base seed for the perturbation and the model.

    Returns:
        The trained bundle if ``return_trained`` is True, otherwise the
        result row (model, level, AUC, Brier score, calibration, timings,
        best parameters and HPO flag).
    """
    tune_model_fn, calibrate_model_fn, compute_calibration_metrics_fn = (
        _require_callbacks(
            tune_model_fn,
            calibrate_model_fn,
            compute_calibration_metrics_fn,
        )
    )

    data = m3_handling.get_dataset(dataset_key)
    X, y = data.X, data.y
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    seed = set_random_seed(
        f"{which_set}|{mechanism}", noise_level, fold_idx, base_seed=base_seed
    )
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator()
    torch_gen.manual_seed(seed)
    X_train, X_val = apply_missingness(
        X_train,
        X_val,
        which_set,
        mechanism,
        noise_level,
        rng,
        torch_gen,
        cont_features,
        cat_features,
    )

    return _run_perturbation(
        model_name,
        model,
        noise_level,
        seed,
        X_train,
        y_train,
        X_val,
        y_val,
        preset_test_name,
        fold_idx,
        return_trained,
        tune_model_fn,
        calibrate_model_fn,
        compute_calibration_metrics_fn,
        predict_n_jobs,
    )


def subgroup_analysis(
    model_name,
    model,
    train_idx,
    val_idx,
    stratify_on="gender",
    dataset_key="m3",
    preset_test_name=None,
    return_trained=False,
    tune_model_fn=None,
    calibrate_model_fn=None,
    compute_calibration_metrics_fn=None,
    predict_n_jobs=1,
    base_seed=42,
):
    """Runs one perturbation task for a single model, fold and level.

    Args:
        model_name: Name of the model (key of the models dict).
        model: Unfitted estimator; it is cloned before fitting.
        train_idx: Positional indices of the training rows.
        val_idx: Positional indices of the validation rows.
        stratify_on: Column used to stratify the evaluation.
        dataset_key: Development set to load.
        preset_test_name: Test name used to look up preset hyperparameters.
        return_trained: If True, return the trained bundle instead of scores.
        tune_model_fn: Callback that tunes and fits the model.
        calibrate_model_fn: Callback that calibrates the fitted model.
        compute_calibration_metrics_fn: Callback returning calibration intercept
            and slope.
        predict_n_jobs: Number of jobs used for prediction.
        base_seed: Base seed for the perturbation and the model.

    Returns:
        The trained bundle if ``return_trained`` is True, otherwise the
        result row (model, level, AUC, Brier score, calibration, timings,
        best parameters and HPO flag).
    """
    tune_model_fn, calibrate_model_fn, compute_calibration_metrics_fn = (
        _require_callbacks(
            tune_model_fn,
            calibrate_model_fn,
            compute_calibration_metrics_fn,
        )
    )

    data = m3_handling.get_dataset(dataset_key)
    mimic_3, y = data.full_frame, data.y
    X_train, X_val = mimic_3.iloc[train_idx], mimic_3.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    seed = set_random_seed("subgroup", stratify_on, 0, base_seed=base_seed)

    try:
        model = clone(model)
    except Exception:
        model = copy.deepcopy(model)

    start = time.time()
    tuned_model, best_params, hpo_done = tune_model_fn(
        model_name,
        model,
        X_train[features],
        y_train,
        random_state=seed,
        preset_context={
            "test_name": preset_test_name,
            "noise_level": None,
            "fold_idx": 0,
        },
    )
    end = time.time()
    train_fit_time = end - start

    tuned_model, X_val, y_val = calibrate_model_fn(
        tuned_model,
        X_val[features],
        y_val,
        calibration_size=0.1,
        random_state=seed,
    )

    bundle = {
        "model": tuned_model,
        "X_val": X_val,
        "y_val": y_val,
        "stratify_on": stratify_on,
        "model_name": model_name,
        "dataset_key": dataset_key,
        "train_fit_time": train_fit_time,
        "best_params": best_params,
        "hpo_done": hpo_done,
    }

    if return_trained:
        return bundle

    # Single source of truth for stratified scoring (also used by the split
    # train/predict path), so the two cannot drift.
    return evaluate_subgroup_bundle(
        bundle,
        compute_calibration_metrics_fn=compute_calibration_metrics_fn,
        predict_n_jobs=predict_n_jobs,
    )


def train_evaluate(
    model_name,
    model,
    X_train,
    y_train,
    df_test,
    stratify_on="gender",
    preset_test_name=None,
    return_trained=False,
    tune_model_fn=None,
    compute_calibration_metrics_fn=None,
    predict_n_jobs=1,
    base_seed=42,
):
    """Trains on a source dataset and evaluates on an external test set.

    Args:
        model_name: Name of the model.
        model: Unfitted estimator; it is cloned before fitting.
        X_train: Source training features.
        y_train: Source training labels.
        df_test: External test frame.
        stratify_on: Column used to stratify the evaluation.
        preset_test_name: Test name used to look up preset hyperparameters.
        return_trained: If True, return the trained bundle instead of scores.
        tune_model_fn: Callback that tunes and fits the model.
        compute_calibration_metrics_fn: Callback returning calibration intercept
            and slope.
        predict_n_jobs: Number of jobs used for prediction.
        base_seed: Base seed for the model.

    Returns:
        The trained bundle if ``return_trained`` is True, otherwise the
        stratified evaluation results.
    """
    if tune_model_fn is None:
        raise ValueError("tune_model_fn must be provided.")
    if compute_calibration_metrics_fn is None:
        raise ValueError("compute_calibration_metrics_fn must be provided.")
    assert tune_model_fn is not None
    assert compute_calibration_metrics_fn is not None

    seed = set_random_seed("m4", stratify_on, 0, base_seed=base_seed)

    try:
        model = clone(model)
    except Exception:
        model = copy.deepcopy(model)

    # Calibrate in-domain (on a held-out MIMIC-III source slice), then transport
    # the source-calibrated model to the external set. No target labels are used
    # for calibration and the full external set is scored, so the reported
    # calibration reflects how the source model's probabilities transport under
    # shift (rather than a recalibration fitted on the target). Mirrors the
    # in-domain calibration used by the cross-validation path.
    cal_seed = set_random_seed("m4_cal", stratify_on, 0, base_seed=base_seed)
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        stratify=y_train,
        random_state=cal_seed,
    )

    start = time.time()
    tuned_model, best_params, hpo_done = tune_model_fn(
        model_name,
        model,
        X_fit,
        y_fit,
        random_state=seed,
        preset_context={
            "test_name": preset_test_name,
            "noise_level": None,
            "fold_idx": 0,
        },
    )
    end = time.time()
    train_fit_time = end - start

    calibrated_clf = CalibratedClassifierCV(FrozenEstimator(tuned_model))
    calibrated_clf.fit(X_cal, y_cal)
    tuned_model = calibrated_clf

    bundle = {
        "model": tuned_model,
        "X_test": df_test,
        "y_test": df_test["hospital_mortality"],
        "stratify_on": stratify_on,
        "model_name": model_name,
        "train_columns": X_train.columns,
        "train_fit_time": train_fit_time,
        "best_params": best_params,
        "hpo_done": hpo_done,
    }

    if return_trained:
        return bundle

    # Single source of truth for temporal/external scoring (also used by the
    # split train/predict path), so the two cannot drift.
    return evaluate_temporal_bundle(
        bundle,
        compute_calibration_metrics_fn=compute_calibration_metrics_fn,
        predict_n_jobs=predict_n_jobs,
    )
