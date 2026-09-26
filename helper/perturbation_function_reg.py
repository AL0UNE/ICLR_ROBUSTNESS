"""Per-task perturbation functions for the regression benchmarks."""

import copy
import gc
import time

import numpy as np
import torch
from sklearn.base import clone

from . import m3_handling_reg
from .helper import resolve_hpo, set_random_seed
from .m3_handling_reg import cat_features, cont_features, features
from .model_helpers_reg import compute_regression_metrics, predict_batched
from .perturbations import (
    add_measurement_noise as _add_measurement_noise,
)
from .perturbations import (
    apply_missingness,
    apply_training_size,
    shuffle_features,
)


def _require_tune_model(tune_model_fn=None):
    if tune_model_fn is None:
        raise ValueError("tune_model_fn must be provided.")
    return tune_model_fn


def _clone_model(model):
    try:
        return clone(model)
    except Exception:
        return copy.deepcopy(model)


def _fit_model(
    model_name,
    model,
    X_train,
    y_train,
    seed,
    preset_test_name,
    noise_level,
    fold_idx,
    tune_model_fn,
):
    model = _clone_model(model)
    start = time.time()
    tuned_model, best_params = tune_model_fn(
        model_name,
        model,
        X_train,
        y_train,
        random_state=seed,
        preset_context={
            "test_name": preset_test_name,
            "noise_level": noise_level,
            "fold_idx": fold_idx,
        },
    )
    end = time.time()
    # Regression has no calibration wrapper, so the base model is reported
    # directly.
    hpo_done, best_params = resolve_hpo(tuned_model, best_params)
    return tuned_model, best_params, hpo_done, end - start


def apply_label_noise_reg(y_train, X_train, kind, level, rng):
    """Returns a label-noised copy of the continuous ``y_train``.

    Shared by the single and composite drivers.

    Adds squared-Gaussian noise scaled by ``level * 0.5 * std(y)``; the
    ``conditional`` variant additionally scales the noise by the age percentile
    rank.
    """
    y_noisy = y_train.copy()
    scale = np.std(y_train) if np.std(y_train) > 0 else 1.0
    base_amp = level * 0.5 * scale
    z = rng.normal(0.0, 1.0, size=len(y_train))
    if kind == "random":
        y_noisy = y_noisy + (z**2) * base_amp
    elif kind == "conditional":
        age_train_perc = X_train.age.rank(pct=True)
        # A prior missing-data step in a composite can leave NaNs in ``age``;
        # the NaN would propagate into y_noisy and crash model.fit. Mirror the
        # classification benchmark (perturbation_function.py): unknown-age rows
        # get no extra noise.
        age_train_perc = age_train_perc.fillna(0.0)
        y_noisy = y_noisy + (z**2) * (base_amp * age_train_perc)
    return y_noisy


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
    predict_n_jobs,
):
    """Shared tail for the regression CV perturbation benchmarks.

    The flow is fit/tune -> (return trained bundle | predict and score).
    """
    tuned_model, best_params, hpo_done, train_fit_time = _fit_model(
        model_name,
        model,
        X_train,
        y_train,
        seed,
        preset_test_name,
        x_value,
        fold_idx,
        tune_model_fn,
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
    y_pred = predict_batched(tuned_model, X_val, n_jobs=predict_n_jobs)
    test_pred_time = time.time() - start

    mae, rmse, r2 = compute_regression_metrics(y_val, y_pred)

    del tuned_model
    gc.collect()

    return (
        model_name,
        x_value,
        mae,
        rmse,
        r2,
        train_fit_time,
        test_pred_time,
        best_params,
        hpo_done,
    )


def evaluate_standard_bundle(bundle, predict_n_jobs=1):
    """Scores a trained bundle on the held-out validation fold.

    Args:
        bundle: Trained bundle returned by a perturbation function with
            ``return_trained=True``.
        predict_n_jobs: Number of jobs used for prediction.

    Returns:
        The evaluation result row(s) for the bundle.
    """
    model = bundle["model"]
    X_val = bundle["X_val"]
    y_val = bundle["y_val"]

    start = time.time()
    y_pred = predict_batched(model, X_val, n_jobs=predict_n_jobs)
    end = time.time()
    test_pred_time = end - start

    mae, rmse, r2 = compute_regression_metrics(y_val, y_pred)

    return (
        bundle["model_name"],
        bundle["x_value"],
        mae,
        rmse,
        r2,
        bundle["train_fit_time"],
        test_pred_time,
        bundle["best_params"],
        bundle["hpo_done"],
    )


def evaluate_subgroup_bundle(bundle, predict_n_jobs=1):
    """Scores a trained bundle on the validation fold, stratified by subgroup.

    Args:
        bundle: Trained bundle returned by a perturbation function with
            ``return_trained=True``.
        predict_n_jobs: Number of jobs used for prediction.

    Returns:
        The evaluation result row(s) for the bundle.
    """
    model = bundle["model"]
    X_val = bundle["X_val"]
    y_val = bundle["y_val"]
    stratify_on = bundle["stratify_on"]
    model_name = bundle["model_name"]

    mimic_3 = m3_handling_reg.get_dataset(
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
        y_pred = predict_batched(model, X_val_strat, n_jobs=predict_n_jobs)
        end = time.time()
        test_pred_time = end - start

        mae, rmse, r2 = compute_regression_metrics(y_val_strat, y_pred)

        stratified_perf.append(
            [
                model_name,
                stratify_on,
                cat,
                mae,
                rmse,
                r2,
                bundle["train_fit_time"],
                test_pred_time,
                bundle["best_params"],
                bundle["hpo_done"],
            ]
        )

    return stratified_perf


def evaluate_temporal_bundle(bundle, predict_n_jobs=1, batch_size=None):
    """Scores a trained bundle on the temporal test set.

    Args:
        bundle: Trained bundle returned by a perturbation function with
            ``return_trained=True``.
        predict_n_jobs: Number of jobs used for prediction.
        batch_size: Optional prediction batch size.

    Returns:
        The evaluation result row(s) for the bundle.
    """
    model = bundle["model"]
    df_test = bundle["df_test"]
    stratify_on = bundle["stratify_on"]
    model_name = bundle["model_name"]
    cols = bundle["train_columns"]
    if batch_size is None:
        batch_size = 100000

    if stratify_on is not None:
        stratified_perf = []
        for cat in df_test[stratify_on].dropna().unique():
            subgroup_mask = df_test[stratify_on] == cat
            subgroup_index = df_test.index[subgroup_mask]
            X_eval = df_test.loc[subgroup_index, cols]
            y_eval = df_test.loc[subgroup_index, "icu_los"]

            if len(X_eval) == 0:
                continue

            start = time.time()
            y_pred = predict_batched(
                model, X_eval, batch_size=batch_size, n_jobs=predict_n_jobs
            )
            end = time.time()
            test_pred_time = end - start

            mae, rmse, r2 = compute_regression_metrics(y_eval, y_pred)

            stratified_perf.append(
                [
                    model_name,
                    stratify_on,
                    cat,
                    mae,
                    rmse,
                    r2,
                    bundle["train_fit_time"],
                    test_pred_time,
                    bundle["best_params"],
                    bundle["hpo_done"],
                ]
            )

        return stratified_perf

    start = time.time()
    y_pred = predict_batched(
        model, df_test[cols], batch_size=batch_size, n_jobs=predict_n_jobs
    )
    end = time.time()
    test_pred_time = end - start

    mae, rmse, r2 = compute_regression_metrics(df_test["icu_los"], y_pred)

    return (
        model_name,
        mae,
        rmse,
        r2,
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
        predict_n_jobs: Number of jobs used for prediction.
        base_seed: Base seed for the perturbation and the model.

    Returns:
        The trained bundle if ``return_trained`` is True, otherwise the
        result row (model, level, regression metrics, timings, best
        parameters and HPO flag).
    """
    tune_model_fn = _require_tune_model(tune_model_fn)

    data = m3_handling_reg.get_dataset(dataset_key)
    X, y = data.X, data.y
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

    seed = set_random_seed(
        noise_type, noise_level, fold_idx, base_seed=base_seed
    )
    rng = np.random.default_rng(seed)
    y_train = apply_label_noise_reg(
        y_train, X_train, noise_type, noise_level, rng
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
        predict_n_jobs,
    )


def add_measurement_noise(
    X_frame, noise_level, feature_type="cont & cat", rng=None
):
    """Regression-bound wrapper for ``perturbations.add_measurement_noise``."""
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
        predict_n_jobs: Number of jobs used for prediction.
        base_seed: Base seed for the perturbation and the model.

    Returns:
        The trained bundle if ``return_trained`` is True, otherwise the
        result row (model, level, regression metrics, timings, best
        parameters and HPO flag).
    """
    tune_model_fn = _require_tune_model(tune_model_fn)

    data = m3_handling_reg.get_dataset(dataset_key)
    X, y = data.X, data.y
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

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
        predict_n_jobs: Number of jobs used for prediction.
        base_seed: Base seed for the perturbation and the model.

    Returns:
        The trained bundle if ``return_trained`` is True, otherwise the
        result row (model, level, regression metrics, timings, best
        parameters and HPO flag).
    """
    tune_model_fn = _require_tune_model(tune_model_fn)

    data = m3_handling_reg.get_dataset(dataset_key)
    X, y = data.X, data.y
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

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
        predict_n_jobs: Number of jobs used for prediction.
        base_seed: Base seed for the perturbation and the model.

    Returns:
        The trained bundle if ``return_trained`` is True, otherwise the
        result row (model, level, regression metrics, timings, best
        parameters and HPO flag).
    """
    tune_model_fn = _require_tune_model(tune_model_fn)

    data = m3_handling_reg.get_dataset(dataset_key)
    X, y = data.X, data.y
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

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
        predict_n_jobs: Number of jobs used for prediction.
        base_seed: Base seed for the perturbation and the model.

    Returns:
        The trained bundle if ``return_trained`` is True, otherwise the
        result row (model, level, regression metrics, timings, best
        parameters and HPO flag).
    """
    tune_model_fn = _require_tune_model(tune_model_fn)

    data = m3_handling_reg.get_dataset(dataset_key)
    X, y = data.X, data.y
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

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
        predict_n_jobs: Number of jobs used for prediction.
        base_seed: Base seed for the perturbation and the model.

    Returns:
        The trained bundle if ``return_trained`` is True, otherwise the
        result row (model, level, regression metrics, timings, best
        parameters and HPO flag).
    """
    tune_model_fn = _require_tune_model(tune_model_fn)

    data = m3_handling_reg.get_dataset(dataset_key)
    mimic_3, y = data.full_frame, data.y
    X_train, X_val = (
        mimic_3.iloc[train_idx].copy(),
        mimic_3.iloc[val_idx].copy(),
    )
    y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

    seed = set_random_seed("subgroup", stratify_on, 0, base_seed=base_seed)

    tuned_model, best_params, hpo_done, train_fit_time = _fit_model(
        model_name,
        model,
        X_train[features],
        y_train,
        seed,
        preset_test_name,
        None,
        0,
        tune_model_fn,
    )

    bundle = {
        "model": tuned_model,
        "X_val": X_val[features],
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
    return evaluate_subgroup_bundle(bundle, predict_n_jobs=predict_n_jobs)


def train_evaluate(
    model_name,
    model,
    X_train,
    y_train,
    df_test,
    stratify_on="gender",
    batch_size=None,
    preset_test_name=None,
    return_trained=False,
    tune_model_fn=None,
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
        batch_size: Optional prediction batch size.
        preset_test_name: Test name used to look up preset hyperparameters.
        return_trained: If True, return the trained bundle instead of scores.
        tune_model_fn: Callback that tunes and fits the model.
        predict_n_jobs: Number of jobs used for prediction.
        base_seed: Base seed for the model.

    Returns:
        The trained bundle if ``return_trained`` is True, otherwise the
        stratified evaluation results.
    """
    tune_model_fn = _require_tune_model(tune_model_fn)
    seed = set_random_seed("m4", stratify_on, 0, base_seed=base_seed)
    if batch_size is None:
        batch_size = 100000

    tuned_model, best_params, hpo_done, train_fit_time = _fit_model(
        model_name,
        model,
        X_train,
        y_train,
        seed,
        preset_test_name,
        None,
        0,
        tune_model_fn,
    )

    bundle = {
        "model": tuned_model,
        "df_test": df_test,
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
        bundle, predict_n_jobs=predict_n_jobs, batch_size=batch_size
    )
