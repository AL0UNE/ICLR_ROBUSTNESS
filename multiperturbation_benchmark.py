"""Composite-perturbation robustness benchmark (classification)."""

import copy
import json
import os
import shutil
import time
from datetime import datetime
from functools import partial
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import helper.m3_handling as m3_handling
from helper.bench_config import CONFIG_PATH, load_config
from helper.composite_perturbations import (
    parse_feature_permutation_variant,
    parse_input_noise_variant,
    parse_missing_data_variant,
    perturb_dic,
    sample_composite_perturbations,
    sample_level_classification,
    save_composites_report,
)
from helper.dataset_registry import DEV_KEYS
from helper.helper import (
    create_directory,
    predict_proba_batched,
    save_results,
    set_random_seed,
)
from helper.hpo_grid import PARAM_GRIDS
from helper.m3_handling import (
    cat_features,
    column_m3,
    cont_features,
)
from helper.missing_data_mechanism import configure_mask_cache
from helper.model_factory import (
    build_classification_models,
    build_param_grids,
    select_models,
)
from helper.model_helpers import (
    apply_calibrator,
    compute_calibration_metrics,
    fit_sigmoid_calibrator,
    split_calibration_holdout,
    tune_model,
)
from helper.perturbation_function import (
    add_measurement_noise,
    apply_imbalance,
    apply_label_noise,
    apply_missingness,
    apply_training_size,
    shuffle_features,
)
from helper.run_helpers import run_cv_parallel_and_save

now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")

DIR_NAME_DEFAULT = f"res_multi_{timestamp}"
OUTPUT_BASE_DIR = os.getenv("BENCHMARK_OUTPUT_BASE_DIR", "result")
DIR_NAME = os.getenv("BENCHMARK_RESULTS_DIR_NAME", DIR_NAME_DEFAULT)
directory_name = (
    os.path.join(OUTPUT_BASE_DIR, DIR_NAME) if OUTPUT_BASE_DIR else DIR_NAME
)

_DEV_KEY_OVERRIDE = os.getenv("BENCHMARK_DEV_KEY")
if _DEV_KEY_OVERRIDE and _DEV_KEY_OVERRIDE not in DEV_KEYS:
    raise ValueError(
        f"BENCHMARK_DEV_KEY={_DEV_KEY_OVERRIDE!r} must be one of {DEV_KEYS}"
    )
ACTIVE_DEV_KEYS = [_DEV_KEY_OVERRIDE] if _DEV_KEY_OVERRIDE else DEV_KEYS

LOG_FILE = os.path.join(
    directory_name, f"boosting_failures_multi_{timestamp}.log"
)
LOG_FILE_CALIB = os.path.join(
    directory_name, f"calibration_failures_multi_{timestamp}.log"
)

cfg = load_config("classification_composite")

NJOBS = cfg.n_jobs
NJOBS_TRAIN = cfg.n_jobs_train
NJOBS_GS = cfg.n_jobs_gs
HPO = cfg.hpo
N_SEARCH_GS = cfg.n_search_gs
NO_INNER_FOLD = cfg.no_inner_fold
RANDOM_STATE = cfg.random_state
N_COMPOSITES = cfg.n_composites
COMPOSITE_START = cfg.composite_start
N_PERTURB_TYPES_PER_COMPOSITE = cfg.n_perturb_types_per_composite

param_grids = (
    build_param_grids(PARAM_GRIDS, cfg.xgboost_device, cfg.lightgbm_device)
    if HPO
    else {}
)

configure_mask_cache(cfg.mask_cache_enabled, cfg.mask_cache_size)

multi_perturb_dic = {
    perturb_type: (
        [variant for variant in variants if not variant.startswith("proxy_")]
        if perturb_type == "label_noise"
        else variants
    )
    for perturb_type, variants in perturb_dic.items()
}


def _apply_label_noise(y_train, X_train, step, fold_idx, dataset_key):
    variant = step["variant"]
    level = step["level"]
    seed = set_random_seed(
        f"multi_{variant}", level, fold_idx, base_seed=RANDOM_STATE
    )
    rng = np.random.default_rng(seed)

    data = m3_handling.get_dataset(dataset_key)
    proxy_icu = (
        data.y_proxy_death_icu.loc[y_train.index]
        if data.y_proxy_death_icu is not None
        else None
    )
    proxy_overall = (
        data.y_proxy_death_overall.loc[y_train.index]
        if data.y_proxy_death_overall is not None
        else None
    )
    return apply_label_noise(
        y_train,
        X_train,
        variant,
        level,
        rng,
        proxy_icu=proxy_icu,
        proxy_overall=proxy_overall,
    )


def _apply_input_noise(X_train, X_val, step, fold_idx):
    variant = step["variant"]
    level = step["level"]

    which_set, feature_type = parse_input_noise_variant(variant)

    seed = set_random_seed(
        f"multi_input_{variant}", level, fold_idx, base_seed=RANDOM_STATE
    )
    rng = np.random.default_rng(seed)

    if which_set == "Train":
        X_train = add_measurement_noise(X_train, level, feature_type, rng=rng)
    elif which_set == "Val":
        X_val = add_measurement_noise(X_val, level, feature_type, rng=rng)
    else:
        X_train = add_measurement_noise(X_train, level, feature_type, rng=rng)
        X_val = add_measurement_noise(X_val, level, feature_type, rng=rng)

    return X_train, X_val


def _apply_missing_data(X_train, X_val, step, fold_idx):
    variant = step["variant"]
    level = step["level"]

    which_set, mechanism = parse_missing_data_variant(variant)

    seed = set_random_seed(
        f"multi_missing_{variant}", level, fold_idx, base_seed=RANDOM_STATE
    )
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator()
    torch_gen.manual_seed(seed)
    return apply_missingness(
        X_train,
        X_val,
        which_set,
        mechanism,
        level,
        rng,
        torch_gen,
        cont_features,
        cat_features,
    )


def _apply_feature_permutation(X_train, X_val, step, fold_idx):
    variant = step["variant"]
    level = step["level"]

    which_set = parse_feature_permutation_variant(variant)

    seed = set_random_seed(
        f"multi_perm_{variant}", level, fold_idx, base_seed=RANDOM_STATE
    )
    rng = np.random.default_rng(seed)

    if which_set == "Train":
        X_train, _ = shuffle_features(X_train, prop=level, rng=rng)
    elif which_set == "Val":
        X_val, _ = shuffle_features(X_val, prop=level, rng=rng)
    else:
        X_train, feat_to_shuffle = shuffle_features(
            X_train, prop=level, rng=rng
        )
        X_val, _ = shuffle_features(
            X_val, prop=level, feat_to_shuffle=feat_to_shuffle, rng=rng
        )

    return X_train, X_val


def _apply_class_imbalance(X_train, y_train, step: dict[str, Any], fold_idx):
    level = float(step["level"])
    seed = set_random_seed(
        "multi_imbalance", level, fold_idx, base_seed=RANDOM_STATE
    )
    return apply_imbalance(X_train, y_train, level, seed)


def _apply_training_regime(X_train, y_train, step, fold_idx):
    level = step["level"]
    seed = set_random_seed(
        "multi_training", level, fold_idx, base_seed=RANDOM_STATE
    )
    return apply_training_size(X_train, y_train, level, seed)


def _apply_composite_perturbations(
    X_train, X_val, y_train, fold_idx, composite_steps, dataset_key
):
    for step in composite_steps:
        perturb_type = step["type"]

        if perturb_type == "label_noise":
            y_train = _apply_label_noise(
                y_train, X_train, step, fold_idx, dataset_key
            )
        elif perturb_type == "input_noise":
            X_train, X_val = _apply_input_noise(X_train, X_val, step, fold_idx)
        elif perturb_type == "missing_data":
            X_train, X_val = _apply_missing_data(X_train, X_val, step, fold_idx)
        elif perturb_type == "feature_permutation":
            X_train, X_val = _apply_feature_permutation(
                X_train, X_val, step, fold_idx
            )
        elif perturb_type == "class_imbalance":
            X_train, y_train = _apply_class_imbalance(
                X_train, y_train, step, fold_idx
            )
        elif perturb_type == "training_data_regime":
            X_train, y_train = _apply_training_regime(
                X_train, y_train, step, fold_idx
            )
        else:
            raise ValueError(
                f"Unsupported perturbation type in composite: {perturb_type}"
            )

    return X_train, X_val, y_train


def composite_perturbation(
    model_name,
    model,
    composite_steps,
    train_idx,
    val_idx,
    fold_idx,
    dataset_key="m3",
    preset_test_name=None,
    tune_model_fn=None,
    compute_calibration_metrics_fn=None,
):
    """Runs one composite perturbation for a single model and fold.

    Args:
        model_name: Name of the model.
        model: Unfitted estimator; it is cloned before fitting.
        composite_steps: Sequence of (type, variant, level) perturbations.
        train_idx: Positional indices of the training rows.
        val_idx: Positional indices of the validation rows.
        fold_idx: Cross-validation fold number.
        dataset_key: Development set to load.
        preset_test_name: Unused; accepted for the shared runner API.
        tune_model_fn: Callback that tunes and fits the model.
        compute_calibration_metrics_fn: Callback returning calibration
            intercept and slope.

    Returns:
        The result row for this model and fold.
    """
    _ = preset_test_name  

    if tune_model_fn is None:
        raise ValueError("tune_model_fn must be provided.")
    if compute_calibration_metrics_fn is None:
        raise ValueError("compute_calibration_metrics_fn must be provided.")

    data = m3_handling.get_dataset(dataset_key)
    X, y = data.X, data.y
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

    X_train, X_val, y_train = _apply_composite_perturbations(
        X_train,
        X_val,
        y_train,
        fold_idx,
        composite_steps,
        dataset_key,
    )

    seed = set_random_seed(
        "composite",
        json.dumps(composite_steps, sort_keys=True),
        fold_idx,
        base_seed=RANDOM_STATE,
    )

    try:
        model = clone(model)
    except Exception:
        model = copy.deepcopy(model)

    start = time.time()
    tuned_model, best_params, hpo_done = tune_model_fn(
        model_name, model, X_train, y_train, random_state=seed
    )
    train_fit_time = time.time() - start

    X_val_final, y_val_final, X_cal, y_cal = split_calibration_holdout(
        X_val,
        y_val,
        calibration_size=0.1,
        random_state=seed,
    )
    y_pred_cal_raw = predict_proba_batched(tuned_model, X_cal)
    calibrator = fit_sigmoid_calibrator(y_pred_cal_raw, y_cal)

    start = time.time()
    y_pred_raw = predict_proba_batched(tuned_model, X_val_final)
    test_pred_time = time.time() - start

    y_pred = apply_calibrator(calibrator, y_pred_raw)

    intercept, slope = compute_calibration_metrics_fn(
        y_val_final, y_pred, model_name
    )

    return (
        model_name,
        json.dumps(composite_steps, sort_keys=True),
        roc_auc_score(y_score=y_pred, y_true=y_val_final),
        brier_score_loss(y_val_final, y_pred),
        intercept,
        slope,
        train_fit_time,
        test_pred_time,
        best_params,
        hpo_done,
    )


perturbation_task_kwargs = {
    "tune_model_fn": partial(
        tune_model,
        param_grids=param_grids,
        n_search_gs=N_SEARCH_GS,
        njobs_gs=NJOBS_GS,
        random_state_global=RANDOM_STATE,
        log_file=LOG_FILE,
        n_jobs_train=NJOBS_TRAIN,
        no_inner_fold=NO_INNER_FOLD,
    ),
    "compute_calibration_metrics_fn": partial(
        compute_calibration_metrics, log_file=LOG_FILE_CALIB
    ),
}


def main():
    """Runs the composite-perturbation benchmark over all development sets."""
    n_folds = cfg.n_folds

    models = build_classification_models(
        cont_features,
        cat_features,
        RANDOM_STATE,
        cfg.xgboost_device,
        cfg.lightgbm_device,
        realmlp_device=cfg.realmlp_device,
    )
    models = select_models(
        models, cfg.models
    ) 

    create_directory(directory_name)

    if os.path.isfile(CONFIG_PATH):
        shutil.copyfile(
            CONFIG_PATH, os.path.join(directory_name, "backup_config.yaml")
        )

    composites = sample_composite_perturbations(
        multi_perturb_dic,
        sample_level_classification,
        n_composites=N_COMPOSITES,
        n_types=N_PERTURB_TYPES_PER_COMPOSITE,
        random_state=RANDOM_STATE,
    )
    composites_report_path = os.path.join(
        directory_name, "composite_combinations.json"
    )
    save_composites_report(composites, composites_report_path)

    print("Sampled composite perturbations:")
    for i, comp in enumerate(composites, start=1):
        print(f"Composite {i}: {json.dumps(comp)}")
    print(f"Saved composite report to {composites_report_path}")

    if COMPOSITE_START < 1 or len(composites) < COMPOSITE_START:
        raise ValueError(
            f"composite_start={COMPOSITE_START} out of range for "
            f"{len(composites)} sampled composites (expected "
            f"1..{len(composites)})."
        )
    if COMPOSITE_START > 1:
        print(
            f"Skipping the first {COMPOSITE_START - 1} composite(s); "
            f"training composites {COMPOSITE_START}..{len(composites)}."
        )
    composites = composites[COMPOSITE_START - 1 :]

    for dev_key in ACTIVE_DEV_KEYS:
        print(
            f"\n========== DEVELOPMENT SET: {dev_key} ==========\n", flush=True
        )
        data = m3_handling.get_dataset(dev_key)
        X, y = data.X, data.y

        kf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE
        )
        splits = list(kf.split(X, y))

        composite_task = partial(
            composite_perturbation,
            dataset_key=dev_key,
            **perturbation_task_kwargs,
        )

        run_cv_parallel_and_save_cfg = partial(
            run_cv_parallel_and_save,
            models=models,
            splits=splits,
            n_jobs=NJOBS,
            n_folds=n_folds,
            save_results_fn=save_results,
            incremental_save=True,
        )

        clean_directory = os.path.join(directory_name, dev_key, "clean")
        create_directory(clean_directory)

        run_cv_parallel_and_save_cfg(
            composite_task,
            [[]],
            clean_directory,
            "CLEAN_BASELINE",
            column_m3,
        )

        directory = os.path.join(directory_name, dev_key, "multi_perturbation")
        create_directory(directory)

        run_cv_parallel_and_save_cfg(
            composite_task,
            composites,
            directory,
            "MULTI_PERTURBATION_COMPOSITE",
            column_m3,
        )

    print("MULTI-PERTURBATION BENCHMARK OVER")


if __name__ == "__main__":
    main()
