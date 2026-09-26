"""Single-perturbation robustness benchmark (classification)."""

import os
import shutil
from datetime import datetime
from functools import partial

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import helper.m3_handling as m3_handling
from helper.bench_config import CONFIG_PATH, load_config
from helper.dataset_registry import DEV_KEYS, TEST_SET_SPECS, external_pairs
from helper.helper import (
    create_directory,
    save_results,
)
from helper.hpo_grid import PARAM_GRIDS
from helper.m3_handling import (
    cat_features,
    column_m3,
    column_m4,
    column_m4_overall,
    cont_features,
)
from helper.missing_data_mechanism import configure_mask_cache
from helper.model_factory import (
    build_classification_models,
    build_param_grids,
    select_models,
)
from helper.model_helpers import (
    calibrate_model,
    compute_calibration_metrics,
    load_preset_best_params_repository,
    tune_model,
    validate_preset_params_coverage,
)
from helper.perturbation_function import (
    evaluate_standard_bundle,
    evaluate_subgroup_bundle,
    evaluate_temporal_bundle,
    imbalance_data,
    input_noise,
    label_noise,
    missing_data,
    permutation_features,
    subgroup_analysis,
    train_evaluate,
    training_data_regime,
)
from helper.run_helpers import (
    run_cv_parallel_and_save,
    run_subgroup_parallel_and_save,
    run_temporal_parallel,
    run_temporal_parallel_and_save,
)

# Datetime
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")


DIR_NAME_DEFAULT = f"res_{timestamp}"
OUTPUT_BASE_DIR = os.getenv("BENCHMARK_OUTPUT_BASE_DIR", "result")
DIR_NAME = os.getenv("BENCHMARK_RESULTS_DIR_NAME", DIR_NAME_DEFAULT)
directory_name = (
    os.path.join(OUTPUT_BASE_DIR, DIR_NAME) if OUTPUT_BASE_DIR else DIR_NAME
)


LOG_FILE = os.path.join(directory_name, f"boosting_failures_{timestamp}.log")
LOG_FILE_CALIB = os.path.join(
    directory_name, f"calibration_failures_{timestamp}.log"
)


cfg = load_config("classification")

NJOBS = cfg.n_jobs
NJOBS_TRAIN = cfg.n_jobs_train
NJOBS_PREDICT = cfg.n_jobs_predict
NJOBS_GS = cfg.n_jobs_gs
SPLIT_TRAIN_PREDICT = cfg.split_train_predict
TRAIN_ACROSS_FOLDS = cfg.train_across_folds
FOLD_NJOBS = cfg.fold_n_jobs

HPO = cfg.hpo
N_SEARCH_GS = cfg.n_search_gs
NO_INNER_FOLD = cfg.no_inner_fold

USE_PRESET_BEST_PARAMS = cfg.preset_enabled
STRICT_PRESET_BEST_PARAMS = cfg.preset_strict
PRESET_RESULTS_PATH = os.getenv(
    "PRESET_RESULTS_PATH", os.path.join(ROOT_DIR, cfg.preset_path)
)
PRESET_TOP_K = cfg.preset_top_k

RANDOM_STATE = cfg.random_state

param_grids = (
    build_param_grids(PARAM_GRIDS, cfg.xgboost_device, cfg.lightgbm_device)
    if HPO
    else {}
)

configure_mask_cache(cfg.mask_cache_enabled, cfg.mask_cache_size)


def run_perturbation_suite(
    dev_dir, proxy_variants, run_cv_cfg, run_subgroup_cfg, tasks
):
    """Runs the full single-perturbation robustness suite for one dataset.

    ``tasks`` maps perturbation names to task partials already bound to the
    development dataset (via ``dataset_key``). ``proxy_variants`` is the list of
    proxy label-noise variants the dataset can run (all three for m3/m4, two for
    eICU which lacks ``death_overall``); the proxy sweep is skipped when it is
    empty.
    """
    create_directory(dev_dir)

    ### Label noise
    random_label_noise_levels = np.linspace(0, 1, 11)
    targeted_label_noise_levels = np.linspace(0, 1, 10, endpoint=False)
    directory = os.path.join(dev_dir, "label_noise")
    create_directory(directory)

    run_cv_cfg(
        tasks["label_noise"],
        random_label_noise_levels,
        directory,
        "RANDOM_LABEL_NOISE",
        column_m3,
        task_kwargs={"noise_type": "random"},
    )
    run_cv_cfg(
        tasks["label_noise"],
        targeted_label_noise_levels,
        directory,
        "01_LABEL_NOISE",
        column_m3,
        task_kwargs={"noise_type": "0to1"},
    )
    run_cv_cfg(
        tasks["label_noise"],
        targeted_label_noise_levels,
        directory,
        "10_LABEL_NOISE",
        column_m3,
        task_kwargs={"noise_type": "1to0"},
    )
    run_cv_cfg(
        tasks["label_noise"],
        random_label_noise_levels,
        directory,
        "AGE_LABEL_NOISE",
        column_m3,
        task_kwargs={"noise_type": "conditional"},
    )
    if proxy_variants:
        run_cv_cfg(
            tasks["label_noise"],
            proxy_variants,
            directory,
            "PROXY_LABEL_NOISE",
            column_m3,
            task_kwargs={"noise_type": "proxy"},
        )

    print("LABEL NOISE OVER")

    ### Measurement noise
    input_noise_level = np.linspace(0, 1, 11)

    directory = os.path.join(dev_dir, "input_noise")
    create_directory(directory)

    run_cv_cfg(
        tasks["input_noise"],
        input_noise_level,
        directory,
        "INPUT_NOISE_TRAIN",
        column_m3,
    )
    run_cv_cfg(
        tasks["input_noise"],
        input_noise_level,
        directory,
        "INPUT_NOISE_VAL",
        column_m3,
        task_kwargs={"which_set": "Val"},
    )
    run_cv_cfg(
        tasks["input_noise"],
        input_noise_level,
        directory,
        "INPUT_NOISE_ALL",
        column_m3,
        task_kwargs={"which_set": "Train_Val"},
    )
    run_cv_cfg(
        tasks["input_noise"],
        input_noise_level,
        directory,
        "INPUT_NOISE_TRAIN_CONT",
        column_m3,
        task_kwargs={"feature_type": "cont"},
    )
    run_cv_cfg(
        tasks["input_noise"],
        input_noise_level,
        directory,
        "INPUT_NOISE_VAL_CONT",
        column_m3,
        task_kwargs={"which_set": "Val", "feature_type": "cont"},
    )
    run_cv_cfg(
        tasks["input_noise"],
        input_noise_level,
        directory,
        "INPUT_NOISE_ALL_CONT",
        column_m3,
        task_kwargs={"which_set": "Train_Val", "feature_type": "cont"},
    )
    run_cv_cfg(
        tasks["input_noise"],
        input_noise_level,
        directory,
        "INPUT_NOISE_TRAIN_CAT",
        column_m3,
        task_kwargs={"feature_type": "cat"},
    )
    run_cv_cfg(
        tasks["input_noise"],
        input_noise_level,
        directory,
        "INPUT_NOISE_VAL_CAT",
        column_m3,
        task_kwargs={"which_set": "Val", "feature_type": "cat"},
    )
    run_cv_cfg(
        tasks["input_noise"],
        input_noise_level,
        directory,
        "INPUT_NOISE_ALL_CAT",
        column_m3,
        task_kwargs={"which_set": "Train_Val", "feature_type": "cat"},
    )

    print("MEASUREMENT NOISE OVER")

    ### Imbalanced data
    imbalance_ratio = np.linspace(1, 0, 10, endpoint=False)

    directory = os.path.join(dev_dir, "imbalance_data")
    create_directory(directory)

    run_cv_cfg(
        tasks["imbalance_data"],
        imbalance_ratio,
        directory,
        "IMBALANCED_DATA",
        column_m3,
    )

    print("IMBALANCE NOISE OVER")

    ### Training data size
    training_data_size = [0.05, 0.1, 0.25, 0.5, 0.8, 1]

    directory = os.path.join(dev_dir, "training_size")
    create_directory(directory)

    run_cv_cfg(
        tasks["training_data_regime"],
        training_data_size,
        directory,
        "TRAINING_SIZE",
        column_m3,
    )

    print("Training data size OVER")

    ### Feature shuffling
    shuffle_ratio = np.linspace(0, 1, 11, endpoint=True)

    directory = os.path.join(dev_dir, "feature_shuffle")
    create_directory(directory)

    run_cv_cfg(
        tasks["permutation_features"],
        shuffle_ratio,
        directory,
        "SHUFFLED_TRAIN_DATA",
        column_m3,
        task_kwargs={"which_set": "Train"},
    )
    run_cv_cfg(
        tasks["permutation_features"],
        shuffle_ratio,
        directory,
        "SHUFFLED_VAL_DATA",
        column_m3,
        task_kwargs={"which_set": "Val"},
    )
    run_cv_cfg(
        tasks["permutation_features"],
        shuffle_ratio,
        directory,
        "SHUFFLED_ALL_DATA",
        column_m3,
        task_kwargs={"which_set": "Train_Val"},
    )

    print("SHUFFLE NOISE OVER")
    ### Missing data
    missing_ratio = np.linspace(0.1, 1, 9, endpoint=False)

    directory = os.path.join(dev_dir, "missing_data")
    create_directory(directory)

    run_cv_cfg(
        tasks["missing_data"],
        missing_ratio,
        directory,
        "MCAR_TRAIN",
        column_m3,
        task_kwargs={"which_set": "Train", "mechanism": "MCAR"},
        verbose=1,
    )
    run_cv_cfg(
        tasks["missing_data"],
        missing_ratio,
        directory,
        "MCAR_VAL",
        column_m3,
        task_kwargs={"which_set": "Val", "mechanism": "MCAR"},
        verbose=1,
    )
    run_cv_cfg(
        tasks["missing_data"],
        missing_ratio,
        directory,
        "MCAR_ALL",
        column_m3,
        task_kwargs={"which_set": "Train_Val", "mechanism": "MCAR"},
        verbose=1,
    )
    run_cv_cfg(
        tasks["missing_data"],
        missing_ratio,
        directory,
        "MAR_TRAIN",
        column_m3,
        task_kwargs={"which_set": "Train", "mechanism": "MAR"},
        verbose=1,
    )
    run_cv_cfg(
        tasks["missing_data"],
        missing_ratio,
        directory,
        "MAR_VAL",
        column_m3,
        task_kwargs={"which_set": "Val", "mechanism": "MAR"},
        verbose=1,
    )
    run_cv_cfg(
        tasks["missing_data"],
        missing_ratio,
        directory,
        "MAR_ALL",
        column_m3,
        task_kwargs={"which_set": "Train_Val", "mechanism": "MAR"},
        verbose=1,
    )
    run_cv_cfg(
        tasks["missing_data"],
        missing_ratio,
        directory,
        "MNAR_TRAIN",
        column_m3,
        task_kwargs={"which_set": "Train"},
        verbose=1,
    )
    run_cv_cfg(
        tasks["missing_data"],
        missing_ratio,
        directory,
        "MNAR_VAL",
        column_m3,
        task_kwargs={"which_set": "Val"},
        verbose=1,
    )
    run_cv_cfg(
        tasks["missing_data"],
        missing_ratio,
        directory,
        "MNAR_ALL",
        column_m3,
        task_kwargs={"which_set": "Train_Val"},
        verbose=1,
    )

    print("MISSING DATA OVER")

    ### Subgroup analysis
    directory = os.path.join(dev_dir, "subgroups")
    create_directory(directory)

    run_subgroup_cfg(
        tasks["subgroup_analysis"],
        "gender",
        directory,
        "SUBGROUP_GENDER",
        column_m4,
    )
    run_subgroup_cfg(
        tasks["subgroup_analysis"],
        "age_group",
        directory,
        "SUBGROUP_AGEGROUP",
        column_m4,
    )
    run_subgroup_cfg(
        tasks["subgroup_analysis"],
        "ICU_unit",
        directory,
        "SUBGROUP_ICU_UNIT",
        column_m4,
    )

    print("SUBGROUP NOISE OVER")


def run_external_validation(
    dev_key,
    test_key,
    train_evaluate_task,
    X_train,
    y_train,
    df_test,
    out_dir,
    run_temporal_save_cfg,
    run_temporal_cfg,
):
    """Trains on the development set and evaluate on one external test set.

    Stratified results follow ``TEST_SET_SPECS[test_key]`` (each test set
    exposes a different set of stratification columns); the non-stratified
    overall result is written to the spec's overall filename.
    """
    create_directory(out_dir)
    spec = TEST_SET_SPECS[test_key]

    for stratify_on, file_stem in spec["stratifications"]:
        run_temporal_save_cfg(
            train_evaluate_task,
            X_train,
            y_train,
            df_test,
            out_dir,
            file_stem,
            column_m4,
            stratify_on=stratify_on,
        )

    results = run_temporal_cfg(
        train_evaluate_task,
        X_train,
        y_train,
        df_test,
        stratify_on=None,
        benchmark_test_name=f"{test_key}_OVERALL",
    )
    pd.DataFrame(results, columns=column_m4_overall).to_csv(
        os.path.join(out_dir, spec["overall_filename"]), index=False
    )

    print(f"[{dev_key} -> {test_key}] EXTERNAL VALIDATION OVER")


def main():
    """Runs the single-perturbation benchmark over all development sets."""
    create_directory(directory_name)

    # Snapshot the config used for this run at the root of the result folder so
    # the results stay reproducible even if config.yaml changes later.
    if os.path.isfile(CONFIG_PATH):
        shutil.copyfile(
            CONFIG_PATH, os.path.join(directory_name, "backup_config.yaml")
        )

    # MODELS (shared across every development set)
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
    )  # config.yaml `models` subset (None = all)

    PRESET_BEST_PARAMS_REPO = (
        load_preset_best_params_repository(
            PRESET_RESULTS_PATH,
            top_k_per_group=PRESET_TOP_K,
            strict=STRICT_PRESET_BEST_PARAMS,
        )
        if USE_PRESET_BEST_PARAMS
        else {}
    )

    if USE_PRESET_BEST_PARAMS:
        validate_preset_params_coverage(
            models,
            PRESET_BEST_PARAMS_REPO,
            strict=STRICT_PRESET_BEST_PARAMS,
        )

    tune_model_fn = partial(
        tune_model,
        param_grids=param_grids,
        n_search_gs=N_SEARCH_GS,
        njobs_gs=NJOBS_GS,
        random_state_global=RANDOM_STATE,
        n_jobs_train=NJOBS_TRAIN,
        log_file=LOG_FILE,
        use_preset_best_params=USE_PRESET_BEST_PARAMS,
        strict_preset_best_params=STRICT_PRESET_BEST_PARAMS,
        preset_best_params_repo=PRESET_BEST_PARAMS_REPO,
        no_inner_fold=NO_INNER_FOLD,
    )
    compute_calib = partial(
        compute_calibration_metrics, log_file=LOG_FILE_CALIB
    )
    standard_eval_kwargs = {
        "compute_calibration_metrics_fn": compute_calib,
        "predict_n_jobs": NJOBS_PREDICT,
    }

    perturbation_task_kwargs = {
        "tune_model_fn": tune_model_fn,
        "calibrate_model_fn": calibrate_model,
        "compute_calibration_metrics_fn": compute_calib,
        "predict_n_jobs": NJOBS_PREDICT,
        "base_seed": RANDOM_STATE,
    }
    # train_evaluate takes its data explicitly, so it is dataset-independent.
    train_evaluate_task = partial(
        train_evaluate,
        tune_model_fn=tune_model_fn,
        compute_calibration_metrics_fn=compute_calib,
        predict_n_jobs=NJOBS_PREDICT,
        base_seed=RANDOM_STATE,
    )

    n_folds = cfg.n_folds

    for dev_key in DEV_KEYS:
        print(
            f"\n========== DEVELOPMENT SET: {dev_key} ==========\n", flush=True
        )
        data = m3_handling.get_dataset(dev_key)
        X, y = data.X, data.y

        # Stratified on the (imbalanced) mortality outcome so folds share
        # prevalence.
        kf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE
        )
        splits = list(kf.split(X, y))

        dev_dir = os.path.join(directory_name, dev_key)

        # Run-config partials bound to this development set's folds.
        run_cv_cfg = partial(
            run_cv_parallel_and_save,
            models=models,
            splits=splits,
            n_jobs=NJOBS,
            n_folds=n_folds,
            save_results_fn=save_results,
            split_train_predict=SPLIT_TRAIN_PREDICT,
            predict_n_jobs=NJOBS_PREDICT,
            train_n_jobs=FOLD_NJOBS if TRAIN_ACROSS_FOLDS else None,
            train_across_folds=TRAIN_ACROSS_FOLDS,
            evaluate_bundle_fn=evaluate_standard_bundle,
            evaluate_bundle_kwargs=standard_eval_kwargs,
        )
        run_subgroup_cfg = partial(
            run_subgroup_parallel_and_save,
            models=models,
            splits=splits,
            n_jobs=NJOBS,
            n_folds=n_folds,
            save_results_fn=save_results,
            split_train_predict=SPLIT_TRAIN_PREDICT,
            predict_n_jobs=NJOBS_PREDICT,
            train_n_jobs=FOLD_NJOBS if TRAIN_ACROSS_FOLDS else None,
            train_across_folds=TRAIN_ACROSS_FOLDS,
            evaluate_bundle_fn=evaluate_subgroup_bundle,
            evaluate_bundle_kwargs=standard_eval_kwargs,
        )
        run_temporal_cfg = partial(
            run_temporal_parallel,
            models=models,
            n_jobs=NJOBS,
            split_train_predict=SPLIT_TRAIN_PREDICT,
            predict_n_jobs=NJOBS_PREDICT,
            evaluate_bundle_fn=evaluate_temporal_bundle,
            evaluate_bundle_kwargs=standard_eval_kwargs,
        )
        run_temporal_save_cfg = partial(
            run_temporal_parallel_and_save,
            models=models,
            n_jobs=NJOBS,
            n_folds=n_folds,
            save_results_fn=save_results,
            split_train_predict=SPLIT_TRAIN_PREDICT,
            predict_n_jobs=NJOBS_PREDICT,
            evaluate_bundle_fn=evaluate_temporal_bundle,
            evaluate_bundle_kwargs=standard_eval_kwargs,
        )

        tasks = {
            name: partial(fn, dataset_key=dev_key, **perturbation_task_kwargs)
            for name, fn in {
                "label_noise": label_noise,
                "input_noise": input_noise,
                "imbalance_data": imbalance_data,
                "training_data_regime": training_data_regime,
                "permutation_features": permutation_features,
                "missing_data": missing_data,
                "subgroup_analysis": subgroup_analysis,
            }.items()
        }

        # Main robustness suite (cross-validated on the development set).
        run_perturbation_suite(
            dev_dir,
            data.available_proxy_variants,
            run_cv_cfg,
            run_subgroup_cfg,
            tasks,
        )

        # External validation on the two other datasets.
        for test_key in external_pairs(dev_key):
            X_train, y_train, df_test = m3_handling.build_external_frames(
                dev_key, test_key
            )
            out_dir = os.path.join(dev_dir, test_key)
            run_external_validation(
                dev_key,
                test_key,
                train_evaluate_task,
                X_train,
                y_train,
                df_test,
                out_dir,
                run_temporal_save_cfg,
                run_temporal_cfg,
            )

    print("ALL DEVELOPMENT SETS OVER")


if __name__ == "__main__":
    main()
