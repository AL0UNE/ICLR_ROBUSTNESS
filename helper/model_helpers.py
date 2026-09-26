"""Model tuning, calibration and preset-parameter helpers (classification)."""

import ast
import copy
import csv
import glob
import itertools
import json
import os
import re
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy import special
from sklearn.calibration import CalibratedClassifierCV, _SigmoidCalibration
from sklearn.compose import ColumnTransformer
from sklearn.experimental import (
    enable_iterative_imputer,  # noqa: F401  # Required to use IterativeImputer
)
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    KFold,
    ParameterGrid,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .helper import log_failure, model_default_params, set_estimator_n_jobs

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None


def _threadpool_ctx(n_jobs):
    if threadpool_limits is None:
        return nullcontext()
    if n_jobs is None:
        return nullcontext()
    if isinstance(n_jobs, int) and n_jobs < 1:
        return nullcontext()
    return threadpool_limits(limits=n_jobs)


def build_preprocessors(cont_features, cat_features, random_state=42):
    """Builds the continuous/categorical preprocessing column transformers.

    Continuous features are imputed with the mean and categorical
    features with a most-frequent (``SimpleImputer``); both add missingness
    indicators. Two preprocessors are returned:

    - ``preprocessor_scaled`` standardises the continuous features (for linear
      models);
    - ``preprocessor_unscaled`` leaves them on their original scale (for
      tree/boosting models).

    Returns:
        A tuple ``(preprocessor_scaled, preprocessor_unscaled)``.
    """
    continuous_transformer_scaled = Pipeline(
        steps=[
            ("imputer", SimpleImputer(add_indicator=True, strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    continuous_transformer_unscaled = Pipeline(
        steps=[
            ("imputer", SimpleImputer(add_indicator=True, strategy="mean")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(add_indicator=True, strategy="most_frequent"),
            )
        ]
    )

    preprocessor_scaled = ColumnTransformer(
        transformers=[
            ("cont", continuous_transformer_scaled, cont_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )
    preprocessor_unscaled = ColumnTransformer(
        transformers=[
            ("cont", continuous_transformer_unscaled, cont_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )
    return preprocessor_scaled, preprocessor_unscaled


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _parse_best_params_cell(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, float) and np.isnan(raw_value):
        return None
    if isinstance(raw_value, dict):
        params = dict(raw_value)
        params.pop("model_n_iter", None)
        return params

    text = str(raw_value).strip()
    if not text or text in {"nan", "None"}:
        return None
    if text == "{}":
        return {}

    text = re.sub(r"['\"]model_n_iter['\"]\s*:\s*array\([^)]*\)\s*,?", "", text)
    text = re.sub(r",\s*}\s*$", "}", text)

    # Normalize numpy reprs so the cell can be parsed with a strict literal
    # evaluator instead of eval(): e.g. np.int64(3) -> 3, float64(0.1) -> 0.1,
    # and array([...]) / np.array([...]) -> [...].
    text = re.sub(
        r"(?:np\.)?(?:u?int(?:8|16|32|64)|float(?:16|32|64)|bool_)\(([^()]*)\)",
        r"\1",
        text,
    )
    text = re.sub(r"(?:np\.)?array\((\[[^\]]*\])\)", r"\1", text)

    try:
        params = ast.literal_eval(text)
    except Exception:
        return None

    if not isinstance(params, dict):
        return None

    params.pop("model_n_iter", None)
    return params


def _parse_hpo_cell(raw_value):
    """Parses a stored HPO flag to bool, or None if it is not interpretable."""
    if isinstance(raw_value, bool):
        return raw_value
    text = str(raw_value).strip().lower()
    if text in {"true", "1", "1.0"}:
        return True
    if text in {"false", "0", "0.0"}:
        return False
    return None


def _read_preset_csv_upto_best_param(csv_path):
    """Reads preset CSV columns up to Best param/Best params.

    The HPO column is included when present.

    Result CSVs can contain unstable fold-level columns after Best param;
    limiting the read to the stable prefix avoids parser/runtime issues. The HPO
    flag is carried alongside (when the source has it) so a preset can inherit
    the HPO status of the result that produced it.
    """
    with open(csv_path, encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.reader(file_obj)
        header = next(reader, None)
        if not header:
            raise ValueError("empty csv")

        bp_col = (
            "Best param"
            if "Best param" in header
            else ("Best params" if "Best params" in header else None)
        )
        if bp_col is None:
            raise ValueError("missing 'Best param'/'Best params' column")

        bp_idx = header.index(bp_col)
        hpo_idx = header.index("HPO") if "HPO" in header else None
        kept_columns = header[: bp_idx + 1]
        if hpo_idx is not None:
            kept_columns = kept_columns + ["HPO"]
        rows = []
        for row in reader:
            if not row:
                continue
            if len(row) <= bp_idx:
                continue
            kept = row[: bp_idx + 1]
            if hpo_idx is not None:
                kept = kept + [row[hpo_idx] if hpo_idx < len(row) else ""]
            rows.append(kept)

    import pandas as pd

    return pd.DataFrame(rows, columns=kept_columns), bp_col


def _infer_test_name_from_csv(csv_path):
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    return re.sub(r"_(-?\d+)$", "", stem)


def _normalize_noise_level(value):
    if value is None:
        return "__NONE__"
    if isinstance(value, float) and np.isnan(value):
        return "__NONE__"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return format(float(value), ".12g")
    return str(value)


def _top_k_unique_params(entries, k):
    scored = sorted(
        entries, key=lambda e: e.get("score", float("-inf")), reverse=True
    )
    out = []
    seen = set()
    for item in scored:
        params = item.get("params")
        if params is None:
            continue
        canonical = json.dumps(
            _to_jsonable(params), sort_keys=True, default=str
        )
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append({"params": params, "hpo": item.get("hpo", bool(params))})
        if len(out) >= k:
            break
    return out


def load_preset_best_params_repository(
    source_path,
    top_k_per_group=5,
    strict=False,
    score_col="AUC",
    higher_is_better=True,
):
    """Loads preset hyperparameters from a CSV file or directory of result CSVs.

    Candidates within a group are ranked by ``score_col`` (best first): the
    classification default is ``AUC`` (higher is better); regression passes
    ``RMSE`` with ``higher_is_better=False``.

    Returns a repository with lookup tiers:
    1) (test_name, model_name, noise_level)
    2) (test_name, model_name)
    3) (model_name)
    """
    repo = {
        "exact": {},
        "test_model": {},
        "model": {},
    }

    if not source_path:
        if strict:
            raise ValueError("source_path must be provided when strict=True")
        warnings.warn(
            "No preset source path provided. Preset hyperparameters disabled."
        )
        return repo

    if os.path.isfile(source_path):
        csv_files = [source_path]
    elif os.path.isdir(source_path):
        csv_files = glob.glob(
            os.path.join(source_path, "**", "*.csv"), recursive=True
        )
    else:
        if strict:
            raise FileNotFoundError(
                f"Preset source path not found: {source_path}"
            )
        warnings.warn(
            f"Preset source path not found: {source_path}. Preset "
            "hyperparameters disabled."
        )
        return repo

    if not csv_files:
        if strict:
            raise ValueError(
                f"No CSV files found in preset source path: {source_path}"
            )
        warnings.warn(
            f"No CSV files found in preset source path: {source_path}"
        )
        return repo

    exact_entries = defaultdict(list)
    test_model_entries = defaultdict(list)
    model_entries = defaultdict(list)
    malformed_files = []

    for csv_path in csv_files:
        # Aggregate exports do not follow the per-model hyperparameter result
        # schema. Ignore them even in strict mode.
        if os.path.basename(csv_path).lower() in {
            "m3.csv",
            "m4.csv",
            "eicu.csv",
        }:
            continue

        try:
            df, bp_col = _read_preset_csv_upto_best_param(csv_path)
        except Exception as exc:
            if strict:
                malformed_files.append(
                    (csv_path, f"unable to read csv prefix: {exc}")
                )
            continue

        if "Model" not in df.columns:
            if strict:
                malformed_files.append((csv_path, "missing 'Model' column"))
            continue

        test_name = _infer_test_name_from_csv(csv_path)
        has_noise = "Noise level" in df.columns
        has_score = score_col in df.columns
        has_hpo = "HPO" in df.columns

        for _, row in df.iterrows():
            model_name = row.get("Model")
            if model_name is None or (
                isinstance(model_name, float) and np.isnan(model_name)
            ):
                continue

            params = _parse_best_params_cell(row.get(bp_col))
            if params is None:
                continue

            # Inherit the source row's HPO status; legacy files without the
            # column fall back to "tuned iff params are non-empty".
            hpo_parsed = _parse_hpo_cell(row.get("HPO")) if has_hpo else None
            hpo_value = hpo_parsed if hpo_parsed is not None else bool(params)

            noise_key = (
                _normalize_noise_level(row.get("Noise level"))
                if has_noise
                else "__NONE__"
            )
            score_raw = row.get(score_col) if has_score else None
            try:
                score_value = float(score_raw)
            except (TypeError, ValueError):
                score_value = float("nan")
            if np.isnan(score_value):
                score_value = float("-inf")
            elif not higher_is_better:
                score_value = -score_value

            entry = {"params": params, "score": score_value, "hpo": hpo_value}
            exact_entries[(test_name, str(model_name), noise_key)].append(entry)
            test_model_entries[(test_name, str(model_name))].append(entry)
            model_entries[str(model_name)].append(entry)

    if strict and malformed_files:
        details = "\n".join(
            [f" - {p}: {msg}" for p, msg in malformed_files[:20]]
        )
        raise ValueError(
            "Result file structure check failed for one or more CSV files:\n"
            f"{details}"
        )

    repo["exact"] = {}
    for key, entries in exact_entries.items():
        selected = _top_k_unique_params(entries, top_k_per_group)
        if selected:
            repo["exact"][key] = selected

    repo["test_model"] = {}
    for key, entries in test_model_entries.items():
        selected = _top_k_unique_params(entries, top_k_per_group)
        if selected:
            repo["test_model"][key] = selected

    repo["model"] = {}
    for key, entries in model_entries.items():
        selected = _top_k_unique_params(entries, top_k_per_group)
        if selected:
            repo["model"][key] = selected

    print(
        "Loaded preset hyperparameters: "
        f"exact={len(repo['exact'])}, test_model={len(repo['test_model'])}, "
        f"model={len(repo['model'])}"
    )
    return repo


def select_preset_params(preset_repo, model_name, preset_context=None):
    """Selects preset hyperparameters for a model from the preset repository.

    Args:
        preset_repo: Repository built from previous result files, or None.
        model_name: Name of the model.
        preset_context: Optional dict with ``test_name``, ``noise_level`` and
            ``fold_idx`` used to pick the most specific entry.

    Returns:
        The selected parameter dict, or None if nothing matches.
    """
    if not preset_repo:
        return None

    model_name = str(model_name)
    preset_context = preset_context or {}
    test_name = preset_context.get("test_name")
    noise_key = _normalize_noise_level(preset_context.get("noise_level"))
    fold_idx = int(preset_context.get("fold_idx", 0))

    candidates = None
    if test_name is not None:
        candidates = preset_repo.get("exact", {}).get(
            (str(test_name), model_name, noise_key)
        )
        if not candidates:
            candidates = preset_repo.get("test_model", {}).get(
                (str(test_name), model_name)
            )

    if not candidates:
        candidates = preset_repo.get("model", {}).get(model_name)

    if not candidates:
        return None

    idx = fold_idx % len(candidates)
    entry = candidates[idx]
    return copy.deepcopy(entry["params"]), entry["hpo"]


def validate_preset_params_coverage(models_dict, preset_repo, strict=False):
    """Checks that the preset repository covers every model.

    Args:
        models_dict: Mapping of model names to estimators.
        preset_repo: Repository built from previous result files, or None.
        strict: If True, raise instead of warning on missing models.

    Raises:
        ValueError: If ``strict`` and a model has no preset parameters.
    """
    if not models_dict:
        return

    repo_models = set((preset_repo or {}).get("model", {}).keys())
    missing_models = [m for m in models_dict if m not in repo_models]
    if missing_models:
        msg = "Missing preset best params for model(s): " + ", ".join(
            missing_models
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)


def _fit_final_model(
    model_name,
    model,
    X_train,
    y_train,
    n_jobs_train,
    effective_seed,
    params,
    stratify=True,
):
    """Fits ``model`` (already configured with its final hyperparameters).

    LightGBM/XGBoost/CatBoost were tuned with a fixed boosting-round ceiling
    plus early stopping (XGBoost's tuned config even carries a constructor-level
    ``early_stopping_rounds``, which errors at fit time without an eval set), so
    replaying their params faithfully requires the same held-out early-stopping
    split used when the params were originally found -- not a plain full-data
    fit. Every other model just fits on the full training data. Shared by the
    live-HPO refit step and the preset-replay path, which both reach this same
    "params are fixed, now produce a trained model" point.
    """
    set_estimator_n_jobs(model, n_jobs_train)

    if model_name not in ("LightGBM", "XGBoost", "CatBoost"):
        with _threadpool_ctx(n_jobs_train):
            model.fit(X_train, y_train)
        return model, params

    X_train_refit, X_early_stop, y_train_refit, y_early_stop = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        stratify=y_train if stratify else None,
        random_state=effective_seed,
    )

    if model_name == "LightGBM":
        with _threadpool_ctx(n_jobs_train):
            model.fit(
                X_train_refit,
                y_train_refit,
                eval_set=[(X_early_stop, y_early_stop)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False)
                ],
            )
        params["model_n_iter"] = model.best_iteration_
    elif model_name == "XGBoost":
        with _threadpool_ctx(n_jobs_train):
            model.fit(
                X_train_refit,
                y_train_refit,
                eval_set=[(X_early_stop, y_early_stop)],
                verbose=False,
            )
        params["model_n_iter"] = model.best_iteration
    else:  # CatBoost
        with _threadpool_ctx(n_jobs_train):
            model.fit(
                X_train_refit,
                y_train_refit,
                eval_set=[(X_early_stop, y_early_stop)],
            )
        params["model_n_iter"] = model.get_best_iteration()

    return model, params


def tune_model(
    model_name,
    model,
    X_train,
    y_train,
    param_grids,
    random_state=None,
    n_search_gs=15,
    njobs_gs=1,
    random_state_global=42,
    log_file=None,
    use_preset_best_params=False,
    strict_preset_best_params=False,
    preset_best_params_repo=None,
    preset_context=None,
    n_jobs_train=None,
    no_inner_fold=False,
):
    """Tunes hyperparameters for a given model.

    Args:
        model_name: Name of the model.
        model: Model to tune.
        X_train: Training features.
        y_train: Training labels.
        param_grids: Dictionary of parameter grids for each model.
        random_state: Random state for CV.
        n_search_gs: Number of hyperparameter combinations to search.
        njobs_gs: Number of jobs for grid search.
        random_state_global: Global random state.
        log_file: Log file path for failures.
        use_preset_best_params: If True, use preset parameters instead of
            tuning.
        strict_preset_best_params: If True, raise when a preset is missing.
        preset_best_params_repo: Repository of preset best parameters, or None.
        preset_context: Optional dict identifying the test, level and fold.
        n_jobs_train: Number of jobs used for training.
        no_inner_fold: If True, skip the inner cross-validation.

    Returns:
        A tuple ``(best_model, best_params, hpo_done)``. ``hpo_done`` is True
        when a grid search ran; for presets it inherits the HPO status of the
        source result; otherwise (no grid / fallback to defaults) it is False
        and ``best_params`` holds the model's default hyperparameters.
    """
    # Stochastic splits below (boosting early-stopping holdouts) derive from the
    # fold-specific seed when one is provided, so randomness varies per
    # fold/level instead of being pinned to a constant. Falls back to
    # random_state_global only when no per-fold seed is passed.
    effective_seed = (
        random_state if random_state is not None else random_state_global
    )

    if use_preset_best_params:
        preset = select_preset_params(
            preset_best_params_repo, model_name, preset_context=preset_context
        )
        if preset is None:
            msg = f"No preset params found for model '{model_name}'" + (
                f" with context {preset_context}" if preset_context else ""
            )
            if strict_preset_best_params:
                raise ValueError(msg)
            warnings.warn(msg + ". Fitting model with default parameters.")
            set_estimator_n_jobs(model, n_jobs_train)
            with _threadpool_ctx(n_jobs_train):
                model.fit(X_train, y_train)
            return model, model_default_params(model), False

        # The preset carries the HPO status of the result that produced it, so a
        # preset-driven run reports the same HPO flag as that source result.
        params, preset_hpo = preset
        try:
            model.set_params(**params)
        except Exception as e:
            if strict_preset_best_params:
                raise ValueError(
                    f"Could not apply preset params for {model_name}: {e}"
                ) from e
            warnings.warn(
                f"Could not apply preset params for {model_name}: {e}. "
                "Fitting model with default parameters."
            )
            set_estimator_n_jobs(model, n_jobs_train)
            with _threadpool_ctx(n_jobs_train):
                model.fit(X_train, y_train)
            return model, model_default_params(model), False

        model, params = _fit_final_model(
            model_name,
            model,
            X_train,
            y_train,
            n_jobs_train,
            effective_seed,
            params,
        )
        return model, params, preset_hpo

    if model_name not in param_grids:
        set_estimator_n_jobs(model, n_jobs_train)
        with _threadpool_ctx(n_jobs_train):
            model.fit(X_train, y_train)
        return model, model_default_params(model), False

    if no_inner_fold:
        inner_train_idx, inner_val_idx = train_test_split(
            np.arange(len(X_train)),
            test_size=0.33,
            stratify=y_train,
            random_state=random_state,
        )
        inner_splits = [(inner_train_idx, inner_val_idx)]
    else:
        inner_splits = list(
            KFold(n_splits=3, shuffle=True, random_state=random_state).split(
                X_train, y_train
            )
        )

    param_dist = param_grids[model_name]
    if model_name not in ["CatBoost", "XGBoost", "LightGBM"]:
        # Prevent sklearn warning when requested n_iter exceeds the finite grid
        # size.
        n_iter = n_search_gs
        try:
            n_iter = min(n_search_gs, len(ParameterGrid(param_dist)))
        except Exception:
            pass

        set_estimator_n_jobs(model, n_jobs_train)
        search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=inner_splits,
            n_jobs=njobs_gs,
            random_state=random_state,
            verbose=0,
        )
        with _threadpool_ctx(n_jobs_train):
            search.fit(X_train, y_train)
        best_model = search.best_estimator_
        if model_name in ["Logistic", "LASSO", "Ridge"]:
            search.best_params_["model_n_iter"] = best_model["lr"].n_iter_
        if model_name == "Gradient Boosting":
            search.best_params_["model_n_iter"] = best_model["gb"].n_estimators_
        if model_name == "MLP":
            search.best_params_["model_n_iter"] = best_model["mlp"].n_iter_

        print(f"Best params for {model_name}: {search.best_params_}")

        return best_model, search.best_params_, True
    keys, values = zip(*param_dist.items())
    param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
    rng = np.random.default_rng(random_state)
    n_to_try = min(n_search_gs, len(param_grid))
    param_idx = rng.choice(len(param_grid), size=n_to_try, replace=False)
    best_auc = -np.inf
    best_model = None
    best_params = None
    for p in param_idx:
        params = param_grid[p]
        try:
            model_step, score = grid_step(
                model_name,
                X_train,
                y_train,
                inner_splits,
                params,
                random_state_global=effective_seed,
                n_jobs_train=n_jobs_train,
            )
        except Exception as e:
            error_msg = traceback.format_exc()
            if log_file:
                log_failure(params, error_msg, logging_file=log_file)
            print(f"Error during tuning {model_name} with params {params}: {e}")
            continue
        if score > best_auc:
            best_auc = score
            best_params = params
            best_model = model_step
    if best_params is None:
        print(
            f"All hyperparameter combinations failed for {model_name}. Using "
            "default parameters."
        )
        set_estimator_n_jobs(model, n_jobs_train)
        with _threadpool_ctx(n_jobs_train):
            model.fit(X_train, y_train)
        return model, model_default_params(model), False
    print(f"Best params for {model_name}: {best_params}")

    if model_name == "LightGBM":
        final_model = LGBMClassifier(**best_params)
    elif model_name == "XGBoost":
        final_model = XGBClassifier(**best_params)
    elif model_name == "CatBoost":
        final_model = CatBoostClassifier(**best_params)
    else:
        final_model = best_model if best_model is not None else model

    final_model, best_params = _fit_final_model(
        model_name,
        final_model,
        X_train,
        y_train,
        n_jobs_train,
        effective_seed,
        best_params,
    )
    return final_model, best_params, True


def grid_step(
    model_name,
    X_tr,
    y_tr,
    inner_splits,
    param_grid,
    random_state_global=42,
    n_jobs_train=None,
):
    """Performs grid search step for a single parameter combination.

    Args:
        model_name: Name of the model.
        X_tr: Training features.
        y_tr: Training labels.
        inner_splits: Inner train/validation splits.
        param_grid: Parameter grid for this step.
        random_state_global: Global random state.
        n_jobs_train: Number of jobs used for training.

    Returns:
        A tuple (model_step, mean_auc).
    """
    aucs = []
    for train_idx, val_idx in inner_splits:
        X_tr_fold, X_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
        y_tr_fold, y_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
        X_tr_fold, X_early_stop, y_tr_fold, y_early_stop = train_test_split(
            X_tr_fold,
            y_tr_fold,
            test_size=0.1,
            stratify=y_tr_fold,
            random_state=random_state_global,
        )

        if model_name == "LightGBM":
            dtrain = lgb.Dataset(X_tr_fold, y_tr_fold)
            dearly_stop = lgb.Dataset(X_early_stop, y_early_stop)
            with _threadpool_ctx(n_jobs_train):
                model_step = lgb.train(
                    params=param_grid,
                    train_set=dtrain,
                    valid_sets=[dearly_stop],
                    callbacks=[
                        lgb.early_stopping(
                            stopping_rounds=50,
                            first_metric_only=True,
                            verbose=False,
                        )
                    ],
                )
            score = roc_auc_score(
                y_val,
                model_step.predict(
                    X_val, num_iteration=model_step.best_iteration
                ),
            )
            aucs.append(score)
        elif model_name == "XGBoost":
            model_step = XGBClassifier()
            model_step.set_params(**param_grid)
            set_estimator_n_jobs(model_step, n_jobs_train)
            with _threadpool_ctx(n_jobs_train):
                model_step.fit(
                    X_tr_fold,
                    y_tr_fold,
                    eval_set=[(X_early_stop, y_early_stop)],
                    verbose=False,
                )
            score = roc_auc_score(y_val, model_step.predict_proba(X_val)[:, 1])
            aucs.append(score)
        elif model_name == "CatBoost":
            model_step = CatBoostClassifier()
            model_step.set_params(**param_grid)
            set_estimator_n_jobs(model_step, n_jobs_train)
            with _threadpool_ctx(n_jobs_train):
                model_step.fit(
                    X_tr_fold,
                    y_tr_fold,
                    eval_set=[(X_early_stop, y_early_stop)],
                )
            score = roc_auc_score(y_val, model_step.predict_proba(X_val)[:, 1])
            aucs.append(score)

    return model_step, np.mean(aucs)


def calibrate_model(
    model, X_val, y_val, calibration_size=0.1, random_state=None
):
    """Calibrates a model using a holdout calibration set.

    Args:
        model: Model to calibrate.
        X_val: Validation features.
        y_val: Validation labels.
        calibration_size: Fraction of data to use for calibration.
        random_state: Random state.

    Returns:
        A tuple (calibrated_model, X_val_updated, y_val_updated).
    """
    X_val, X_cal, y_val, y_cal = train_test_split(
        X_val,
        y_val,
        test_size=calibration_size,
        stratify=y_val,
        random_state=random_state,
    )
    calibrated_clf = CalibratedClassifierCV(FrozenEstimator(model))
    calibrated_clf.fit(X_cal, y_cal)
    return calibrated_clf, X_val, y_val


def split_calibration_holdout(
    X_val, y_val, calibration_size=0.1, random_state=None
):
    """Splits off a small calibration holdout from the validation set.

    Returns ``(X_val_final, y_val_final, X_cal, y_cal)``: the "final" portion is
    what gets timed/scored as the official evaluation set; ``X_cal``/``y_cal``
    is the small holdout used only to fit a calibration curve.
    """
    X_val_final, X_cal, y_val_final, y_cal = train_test_split(
        X_val,
        y_val,
        test_size=calibration_size,
        stratify=y_val,
        random_state=random_state,
    )
    return X_val_final, y_val_final, X_cal, y_cal


def fit_sigmoid_calibrator(y_pred_cal, y_cal):
    """Fits a Platt/sigmoid calibration mapping on already-computed raw scores.

    Statistically equivalent to ``CalibratedClassifierCV(FrozenEstimator(model),
    method="sigmoid").fit(X_cal, y_cal)`` -- both fit sklearn's
    ``_SigmoidCalibration`` on the model's raw scores for the calibration
    holdout. The difference is call count: with a frozen (already-fitted)
    estimator, ``CalibratedClassifierCV``'s default ``cv=None`` still routes
    through an internal 5-fold ``cross_val_predict``, so ``model.predict_proba``
    gets invoked 5 times on the (small) calibration holdout, plus once more for
    the official evaluation predict -- 6 calls per fold total. For models whose
    inference reprocesses the full training context on every call regardless of
    query size (TabPFN/TabICL's in-context prediction), those 5 extra calls are
    pure overhead. This fits the identical calibration curve directly from one
    already-computed set of scores -- 1 call for the holdout, 1 for the timed
    official predict.
    """
    return _SigmoidCalibration().fit(y_pred_cal, y_cal)


def apply_calibrator(calibrator, y_pred_raw):
    """Maps raw scores through a calibrator from ``fit_sigmoid_calibrator``."""
    return calibrator.predict(y_pred_raw)


def compute_calibration_metrics(y_true, y_pred, model_name=None, log_file=None):
    """Computes calibration intercept and slope.

    Args:
        y_true: True labels.
        y_pred: Predicted probabilities.
        model_name: Name of the model (for logging).
        log_file: Log file path for failures.

    Returns:
        A tuple (intercept, slope).
    """
    intercept, slope = None, None

    if np.unique(np.asarray(y_true)).size < 2:
        return intercept, slope

    try:
        logits = special.logit(np.clip(y_pred, 1e-10, 1 - 1e-10))
        calib_model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=500)
        calib_model.fit(logits.reshape(-1, 1), y_true)
        intercept = calib_model.intercept_[0]
        slope = calib_model.coef_[0][0]
    except Exception:
        error_msg = traceback.format_exc()
        if model_name and log_file:
            log_failure(
                {"model": model_name, "step": "calibration"},
                error_msg,
                log_file,
            )

    return intercept, slope
