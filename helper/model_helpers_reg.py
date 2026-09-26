"""Model tuning and metric helpers for the regression benchmarks."""

import itertools
import traceback
import warnings

import lightgbm as lgb
import numpy as np
import torch
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    ParameterGrid,
    RandomizedSearchCV,
    train_test_split,
)
from xgboost import XGBRegressor

from .helper import (
    _threadpool_ctx,
    log_failure,
    set_estimator_n_jobs,
)
from .model_helpers import (
    _fit_final_model,
    build_preprocessors,  # noqa: F401  # shared preprocessing factory
    select_preset_params,
)


def predict_batched(model, X, batch_size: int = 100000, n_jobs=1):
    """Predicts in batches to avoid very large inference calls."""
    set_estimator_n_jobs(model, n_jobs)

    if len(X) <= batch_size:
        with _threadpool_ctx(n_jobs):
            predictions = model.predict(X)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.asarray(predictions).ravel()

    out = []
    for start in range(0, len(X), batch_size):
        X_batch = (
            X.iloc[start : start + batch_size]
            if hasattr(X, "iloc")
            else X[start : start + batch_size]
        )
        with _threadpool_ctx(n_jobs):
            predictions = model.predict(X_batch)
        out.append(np.asarray(predictions).ravel())
        del predictions
        del X_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(out)


def compute_regression_metrics(y_true, y_pred):
    """Computes regression metrics.

    Args:
        y_true: True targets.
        y_pred: Predicted targets.

    Returns:
        A tuple ``(mae, rmse, r2)``.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def _set_iteration_metadata(model_name, best_model, best_params):
    if model_name in ["LASSO", "Ridge"]:
        n_iter = getattr(best_model["lr"], "n_iter_", None)
        if n_iter is not None:
            best_params["model_n_iter"] = n_iter
    if model_name == "Gradient Boosting":
        best_params["model_n_iter"] = best_model["gb"].n_estimators_
    if model_name == "MLP":
        best_params["model_n_iter"] = best_model["mlp"].n_iter_


def _fit_with_preset(
    model_name,
    model,
    X_train,
    y_train,
    preset_repo,
    preset_context,
    strict,
    n_jobs_train,
    effective_seed,
):
    """Fits ``model`` with preset hyperparameters instead of searching."""

    def _fit_defaults(reason):
        msg = f"{reason} for model '{model_name}'" + (
            f" with context {preset_context}" if preset_context else ""
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg + ". Fitting model with default parameters.")
        set_estimator_n_jobs(model, n_jobs_train)
        with _threadpool_ctx(n_jobs_train):
            model.fit(X_train, y_train)
        return model, {}

    preset = select_preset_params(
        preset_repo, model_name, preset_context=preset_context
    )
    if preset is None:
        return _fit_defaults("No preset params found")

    params, _ = preset
    if not params:
        return _fit_defaults("Empty preset params")
    try:
        model.set_params(**params)
    except Exception as exc:
        return _fit_defaults(f"Could not apply preset params ({exc})")

    model, params = _fit_final_model(
        model_name,
        model,
        X_train,
        y_train,
        n_jobs_train,
        effective_seed,
        params,
        stratify=False,
    )
    if model_name not in ("CatBoost", "XGBoost", "LightGBM"):
        _set_iteration_metadata(model_name, model, params)
    print(f"Preset params for {model_name}: {params}")
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
    """Tunes hyperparameters for a regression model.

    With ``use_preset_best_params`` the search is skipped and the parameters
    are replayed from ``preset_best_params_repo`` (see
    :func:`helper.model_helpers.load_preset_best_params_repository`, loaded
    with ``score_col="RMSE"``). A missing or unusable preset raises when
    ``strict_preset_best_params`` is set, otherwise it warns and fits the
    model's defaults.

    Returns:
        A tuple ``(best_model, best_params)``; ``best_params`` is empty when
        the model was fit with its defaults.
    """
    # Boosting early-stopping holdouts derive from the fold-specific seed when
    # one is provided, so randomness varies per fold/level instead of being
    # pinned to a constant. Falls back to random_state_global only when no
    # per-fold seed is passed.
    effective_seed = (
        random_state if random_state is not None else random_state_global
    )

    if use_preset_best_params:
        return _fit_with_preset(
            model_name,
            model,
            X_train,
            y_train,
            preset_best_params_repo,
            preset_context,
            strict_preset_best_params,
            n_jobs_train,
            effective_seed,
        )

    if model_name not in param_grids:
        set_estimator_n_jobs(model, n_jobs_train)
        with _threadpool_ctx(n_jobs_train):
            model.fit(X_train, y_train)
        return model, {}

    if no_inner_fold:
        inner_train_idx, inner_val_idx = train_test_split(
            np.arange(len(X_train)),
            test_size=0.33,
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
            scoring="neg_mean_squared_error",
            cv=inner_splits,
            n_jobs=njobs_gs,
            random_state=random_state,
            verbose=0,
        )
        with _threadpool_ctx(n_jobs_train):
            search.fit(X_train, y_train)
        best_model = search.best_estimator_
        _set_iteration_metadata(model_name, best_model, search.best_params_)

        print(f"Best params for {model_name}: {search.best_params_}")
        return best_model, search.best_params_

    keys, values = zip(*param_dist.items())
    param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
    rng = np.random.default_rng(random_state)
    n_to_try = min(n_search_gs, len(param_grid))
    param_idx = rng.choice(len(param_grid), size=n_to_try, replace=False)
    best_score = np.inf
    best_params = None

    for p in param_idx:
        params = param_grid[p]
        try:
            _, score = grid_step(
                model_name,
                X_train,
                y_train,
                inner_splits,
                params,
                random_state_global=effective_seed,
                n_jobs_train=n_jobs_train,
            )
        except Exception as exc:
            error_msg = traceback.format_exc()
            if log_file:
                log_failure(params, error_msg, logging_file=log_file)
            print(
                f"Error during tuning {model_name} with params {params}: {exc}"
            )
            continue
        if score < best_score:
            best_score = score
            best_params = params

    if best_params is None:
        print(
            f"All hyperparameter combinations failed for {model_name}. Using "
            "default parameters."
        )
        set_estimator_n_jobs(model, n_jobs_train)
        with _threadpool_ctx(n_jobs_train):
            model.fit(X_train, y_train)
        return model, {}

    print(f"Best params for {model_name}: {best_params}")

    X_train_refit, X_early_stop, y_train_refit, y_early_stop = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=effective_seed,
    )

    if model_name == "LightGBM":
        final_model = LGBMRegressor(**best_params)
        set_estimator_n_jobs(final_model, n_jobs_train)
        with _threadpool_ctx(n_jobs_train):
            final_model.fit(
                X_train_refit,
                y_train_refit,
                eval_set=[(X_early_stop, y_early_stop)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False)
                ],
            )
        best_params["model_n_iter"] = final_model.best_iteration_
    elif model_name == "XGBoost":
        final_model = XGBRegressor(**best_params)
        set_estimator_n_jobs(final_model, n_jobs_train)
        with _threadpool_ctx(n_jobs_train):
            final_model.fit(
                X_train_refit,
                y_train_refit,
                eval_set=[(X_early_stop, y_early_stop)],
                verbose=False,
            )
        best_params["model_n_iter"] = final_model.best_iteration
    elif model_name == "CatBoost":
        from catboost import CatBoostRegressor

        final_model = CatBoostRegressor(**best_params)
        set_estimator_n_jobs(final_model, n_jobs_train)
        with _threadpool_ctx(n_jobs_train):
            final_model.fit(
                X_train_refit,
                y_train_refit,
                eval_set=[(X_early_stop, y_early_stop)],
            )
        best_params["model_n_iter"] = final_model.get_best_iteration()
    else:
        final_model = model
        set_estimator_n_jobs(final_model, n_jobs_train)
        with _threadpool_ctx(n_jobs_train):
            final_model.fit(X_train, y_train)

    return final_model, best_params


def grid_step(
    model_name,
    X_tr,
    y_tr,
    inner_splits,
    param_grid,
    random_state_global=42,
    n_jobs_train=None,
):
    """Scores one parameter combination with the inner cross-validation splits.

    Args:
        model_name: Name of the model.
        X_tr: Training features.
        y_tr: Training targets.
        inner_splits: Inner train/validation index splits.
        param_grid: Parameters for this step.
        random_state_global: Global random state.
        n_jobs_train: Number of jobs used for training.

    Returns:
        A tuple ``(model_step, mean_score)``.
    """
    scores = []
    model_step = None

    for train_idx, val_idx in inner_splits:
        X_tr_fold, X_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
        y_tr_fold, y_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
        X_tr_fold, X_early_stop, y_tr_fold, y_early_stop = train_test_split(
            X_tr_fold,
            y_tr_fold,
            test_size=0.1,
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
            score = np.sqrt(
                mean_squared_error(y_val, model_step.predict(X_val))
            )
            scores.append(score)
        elif model_name == "XGBoost":
            model_step = XGBRegressor()
            model_step.set_params(**param_grid)
            set_estimator_n_jobs(model_step, n_jobs_train)
            with _threadpool_ctx(n_jobs_train):
                model_step.fit(
                    X_tr_fold,
                    y_tr_fold,
                    eval_set=[(X_early_stop, y_early_stop)],
                    verbose=False,
                )
            score = np.sqrt(
                mean_squared_error(y_val, model_step.predict(X_val))
            )
            scores.append(score)
        elif model_name == "CatBoost":
            from catboost import CatBoostRegressor

            model_step = CatBoostRegressor()
            model_step.set_params(**param_grid)
            set_estimator_n_jobs(model_step, n_jobs_train)
            with _threadpool_ctx(n_jobs_train):
                model_step.fit(
                    X_tr_fold,
                    y_tr_fold,
                    eval_set=[(X_early_stop, y_early_stop)],
                )
            score = np.sqrt(
                mean_squared_error(y_val, model_step.predict(X_val))
            )
            scores.append(score)

    return model_step, np.mean(scores)
