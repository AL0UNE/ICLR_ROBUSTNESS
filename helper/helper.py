"""Small shared utilities: result saving, HPO reporting, batched prediction."""

import hashlib
import json
import numbers
import os
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import special

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None


def stable_hash(s: str) -> int:
    """Returns a stable hash of the given string."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def _is_simple_param(value):
    """Returns True for values that document cleanly in a results cell.

    Scalars/None and flat sequences of them qualify, so estimator/step objects
    from ``get_params`` are skipped.
    """
    if value is None or isinstance(value, (str, bytes, numbers.Number)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_simple_param(v) for v in value)
    return False


def model_default_params(model):
    """The model's configured hyperparameters as a flat dict of simple values.

    Used to document a model that was fit *without* HPO: the row then records
    the actual (default) parameters the model ran with instead of an empty dict.
    The nested estimator/step objects returned by ``get_params(deep=True)`` are
    dropped, leaving the scalar hyperparameters (e.g. ``lr__C``,
    ``n_estimators``).
    """
    try:
        params = model.get_params(deep=True)
    except Exception:
        return {}
    return {k: v for k, v in params.items() if _is_simple_param(v)}


def resolve_hpo(model, best_params):
    """Maps a tuner result to ``(hpo_done, params_for_report)``.

    The pair is used for a results row. HPO ran iff the tuner returned a
    non-empty best-params dict; otherwise the model was fit with its defaults,
    so report those (via
    :func:`model_default_params`) rather than an empty dict. Call this on the
    *base* model, before any calibration wrapper replaces it.
    """
    if best_params:
        return True, best_params
    return False, model_default_params(model)


def set_random_seed(test_name, noise_level, fold_idx, base_seed=42):
    """Returns a deterministic integer seed for the given test/noise/fold.

    NOTE: this function does not mutate global RNG state. Callers should use
    the returned seed to construct local numpy and torch generators.
    """
    combined = f"{test_name}_{noise_level}_{fold_idx}_{base_seed}"
    seed = abs(stable_hash(combined)) % (2**32)
    return int(seed)


def create_directory(directory_name):
    """Creates ``directory_name`` (and parents) if it does not exist."""
    os.makedirs(directory_name, exist_ok=True)


def _threadpool_ctx(n_jobs):
    if threadpool_limits is None:
        return nullcontext()
    if n_jobs is None:
        return nullcontext()
    if isinstance(n_jobs, int) and n_jobs < 1:
        return nullcontext()
    return threadpool_limits(limits=n_jobs)


def _n_jobs_owner(estimator, params, key):
    """Returns the sub-estimator that owns the given ``*n_jobs`` param key."""
    if key == "n_jobs":
        return estimator
    parent = key[: -len("__n_jobs")] if key.endswith("__n_jobs") else None
    return params.get(parent) if parent is not None else None


def _n_jobs_is_deprecated(owner):
    """True if ``owner`` deprecated ``n_jobs`` as a no-op (scikit-learn >= 1.8).

    Setting it would emit a FutureWarning at fit time, so we skip such params.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return False
    return isinstance(owner, LogisticRegression)


def set_estimator_n_jobs(estimator, n_jobs):
    """Best effort setter for nested estimator thread/n_jobs parameters.

    Plain ``n_jobs`` is one knob among several: pytabkit's RealMLP exposes
    ``n_threads`` instead (defaults to all physical cores via
    ``psutil.cpu_count(logical=False)``), and CatBoost's ``thread_count`` never
    shows up in ``get_params()`` unless it was passed at construction time, so a
    suffix-match over ``get_params()`` alone would miss both and leave
    every fold-level joblib worker oversubscribing every physical core.
    """
    if n_jobs is None:
        return
    if isinstance(n_jobs, int) and n_jobs < 1:
        return
    if (
        estimator is None
        or not hasattr(estimator, "get_params")
        or not hasattr(estimator, "set_params")
    ):
        return

    try:
        params = estimator.get_params(deep=True)
    except Exception:
        return

    updates = {
        k: n_jobs
        for k in params
        if k.endswith("n_jobs")
        and not _n_jobs_is_deprecated(_n_jobs_owner(estimator, params, k))
    }

    # pytabkit (RealMLP) exposes n_threads instead of n_jobs;
    updates.update({k: n_jobs for k in params if k.endswith("n_threads")})

    if updates:
        try:
            estimator.set_params(**updates)
        except Exception:
            pass

    # CatBoost has no n_jobs param, and get_params() only reports
    # explicitly-passed constructor args, so thread_count never shows up there
    # either. Set it directly on every CatBoost step found in the (possibly
    # Pipeline-wrapped) estimator.
    candidates = [estimator] + [
        v for v in params.values() if hasattr(v, "get_params")
    ]
    for candidate in candidates:
        if type(candidate).__module__.startswith("catboost"):
            try:
                candidate.set_params(thread_count=n_jobs)
            except Exception:
                pass


def save_results(
    results,
    directory,
    n_folds,
    columns=(
        "Model",
        "Noise level",
        "AUC",
        "Brier score",
        "Intercept",
        "Slope",
        "Train fit time",
        "Test pred time",
        "Best param",
        "HPO",
    ),
    test_name="MEASUREMENT NOISE",
    n_tr=None,
):
    """Saves result rows to ``<directory>/<test_name>.csv`` atomically.

    Args:
        results: Iterable of result rows.
        directory: Output directory.
        n_folds: Unused; kept for call-site compatibility.
        columns: Column names of the CSV.
        test_name: Name of the test; used as the file stem.
        n_tr: Optional training size appended to the file stem.
    """
    df_results = pd.DataFrame(results, columns=columns)
    file_stem = test_name if n_tr is None else f"{test_name}_{n_tr}"
    final_path = os.path.join(directory, f"{file_stem}.csv")
    tmp_path = os.path.join(directory, f"{file_stem}.csv.tmp-{os.getpid()}")
    df_results.to_csv(tmp_path, index=False)
    os.replace(tmp_path, final_path)
    print(f"Saved {test_name} to {directory}")
    print("\n ======= \n")
    print(test_name, " OVER")


def make_json_safe(params):
    """Converts NumPy scalar values in ``params`` to Python scalars."""
    safe = {}
    for k, v in params.items():
        if isinstance(v, (np.integer, np.floating)):
            safe[k] = v.item()
        else:
            safe[k] = v
    return safe


def log_failure(params, error_msg, logging_file):
    """Appends a failure message to a log file."""
    params = make_json_safe(params)
    with open(logging_file, "a") as f:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "params": params,
            "error": error_msg.strip().split("\n")[-1],
        }
        f.write(json.dumps(log_entry) + "\n")


def predict_proba_batched(model, X, batch_size: int = 100000, n_jobs=1):
    """Returns class-1 probabilities, predicting in batches.

    Works around the CUDA 65 535-block limit in TabPFN's SDPA kernel by
    splitting any large matrix into manageable chunks. The probabilities are
    concatenated in order.
    """
    set_estimator_n_jobs(model, n_jobs)

    if hasattr(model, "predict_proba") and len(X) <= batch_size:
        with _threadpool_ctx(n_jobs):
            return model.predict_proba(X)[:, 1]

    # Best-effort fallback for estimators without predict_proba.
    if not hasattr(model, "predict_proba"):
        if hasattr(model, "decision_function"):
            with _threadpool_ctx(n_jobs):
                scores = model.decision_function(X)
            return special.expit(scores)
        raise TypeError(
            f"Model of type {type(model).__name__} does not support "
            "predict_proba or decision_function"
        )

    out = []
    with _threadpool_ctx(n_jobs):
        for start in range(0, len(X), batch_size):
            X_batch = (
                X.iloc[start : start + batch_size]
                if hasattr(X, "iloc")
                else X[start : start + batch_size]
            )
            out.append(model.predict_proba(X_batch)[:, 1])
    return np.concatenate(out)
