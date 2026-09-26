"""Config-driven selection of which models a benchmark runs.

Lets a run be restricted to a subset of models from ``config.yaml`` (the
``models`` key) so the benchmark can be executed partially -- e.g. skip the
heavy GPU / foundation models for a quick pass. The semantics are designed so a
model can be switched off either by setting it to ``false`` or by
commenting/omitting its line:

  * ``models`` absent / empty / null   -> run **all** models (the default).
  * ``models`` present (non-empty map)  -> run exactly the models whose value is
    truthy. A model set to ``false``, omitted, or commented out is skipped.

This module is intentionally dependency-free (pure builtins): it must be
importable and testable without torch / catboost / tabpfn / tabicl, unlike
``model_factory`` which constructs the actual estimators.
"""


CLASSIFICATION_MODEL_NAMES = (
    "Logistic",
    "LASSO",
    "Ridge",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "TabPFNv3",
    "TabICLv2",
    "TabDPT",
    "RealMLP",
)
REGRESSION_MODEL_NAMES = (
    "Linear",
    "LASSO",
    "Ridge",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "TabPFNv3",
    "TabICLv2",
    "TabDPT",
    "RealMLP",
)
KNOWN_MODEL_NAMES = frozenset(CLASSIFICATION_MODEL_NAMES) | frozenset(
    REGRESSION_MODEL_NAMES
)


def select_models(models, models_filter):
    """Filters the built ``models`` dict by the ``models`` config mapping.

    ``models`` is the ordered name -> estimator dict from ``model_factory``;
    ``models_filter`` is ``cfg.models`` (a name -> bool mapping, or ``None``).

    Returns an ordered dict (factory order preserved) of the models to run. A
    falsey / missing ``models_filter`` returns every model. Names in the filter
    that are valid but belong to the other task (e.g. ``"Logistic"`` while
    running regression) are simply ignored, so one shared config block can cover
    all four benchmarks.

    Raises ``ValueError`` on an unknown model name (typo guard) or when the
    filter disables every model for this benchmark, and ``TypeError`` if the
    filter is not a mapping.
    """
    if not models_filter:
        return models

    if not isinstance(models_filter, dict):
        raise TypeError(
            "config 'models' must be a mapping of model name -> true/false "
            "(e.g. {'TabPFNv3': false}); got "
            f"{type(models_filter).__name__}."
        )

    unknown = sorted(
        name for name in models_filter if name not in KNOWN_MODEL_NAMES
    )
    if unknown:
        raise ValueError(
            f"Unknown model name(s) in config 'models': {unknown}. "
            f"Known model names: {sorted(KNOWN_MODEL_NAMES)}."
        )

    selected = {
        name: est for name, est in models.items() if models_filter.get(name)
    }
    if not selected:
        raise ValueError(
            "config 'models' leaves no model enabled for this benchmark. "
            "Enable at "
            "least one (set it to true) or remove the 'models' block to run "
            "all models."
        )

    skipped = [name for name in models if name not in selected]
    if skipped:
        print(
            f"[models] enabled: {list(selected)} | skipped: {skipped}",
            flush=True,
        )

    return selected
