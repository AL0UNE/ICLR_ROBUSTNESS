"""Typed benchmark configuration loaded from ``config.yaml``.

``load_config(section)`` returns a :class:`BenchConfig` for one of the four
entry-point scripts. Every section shares the same single ``defaults`` block —
per-benchmark overrides are deliberately not supported, so the four experiments
always run under identical settings and their results stay comparable. The
configuration file (and PyYAML) are optional: if ``config.yaml`` is absent the
built-in defaults are used; if it is present its ``defaults`` block overlays
them.
"""

import os
import warnings
from dataclasses import dataclass, field, fields

try:
    import yaml
except (
    Exception
):  # pragma: no cover - PyYAML is an optional convenience dependency
    yaml = None

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.getenv(
    "BENCHMARK_CONFIG", os.path.join(ROOT_DIR, "config", "config.yaml")
)

# Defaults shared by every benchmark. Kept in sync with config.yaml's committed
# `defaults` block (except `models`, where None means "run every model"), so
# deleting config.yaml reproduces the same run.
_DEFAULTS = {
    "random_state": 42,
    "n_folds": 10,
    "n_jobs": 1,
    "n_jobs_train": 1,
    "n_jobs_predict": 1,
    "n_jobs_gs": 1,
    "hpo": False,
    "n_search_gs": 15,
    "no_inner_fold": False,
    "split_train_predict": True,
    "fold_n_jobs": 1,
    "xgboost_device": "cpu",
    "lightgbm_device": "cpu",
    "realmlp_device": "cpu",
    "preset_enabled": False,
    "preset_strict": False,
    "preset_path": "preset/results_hpo_cpu",
    "preset_path_reg": "preset/results_hpo_reg_cpu",
    "preset_top_k": 5,
    "n_composites": 10,
    "composite_start": 1,
    "n_perturb_types_per_composite": 3,
    "batch_size": None,
    "n_training_sample": 1,
    "mask_cache_enabled": True,
    "mask_cache_size": 64,
    # None -> run all models; a {name: bool} map restricts the run.
    "models": None,
}

_SECTIONS = (
    "classification",
    "classification_composite",
    "regression",
    "regression_composite",
)


@dataclass(frozen=True)
class BenchConfig:
    """Typed benchmark settings loaded from ``config.yaml``."""

    random_state: int
    n_folds: int
    n_jobs: int
    n_jobs_train: int
    n_jobs_predict: int
    n_jobs_gs: int
    hpo: bool
    n_search_gs: int
    no_inner_fold: bool
    split_train_predict: bool
    train_across_folds: bool
    fold_n_jobs: int
    xgboost_device: str
    lightgbm_device: str
    realmlp_device: str
    preset_enabled: bool
    preset_strict: bool
    preset_path: str
    preset_path_reg: str
    preset_top_k: int
    n_composites: int
    composite_start: int
    n_perturb_types_per_composite: int
    batch_size: int | None
    n_training_sample: int
    mask_cache_enabled: bool
    mask_cache_size: int
    # Optional {model_name: bool} selection; None means run every model.
    models: dict | None = field(default=None, hash=False)


def load_config(section, path=None):
    """Loads configuration for ``section`` (e.g. ``"classification"``).

    Every section resolves to the same settings — built-in ``_DEFAULTS``
    overlaid by ``config.yaml``'s ``defaults`` block. 
    """
    if section not in _SECTIONS:
        raise ValueError(
            f"Unknown config section {section!r}; expected one of "
            f"{sorted(_SECTIONS)}"
        )

    merged = dict(_DEFAULTS)

    cfg_path = path or CONFIG_PATH
    if os.path.isfile(cfg_path):
        if yaml is None:
            warnings.warn(
                f"{cfg_path} is present but PyYAML is not installed; using "
                "built-in defaults. "
                "Install it with `pip install pyyaml`."
            )
        else:
            with open(cfg_path, encoding="utf-8") as file_obj:
                doc = yaml.safe_load(file_obj) or {}
            if doc.get("benchmarks"):
                raise ValueError(
                    f"{cfg_path} contains a 'benchmarks' block. Per-benchmark "
                    "overrides are "
                    "not supported (all four experiments must share the same "
                    "settings); "
                    "move these keys into the 'defaults' block."
                )
            merged.update(doc.get("defaults") or {})

    known = {f.name for f in fields(BenchConfig)}
    unknown = set(merged) - known
    if unknown:
        raise ValueError(
            f"Unknown configuration keys for section {section!r}: "
            f"{sorted(unknown)}"
        )

    return BenchConfig(**{name: merged[name] for name in known})
