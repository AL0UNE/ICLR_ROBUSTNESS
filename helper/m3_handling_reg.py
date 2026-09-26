"""Dataset loading and access for the regression benchmarks."""

import os
from dataclasses import dataclass

import pandas as pd

from . import dataset_registry as registry

cont_features = [
    "age",
    "heartrate_max",
    "heartrate_min",
    "sysbp_max",
    "sysbp_min",
    "tempc_max",
    "tempc_min",
    "urineoutput",
    "bun_min",
    "bun_max",
    "wbc_min",
    "wbc_max",
    "potassium_min",
    "potassium_max",
    "sodium_min",
    "sodium_max",
    "bicarbonate_min",
    "bicarbonate_max",
    "mingcs",
    "pao2fio2_vent_min",
    "bilirubin_min",
    "bilirubin_max",
]

cat_features = ["aids", "hem", "mets", "admissiontype"]
features = cont_features + cat_features

TARGET = "icu_los"
STRATIFY_COLUMNS = ["gender", "age_group", "ICU_unit"]

# m3 keeps its historical column set (features + intime/outtime to derive the
# ICU length-of-stay target + stratifiers).
required_columns = list(
    dict.fromkeys(features + ["intime", "outtime"] + STRATIFY_COLUMNS)
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
M3_CSV_PATH = os.getenv("M3_CSV_PATH", os.path.join(_DATA_DIR, "m3.csv"))
M4_CSV_PATH = os.getenv("M4_CSV_PATH", os.path.join(_DATA_DIR, "m4.csv"))
EICU_CSV_PATH = os.getenv("EICU_CSV_PATH", os.path.join(_DATA_DIR, "eICU.csv"))


# Per-dataset read spec. ``los_source`` is "times" when ``icu_los`` is derived
# from ``intime``/``outtime`` (MIMIC) and "column" when it is read directly
# (eICU). Columns differ because only m4 carries ``anchor_year_group`` and only
# eICU carries ``region`` / a precomputed ``icu_los``.
_DATASET_LOAD = {
    "m3": {
        "path": M3_CSV_PATH,
        "usecols": required_columns,
        "los_source": "times",
    },
    "m4": {
        "path": M4_CSV_PATH,
        "usecols": list(
            dict.fromkeys(
                features
                + ["intime", "outtime"]
                + STRATIFY_COLUMNS
                + ["anchor_year_group"]
            )
        ),
        "los_source": "times",
    },
    "eICU": {
        "path": EICU_CSV_PATH,
        "usecols": list(
            dict.fromkeys(features + [TARGET] + STRATIFY_COLUMNS + ["region"])
        ),
        "los_source": "column",
    },
}


@dataclass
class DatasetBundle:
    """Development-set view of one dataset for regression.

    Holds the full frame (with the derived ``icu_los`` target), model matrix,
    and target series.
    """

    full_frame: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series


_BUNDLE_CACHE = {}


def _with_icu_los(frame, los_source):
    """Attaches the ``icu_los`` target and drops rows where it is undefined.

    Returns:
        A clean, RangeIndex-ed frame.
    """
    frame = frame.copy()
    if los_source == "times":
        delta = pd.DatetimeIndex(frame["outtime"]) - pd.DatetimeIndex(
            frame["intime"]
        )
        frame[TARGET] = delta.total_seconds() / 3600 / 24
    else:  # "column"
        frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    return frame.loc[~frame[TARGET].isna()].reset_index(drop=True)


def get_dataset(key):
    """Returns the cached :class:`DatasetBundle` for ``key``.

    Args:
        key: Dataset key, one of ``m3``, ``m4`` or ``eICU``.
    """
    if key not in _DATASET_LOAD:
        raise KeyError(
            f"Unknown dataset key: {key!r} (expected one of "
            f"{list(_DATASET_LOAD)})"
        )
    if key not in _BUNDLE_CACHE:
        spec = _DATASET_LOAD[key]
        frame = pd.read_csv(spec["path"], usecols=spec["usecols"])
        frame = _with_icu_los(frame, spec["los_source"])
        _BUNDLE_CACHE[key] = DatasetBundle(
            full_frame=frame,
            X=frame[features],
            y=frame[TARGET],
        )
    return _BUNDLE_CACHE[key]


def build_external_frames(dev_key, test_key):
    """Builds ``(X_train, y_train, df_test)`` for a dev -> test external run.

    Mirrors the classification builder: the m3 <-> m4 overlap cut is applied to
    whichever side is m4, and the missingness filter to the test frame.
    ``df_test`` already carries the ``icu_los`` target (derived/read in
    :func:`get_dataset`).
    """
    dev = get_dataset(dev_key)
    df_test = get_dataset(test_key).full_frame

    X_train, y_train = dev.X, dev.y
    if registry.needs_overlap_cut(dev_key, test_key):
        if dev_key == "m4":
            restricted = registry.restrict_m4_overlap(dev.full_frame)
            X_train, y_train = restricted[features], restricted[TARGET]
        else:  # test_key == "m4"
            df_test = registry.restrict_m4_overlap(df_test)

    df_test = registry.apply_missingness_filter(df_test, features)
    return X_train, y_train, df_test


def load_data(path=None):
    """Backward-compatible m3 loader returning the legacy dict shape.

    Retained for any external caller; new code should use :func:`get_dataset`.
    """
    if path is not None and path != M3_CSV_PATH:
        frame = _with_icu_los(
            pd.read_csv(path, usecols=required_columns), "times"
        )
        return {"mimic_3": frame, "X": frame[features], "y": frame[TARGET]}
    bundle = get_dataset("m3")
    return {"mimic_3": bundle.full_frame, "X": bundle.X, "y": bundle.y}


# PEP 562 lazy attributes so the historical ``m3_handling_reg.X`` / ``.y`` /
# ``.mimic_3`` imports keep resolving to MIMIC-III (the default development
# set).
_M3_ALIASES = {"X": "X", "y": "y", "mimic_3": "full_frame"}


def __getattr__(name):
    if name in _M3_ALIASES:
        return getattr(get_dataset("m3"), _M3_ALIASES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


column_m3 = [
    "Model",
    "Noise level",
    "MAE",
    "RMSE",
    "R2",
    "Train fit time",
    "Test pred time",
    "Best param",
    "HPO",
]

column_m4 = [
    "Model",
    "Variable",
    "Category",
    "MAE",
    "RMSE",
    "R2",
    "Train fit time",
    "Test pred time",
    "Best params",
    "HPO",
]

# Schema for the non-stratified ("overall") temporal/external files (m4.csv,
# eICU.csv). These rows come from evaluate_temporal_bundle's stratify_on=None
# branch, which has no Variable/Category/Best params columns (review N3).
column_m4_overall = [
    "Model",
    "MAE",
    "RMSE",
    "R2",
    "Train fit time",
    "Test pred time",
    "HPO",
]

# Retained for backward compatibility; the per-test-set stratification lists now
# live in dataset_registry.TEST_SET_SPECS.
external_stratifications = [
    ("gender", "eICU_GENDER"),
    ("age_group", "MIMIC_eICU_AGE"),
    ("ICU_unit", "eICU_ICU_UNIT"),
    ("region", "eICU_region"),
]
