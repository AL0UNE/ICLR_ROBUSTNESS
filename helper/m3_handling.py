"""Dataset loading and access for the classification benchmarks."""

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
outcome = ["hospital_mortality"]

TARGET = "hospital_mortality"
# Proxy label-noise variants, mapped to the CSV column each reads. The
# ``hospital_death`` baseline reuses the target itself so needs no extra column;
# the real proxies require their column to be present in the dataset. A dataset
# offers a variant only when its column is loaded (see ``proxy_columns`` in
# ``_DATASET_LOAD``): m3 and m4 carry both, eICU only ``death_icu``.
PROXY_VARIANTS = {
    "hospital_death": None,
    "icu_death": "death_icu",
    "overall_death": "death_overall",
}
PROXY_COLUMNS = ["death_icu", "death_overall"]
STRATIFY_COLUMNS = ["gender", "age_group", "ICU_unit"]

# m3 keeps its historical column set (features + target + proxies +
# stratifiers).
required_columns = list(
    dict.fromkeys(features + [TARGET] + PROXY_COLUMNS + STRATIFY_COLUMNS)
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
M3_CSV_PATH = os.getenv("M3_CSV_PATH", os.path.join(_DATA_DIR, "m3.csv"))
M4_CSV_PATH = os.getenv("M4_CSV_PATH", os.path.join(_DATA_DIR, "m4.csv"))
EICU_CSV_PATH = os.getenv("EICU_CSV_PATH", os.path.join(_DATA_DIR, "eICU.csv"))


# Columns to read per dataset. They differ because m3 and m4 carry both proxy
# outcomes, eICU only ``death_icu``; only m4 carries ``anchor_year_group`` and
# only eICU carries ``region``. ``proxy_columns`` lists which proxy outcomes
# each set provides. Listing columns explicitly keeps the read strict (a missing
# column fails loudly) and small.
_DATASET_LOAD = {
    "m3": {
        "path": M3_CSV_PATH,
        "usecols": required_columns,
        "proxy_columns": ["death_icu", "death_overall"],
    },
    "m4": {
        "path": M4_CSV_PATH,
        "usecols": list(
            dict.fromkeys(
                features
                + [TARGET]
                + PROXY_COLUMNS
                + STRATIFY_COLUMNS
                + ["anchor_year_group"]
            )
        ),
        "proxy_columns": ["death_icu", "death_overall"],
    },
    "eICU": {
        "path": EICU_CSV_PATH,
        "usecols": list(
            dict.fromkeys(
                features + [TARGET] + STRATIFY_COLUMNS + ["region", "death_icu"]
            )
        ),
        "proxy_columns": ["death_icu"],
    },
}


@dataclass
class DatasetBundle:
    """Development-set view of one dataset.

    Holds the full frame, model matrix, target and whichever proxy-outcome
    series the dataset provides for proxy label noise (m3 and m4: both, eICU:
    ``death_icu`` only).
    """

    full_frame: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    y_proxy_death_icu: pd.Series | None = None
    y_proxy_death_overall: pd.Series | None = None

    @property
    def available_proxy_variants(self):
        """Proxy label-noise variants this dataset can run.

        The ``hospital_death`` baseline is included only alongside at least one
        real proxy: m3/m4 run all three, eICU runs
        ``hospital_death``/``icu_death``.
        """
        real = [
            name
            for name, column in PROXY_VARIANTS.items()
            if column is not None
            and getattr(self, f"y_proxy_{column}") is not None
        ]
        return ["hospital_death"] + real if real else []

    @property
    def has_proxy(self) -> bool:
        """True if at least one proxy label-noise variant is available."""
        return bool(self.available_proxy_variants)


_BUNDLE_CACHE = {}


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
        proxy_columns = spec["proxy_columns"]
        _BUNDLE_CACHE[key] = DatasetBundle(
            full_frame=frame,
            X=frame[features],
            y=frame[TARGET],
            y_proxy_death_icu=frame["death_icu"]
            if "death_icu" in proxy_columns
            else None,
            y_proxy_death_overall=frame["death_overall"]
            if "death_overall" in proxy_columns
            else None,
        )
    return _BUNDLE_CACHE[key]


def build_external_frames(dev_key, test_key):
    """Builds ``(X_train, y_train, df_test)`` for a dev -> test external run.

    Applies the m3 <-> m4 overlap cut to whichever side is m4 (see
    :mod:`helper.dataset_registry`) and the defensive missingness filter to the
    test frame. The development model matrix/target come from the dev bundle,
    except that when m4 is the *training* side of an m3 pairing it is restricted
    to the non-overlapping years.
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
        frame = pd.read_csv(path, usecols=required_columns)
        return {
            "mimic_3": frame,
            "X": frame[features],
            "y": frame[TARGET],
            "y_proxy_death_icu": frame["death_icu"],
            "y_proxy_death_overall": frame["death_overall"],
        }
    bundle = get_dataset("m3")
    return {
        "mimic_3": bundle.full_frame,
        "X": bundle.X,
        "y": bundle.y,
        "y_proxy_death_icu": bundle.y_proxy_death_icu,
        "y_proxy_death_overall": bundle.y_proxy_death_overall,
    }

_M3_ALIASES = {
    "X": "X",
    "y": "y",
    "mimic_3": "full_frame",
    "y_proxy_death_icu": "y_proxy_death_icu",
    "y_proxy_death_overall": "y_proxy_death_overall",
}


def __getattr__(name):
    if name in _M3_ALIASES:
        return getattr(get_dataset("m3"), _M3_ALIASES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


column_m3 = [
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
]

column_m4 = [
    "Model",
    "Variable",
    "Category",
    "AUC",
    "Brier score",
    "Intercept",
    "Slope",
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
    "AUC",
    "Brier score",
    "Intercept",
    "Slope",
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
