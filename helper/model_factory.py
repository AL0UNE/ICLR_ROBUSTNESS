"""Factories for the benchmark model dictionaries and device-aware HPO grids.
Centralises the estimator definitions.
"""

import copy

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tabicl import TabICLClassifier, TabICLRegressor
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion
from xgboost import XGBClassifier, XGBRegressor

from .model_helpers import build_preprocessors
from .model_selection import (
    select_models,  # noqa: F401  (re-exported for the drivers)
)

try:
    from pytabkit import (
        RealMLP_TD_Classifier,
        RealMLP_TD_Regressor,
    )

    _HAS_PYTABKIT = True
except Exception:  # pragma: no cover - exercised only when pytabkit is absent
    RealMLP_TD_Classifier = RealMLP_TD_Regressor = None
    _HAS_PYTABKIT = False


FOUNDATION_DEVICE = "cuda"
_TABICL_V2_CKPT = "tabicl-classifier-v2-20260212.ckpt"


class _MissingRealMLP(BaseEstimator):
    """Stand-in for RealMLP when ``pytabkit`` is not installed.

    Keeps "RealMLP" in the model lineup (so enabling it in config is never
    silently dropped) while constructing the models dict without pytabkit
    present. It accepts the RealMLP hyper-parameters so an HPO ``set_params``
    does not fail before fit; the actionable error is raised only if the model
    is actually fit/used.
    """

    _MSG = (
        "RealMLP requires the optional 'pytabkit' package, which is not "
        "installed. "
        "Install it with `uv pip install pytabkit` (CPU-only is fine)."
    )

    def __init__(
        self,
        num_emb_type="pbld",
        add_front_scale=True,
        lr=0.04,
        p_drop=0.15,
        act="selu",
        hidden_sizes=(256, 256, 256),
        wd=0.0,
        plr_sigma=0.1,
        ls_eps=0.1,
        device="cpu",
        random_state=None,
        n_threads=None,
    ):
        self.num_emb_type = num_emb_type
        self.add_front_scale = add_front_scale
        self.lr = lr
        self.p_drop = p_drop
        self.act = act
        self.hidden_sizes = hidden_sizes
        self.wd = wd
        self.plr_sigma = plr_sigma
        self.ls_eps = ls_eps
        self.device = device
        self.random_state = random_state
        self.n_threads = n_threads

    def fit(self, X, y=None):
        raise ImportError(self._MSG)

    def predict(self, X):
        raise ImportError(self._MSG)

    def predict_proba(self, X):
        raise ImportError(self._MSG)


def _make_realmlp(task, random_state, device="cpu"):
    """Builds a RealMLP estimator (or a placeholder if pytabkit is absent).

    Defaults to CPU (the baseline protocol used elsewhere), but accepts an
    explicit ``device`` override (e.g. "cuda") for callers that want RealMLP
    run on GPU. The TD hyper-parameters are left at their library defaults and
    overridden by the HPO grid (see ``RealMLP_grid`` in the hpo modules).
    """
    if not _HAS_PYTABKIT:
        return _MissingRealMLP(device=device, random_state=random_state)
    cls = (
        RealMLP_TD_Classifier
        if task == "classification"
        else RealMLP_TD_Regressor
    )
    return cls(device=device, random_state=random_state)


class _TabDPTClassifierLazy(ClassifierMixin, BaseEstimator):
    """Deferred-construction sklearn adapter around ``tabdpt.TabDPTClassifier``.

    Besides laziness, this papers over two sklearn-compatibility gaps in tabdpt:
    it expects float ndarrays with integer labels 0..K-1 and predicts encoded
    class indices, and its ``get_params`` is broken (``__init__`` does not store
    all arguments). Labels are round-tripped through a ``LabelEncoder`` so
    ``classes_``/``predict``/``predict_proba`` behave like any sklearn
    classifier (as required by the
    ``CalibratedClassifierCV(FrozenEstimator(...))`` step).
    """

    def __init__(self, random_state=None, device=None):
        self.random_state = random_state
        self.device = device

    def fit(self, X, y):
        from tabdpt import TabDPTClassifier

        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(np.asarray(y).ravel())
        self.classes_ = self._label_encoder.classes_
        # compile=False: torch.compile needs a host C compiler (triton) and its
        # warm-up would land in the benchmark's first prediction-time
        # measurement.
        self.estimator_ = TabDPTClassifier(
            verbose=False, compile=False, device=self.device
        )
        self.estimator_.fit(np.asarray(X, dtype=np.float64), y_encoded)
        return self

    def predict(self, X):
        indices = self.estimator_.predict(
            np.asarray(X, dtype=np.float64), seed=self.random_state
        )
        return self.classes_[np.asarray(indices, dtype=int)]

    def predict_proba(self, X):
        return self.estimator_.ensemble_predict_proba(
            np.asarray(X, dtype=np.float64), seed=self.random_state
        )


class _TabDPTRegressorLazy(RegressorMixin, BaseEstimator):
    """Deferred-construction adapter around ``tabdpt.TabDPTRegressor``."""

    def __init__(self, random_state=None, device=None):
        self.random_state = random_state
        self.device = device

    def fit(self, X, y):
        from tabdpt import TabDPTRegressor

        # compile=False: see _TabDPTClassifierLazy.fit.
        self.estimator_ = TabDPTRegressor(
            verbose=False, compile=False, device=self.device
        )
        self.estimator_.fit(
            np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.float64).ravel(),
        )
        return self

    def predict(self, X):
        return self.estimator_.predict(
            np.asarray(X, dtype=np.float64), seed=self.random_state
        )


def _lgbm_kwargs(lightgbm_device):
    kwargs = {"device": lightgbm_device}
    if lightgbm_device == "gpu":
        kwargs["gpu_use_dp"] = True
    return kwargs


def build_classification_models(
    cont_features,
    cat_features,
    random_state,
    xgboost_device="cuda",
    lightgbm_device="gpu",
    realmlp_device="cpu",
):
    """Builds the classification model dictionary (hospital mortality)."""
    preprocessor_scaled, preprocessor_unscaled = build_preprocessors(
        cont_features, cat_features, random_state
    )
    return {
        # Linear methods
        "Logistic": Pipeline(
            [
                ("preprocessor", preprocessor_scaled),
                ("lr", LogisticRegression(C=1e12)),
            ]
        ),
        "LASSO": Pipeline(
            [
                ("preprocessor", preprocessor_scaled),
                ("lr", LogisticRegression(l1_ratio=1, solver="liblinear")),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("preprocessor", preprocessor_scaled),
                ("lr", LogisticRegression(l1_ratio=0)),
            ]
        ),
        # Tree-based methods
        "Random Forest": RandomForestClassifier(n_jobs=1),
        "Gradient Boosting": Pipeline(
            [
                ("preprocessor", preprocessor_unscaled),
                ("gb", GradientBoostingClassifier()),
            ]
        ),
        "XGBoost": XGBClassifier(tree_method="hist", device=xgboost_device),
        "LightGBM": LGBMClassifier(**_lgbm_kwargs(lightgbm_device)),
        "CatBoost": CatBoostClassifier(logging_level="Silent"),
        # Tabular foundation models
        "TabPFNv3": TabPFNClassifier.create_default_for_version(
            ModelVersion.V3, device=FOUNDATION_DEVICE
        ),
        "TabICLv2": TabICLClassifier(
            checkpoint_version=_TABICL_V2_CKPT, device=FOUNDATION_DEVICE
        ),
        "TabDPT": _TabDPTClassifierLazy(
            random_state=random_state, device=FOUNDATION_DEVICE
        ),
        # Per-dataset neural baselines (pytabkit)
        "RealMLP": Pipeline(
            [
                ("preprocessor", preprocessor_unscaled),
                (
                    "realmlp",
                    _make_realmlp(
                        "classification", random_state, device=realmlp_device
                    ),
                ),
            ]
        ),
    }


def build_regression_models(
    cont_features,
    cat_features,
    random_state,
    xgboost_device="cuda",
    lightgbm_device="gpu",
    realmlp_device="cpu",
):
    """Builds the regression model dictionary (ICU length of stay)."""
    preprocessor_scaled, preprocessor_unscaled = build_preprocessors(
        cont_features, cat_features, random_state
    )
    return {
        # Linear methods
        "Linear": Pipeline(
            [("preprocessor", preprocessor_scaled), ("lr", LinearRegression())]
        ),
        "LASSO": Pipeline(
            [("preprocessor", preprocessor_scaled), ("lr", Lasso())]
        ),
        "Ridge": Pipeline(
            [("preprocessor", preprocessor_scaled), ("lr", Ridge())]
        ),
        # Tree-based methods
        "Random Forest": RandomForestRegressor(n_jobs=1),
        "Gradient Boosting": Pipeline(
            [
                ("preprocessor", preprocessor_unscaled),
                ("gb", GradientBoostingRegressor()),
            ]
        ),
        "XGBoost": XGBRegressor(tree_method="hist", device=xgboost_device),
        "LightGBM": LGBMRegressor(**_lgbm_kwargs(lightgbm_device)),
        "CatBoost": CatBoostRegressor(),
        # Tabular foundation models.
        "TabPFNv3": TabPFNRegressor.create_default_for_version(
            ModelVersion.V3,
            memory_saving_mode=True,
            fit_mode="low_memory",
            device=FOUNDATION_DEVICE,
        ),
        "TabICLv2": TabICLRegressor(device=FOUNDATION_DEVICE),
        "TabDPT": _TabDPTRegressorLazy(
            random_state=random_state, device=FOUNDATION_DEVICE
        ),
        # Per-dataset neural baselines (pytabkit)
        "RealMLP": Pipeline(
            [
                ("preprocessor", preprocessor_unscaled),
                (
                    "realmlp",
                    _make_realmlp(
                        "regression", random_state, device=realmlp_device
                    ),
                ),
            ]
        ),
    }


def build_param_grids(
    param_grids, xgboost_device="cuda", lightgbm_device="gpu"
):
    """Returns a copy of ``param_grids`` with XGBoost/LightGBM devices set.

    Lets the HPO search spaces follow the same CPU/GPU choice as the models.
    """
    grids = copy.deepcopy(param_grids)
    if "XGBoost" in grids and "device" in grids["XGBoost"]:
        grids["XGBoost"]["device"] = [xgboost_device]
    if "LightGBM" in grids and "device" in grids["LightGBM"]:
        grids["LightGBM"]["device"] = [lightgbm_device]
    return grids
