"""Configuration file for hyperparameter grids used in model tuning.

References:
https://arxiv.org/pdf/2505.14415 (LASSO, Ridge, RF, XGBoost, CatBoost)
https://arxiv.org/pdf/2207.08815 (Gradient Boosting)
https://arxiv.org/pdf/2407.04491 (LightGBM)
"""

import numpy as np

PARAM_GRIDS = {
    "Linear": {},
    "LASSO": {
        "lr__alpha": [0.01, 0.1, 1, 10, 100],
    },
    "Ridge": {
        "lr__alpha": [0.01, 0.1, 1, 10, 100],
    },
    "Random Forest": {
        "n_estimators": np.arange(50, 300, 50),
        "max_depth": [None, 2, 3, 4],
        "max_features": ["sqrt", "log2", None, 0.2, 0.4, 0.6, 0.8],
        "min_samples_leaf": list(
            np.round(
                np.logspace(np.log10(1.5), np.log10(50.5), num=5).astype(int)
            )
        ),
        "bootstrap": [True, False],
        "min_impurity_decrease": [0, 0.01, 0.02, 0.05],
        "n_jobs": [1],
    },
    "Gradient Boosting": {
        "gb__learning_rate": list(
            np.round(np.logspace(np.log10(1e-5), np.log10(1), num=6), 5)
        ),
        "gb__subsample": [0.5, 0.8, 1],
        "gb__n_estimators": [1000],
        "gb__max_depth": [None, 2, 3, 4, 5],
        "gb__min_samples_split": [2, 3],
        "gb__min_samples_leaf": list(
            np.round(
                np.logspace(np.log10(1.5), np.log10(50.5), num=5).astype(int)
            )
        ),
        "gb__min_impurity_decrease": [0, 0.01, 0.02, 0.05],
        "gb__max_leaf_nodes": [None, 5, 10, 15],
        "gb__validation_fraction": [0.1],
        "gb__n_iter_no_change": [50],
    },
    "XGBoost": {
        "n_estimators": [1000],
        "max_depth": np.arange(2, 11, 2),
        "learning_rate": list(
            np.round(np.logspace(np.log10(1e-5), np.log10(1), num=6), 5)
        ),
        "min_child_weight": list(
            np.round(np.logspace(np.log10(1), np.log10(100), num=6), 5)
        ),
        "subsample": [0.5, 0.8, 1],
        "colsample_bylevel": [0.5, 0.8, 1],
        "colsample_bytree": [0.5, 0.8, 1],
        "gamma": list(
            np.round(np.logspace(np.log10(1e-8), np.log10(7), num=6), 5)
        ),
        "lambda": list(
            np.round(np.logspace(np.log10(1), np.log10(4), num=4), 3)
        ),
        "alpha": list(
            np.round(np.logspace(np.log10(1e-8), np.log10(100), num=5), 7)
        ),
        "early_stopping_rounds": [50],
        "tree_method": ["hist"],
        "device": ["cuda"],
    },
    "LightGBM": {
        "n_estimators": [1000],
        "bagging_freq": [1],
        "num_leaves": [2, 10, 100, 1000, 10000],
        "learning_rate": list(
            np.round(np.logspace(np.log10(1e-5), np.log10(1), num=6), 5)
        ),
        "subsample": [0.5, 0.8, 1],
        "feature_fraction": [0.5, 0.8, 1],
        "min_data_in_leaf": list(
            np.round(
                np.logspace(np.log10(100), np.log10(1e5), num=6).astype(int)
            )
        ),
        "min_sum_hessian_in_leaf": list(
            np.logspace(np.log10(1e-7), np.log10(1e5), num=5)
        ),
        "lambda_l1": [0, 1e-16, 1e-10, 0.0001, 100.0],
        "lambda_l2": [0, 1e-16, 1e-10, 0.0001, 100.0],
        "verbose": [-1],
        "objective": ["regression"],
        "metric": ["rmse"],
        "device": ["gpu"],
        "gpu_use_dp": [True],
    },
    "CatBoost": {
        "iterations": [1000],
        "depth": np.arange(2, 7, 2),
        "learning_rate": list(
            np.round(np.logspace(np.log10(1e-5), np.log10(1), num=6), 5)
        ),
        "bagging_temperature": [0, 0.25, 0.5, 0.75, 1],
        "l2_leaf_reg": list(
            np.round(np.logspace(np.log10(1), np.log10(10), num=4), 1)
        ),
        "one_hot_max_size": [0, 5, 15, 25],
        "random_strength": [1, 10, 20],
        "leaf_estimation_iterations": [1, 10, 20],
        "od_wait": [50],
        "od_type": ["Iter"],
        "eval_metric": ["RMSE"],
        "logging_level": ["Silent"],
    },
    "RealMLP": {
        "realmlp__num_emb_type": [
            "none",
            "pbld",
            "plr",
        ], 
        "realmlp__add_front_scale": [True, False], 
        "realmlp__lr": [0.02, 0.1, 0.3],
        "realmlp__p_drop": [0.0, 0.15, 0.3], 
        "realmlp__act": ["relu", "selu", "mish"], 
        "realmlp__hidden_sizes": [
            [64, 64, 64, 64, 64],
            [512],
        ], 
        "realmlp__wd": [0.0, 2e-2],
        "realmlp__plr_sigma": [0.05, 0.1, 0.5],
        "realmlp__use_ls": [
            False
        ], 
        "realmlp__use_early_stopping": [True],
    },
}
