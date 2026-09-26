"""Configuration file for hyperparameter grids used in model tuning.

References:
https://arxiv.org/pdf/2506.16791
https://arxiv.org/pdf/2207.08815 (Gradient Boosting)
"""

import numpy as np

PARAM_GRIDS = {
    "Logistic": {
        "lr__C": [1e12],
    },
    "LASSO": {
        "lr__l1_ratio": [1],
        "lr__solver": ["liblinear"],
        "lr__C": [0.01, 0.1, 1, 10, 100, 1000],
    },
    "Ridge": {
        "lr__l1_ratio": [0],
        "lr__C": [0.01, 0.1, 1, 10, 100, 1000],
    },
    "Random Forest": {
        # In TabArena they use ensembling over cross-validation, resulting in
        # 400 trees overall.
        "n_estimators": [
            100,
            200,
            400,
        ],
        "max_features": [0.4, 0.6, 0.8, 1],
        "max_samples": [0.5, 0.8, 1],
        "min_samples_split": [2, 3, 4],
        "bootstrap": [True, False],
        "min_impurity_decrease": [0.00001, 0.0001, 0.001],
        "n_jobs": [1],
    },
    "Gradient Boosting": {
        "gb__loss": ["log_loss", "exponential"],
        "gb__learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1],
        "gb__subsample": [0.5, 0.8, 1],
        # In the paper they use between 10 and 1000 (need to check if it is with
        # or without early stopping).
        "gb__n_estimators": [1000],
        "gb__criterion": ["friedman_mse", "squared_error"],
        "gb__max_depth": [None, 2, 3, 4, 5],
        "gb__min_samples_split": [2, 3],
        "gb__min_samples_leaf": [1, 3, 8, 20, 50],
        "gb__min_impurity_decrease": [0, 0.01, 0.02, 0.05],
        "gb__max_leaf_nodes": [None, 5, 10, 15],
        "gb__validation_fraction": [0.1],
        "gb__n_iter_no_change": [50],
    },
    "XGBoost": {
        "n_estimators": [1000],
        "learning_rate": list(
            np.round(np.logspace(np.log10(5e-3), np.log10(1e-1), num=4), 5)
        ),
        "max_depth": [4, 6, 8, 10],
        "min_child_weight": list(
            np.round(np.logspace(np.log10(1e-3), np.log10(5), num=4), 5)
        ),
        "subsample": [0.6, 0.8, 1],
        "colsample_bylevel": [0.6, 0.8, 1],
        "colsample_bynode": [0.6, 0.8, 1],
        "reg_lambda": [
            0.0001,
            0.001,
            0.1,
            1,
            5,
        ],  ## tabarena does not use loguniform but uniform
        "reg_alpha": [
            0.0001,
            0.001,
            0.1,
            1,
            5,
        ],  ## tabarena does not use loguniform but uniform
        "grow_policy": ["depthwise", "lossguide"],
        # Not sure this one matters since we do not have many categorical
        # features.
        "max_cat_to_onehot": [
            8,
            16,
            32,
            64,
            100,
        ],
        "max_leaves": [8, 32, 64, 256, 1014],
        "early_stopping_rounds": [50],
        "tree_method": ["hist"],
    },
    "LightGBM": {
        "n_estimators": [1000],
        "learning_rate": list(
            np.round(np.logspace(np.log10(5e-3), np.log10(1e-1), num=4), 5)
        ),
        "colsample_bytree": [0.4, 0.6, 0.8, 1],
        "subsample": [0.7, 0.8, 0.9, 1],
        "subsample_freq": [1],
        "num_leaves": [2, 25, 50, 100, 200],
        "min_child_samples": [1, 4, 16, 64],
        "extra_trees": [False, True],
        "min_data_per_group": [2, 25, 50, 100],
        "cat_l2": list(
            np.round(np.logspace(np.log10(5e-3), np.log10(2), num=4), 5)
        ),
        "cat_smooth": list(
            np.round(np.logspace(np.log10(1e-3), np.log10(100), num=4), 5)
        ),
        "max_cat_to_onehot": [8, 15, 50, 100],
        "reg_alpha": [
            0.0001,
            0.001,
            0.01,
            0.1,
            1,
        ],  ## tabarena does not use loguniform but uniform
        "reg_lambda": [
            0.0001,
            0.001,
            0.01,
            0.1,
            2,
        ],  ## tabarena does not use loguniform but uniform
        "verbose": [-1],
        "objective": ["binary"],
        "metric": ["auc"],
    },
    "CatBoost": {
        "iterations": [1000],
        "learning_rate": list(
            np.round(np.logspace(np.log10(5e-3), np.log10(1e-1), num=4), 5)
        ),
        "bootstrap_type": ["Bernoulli"],
        "subsample": [0.7, 0.8, 0.9, 1],
        "grow_policy": ["SymmetricTree", "Depthwise"],
        "depth": [4, 6, 8],
        "colsample_bylevel": [0.85, 0.90, 0.95, 1],
        "l2_leaf_reg": list(
            np.round(np.logspace(np.log10(1e-4), np.log10(5), num=4), 5)
        ),
        "leaf_estimation_iterations": [1, 5, 10, 20],
        "one_hot_max_size": [8, 16, 32, 100],
        "model_size_reg": list(
            np.round(np.logspace(np.log10(1e-1), np.log10(1.5), num=4), 3)
        ),
        "max_ctr_complexity": [2, 3, 4, 5],
        "boosting_type": ["Plain"],
        "max_bin": [254],
        "od_wait": [50],
        "od_type": ["Iter"],
        "eval_metric": ["AUC"],
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
        "realmlp__ls_eps": [0.0, 0.1],
        "realmlp__use_early_stopping": [True],
    },
}
