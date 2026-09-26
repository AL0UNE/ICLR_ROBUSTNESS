"""Task-agnostic perturbation primitives for the benchmarks.

Shared by the classification and regression benchmarks (single and composite
drivers).

Keeping a single definition here removes the copy-paste divergence that
previously existed across ``perturbation_function.py``,
``perturbation_function_reg.py`` and the two ``multiperturbation_benchmark*``
scripts. Task-specific primitives (e.g. binary vs. continuous label noise) live
in their respective modules.
"""

import numpy as np
import pandas as pd

from .missing_data_mechanism import add_missingness


def shuffle_features(X_frame, prop=0.5, feat_to_shuffle=None, rng=None):
    """Shuffles a proportion ``prop`` of the columns of ``X_frame``.

    Each selected column is shuffled independently.

    Returns the perturbed frame and the list of shuffled columns (so the same
    columns can be shuffled in a paired validation set).
    """
    X_noisy = X_frame.copy()
    size = X_noisy.shape

    if rng is None:
        rng = np.random.default_rng()

    n_feat_to_shuffle = int(size[1] * prop)
    if feat_to_shuffle is None:
        feat_to_shuffle = list(
            rng.choice(X_noisy.columns, size=n_feat_to_shuffle, replace=False)
        )

    for col in feat_to_shuffle:
        X_noisy[col] = rng.permutation(X_noisy[col].values)

    return X_noisy, feat_to_shuffle


def add_measurement_noise(
    X_frame,
    noise_level,
    cont_features,
    cat_features,
    feature_type="cont & cat",
    rng=None,
):
    """Adds measurement noise to ``X_frame``.

    Continuous features get additive Gaussian noise scaled by ``noise_level * 2
    * std``; categorical features are resampled from their marginal distribution
    with probability ``noise_level``. ``feature_type`` selects which families
    are noised (``"cont"``, ``"cat"`` or ``"cont & cat"``).
    """
    X_noisy = X_frame.copy()
    size = X_noisy.shape
    if rng is None:
        rng = np.random.default_rng()

    if "cont" in feature_type:
        for feature in cont_features:
            std_j = X_noisy[feature].std()
            noise_j = rng.normal(
                loc=0.0, scale=std_j * (noise_level * 2), size=size[0]
            )
            X_noisy[feature] = X_noisy[feature] + noise_j

    if "cat" in feature_type:
        for feature in cat_features:
            mask = rng.random(len(X_frame)) < noise_level

            value_counts = X_noisy[feature].value_counts(normalize=True)
            categories = value_counts.index.values
            probabilities = value_counts.values

            replacements = rng.choice(
                categories, size=mask.sum(), p=probabilities
            )
            X_noisy.loc[mask, feature] = replacements

    return X_noisy


def apply_training_size(X_train, y_train, frac, seed):
    """Subsamples the training set to a fraction ``frac`` (task-agnostic)."""
    X_sampled = X_train.sample(frac=frac, random_state=seed)
    y_sampled = y_train.loc[X_sampled.index]
    return X_sampled, y_sampled


def apply_missingness(
    X_train,
    X_val,
    which_set,
    mechanism,
    level,
    rng,
    torch_gen,
    cont_features,
    cat_features,
    prop_cond_features=0.5,
):
    """Injects missingness into the train and/or validation features.

    Shared by the single and composite drivers.

    The matrix is first completed (continuous columns filled with the
    **training** mean, categorical with the training mode) so that the
    MCAR/MAR/MNAR mechanism in :func:`missing_data_mechanism.add_missingness`
    operates on a complete frame; the resulting mask is then re-applied as
    ``NaN``. For ``Train_Val`` the two sets are masked jointly and split back by
    their (live) row labels, so this stays correct even when a prior composite
    step has changed ``X_train``'s index. ``rng`` and ``torch_gen`` are supplied
    by the caller and consumed in mechanism order.

    Returns ``(X_train, X_val)``.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()

    if which_set == "Train":
        X_noisy = X_train.copy()
        mask = add_missingness(
            X_noisy,
            level,
            mechanism=mechanism,
            prop_cond_features=prop_cond_features,
            rng=rng,
            torch_gen=torch_gen,
            cat_features=cat_features,
            cont_features=cont_features,
        )
        X_train = X_noisy.mask(mask, np.nan)

    elif which_set == "Val":
        X_noisy = X_val.copy()
        mask = add_missingness(
            X_noisy,
            level,
            mechanism=mechanism,
            prop_cond_features=prop_cond_features,
            rng=rng,
            torch_gen=torch_gen,
            cat_features=cat_features,
            cont_features=cont_features,
        )
        X_val = X_noisy.mask(mask, np.nan)

    elif which_set == "Train_Val":
        train_labels, val_labels = X_train.index, X_val.index
        X_noisy = pd.concat([X_train, X_val]).copy()
        mask = add_missingness(
            X_noisy,
            level,
            mechanism=mechanism,
            prop_cond_features=prop_cond_features,
            rng=rng,
            torch_gen=torch_gen,
            cat_features=cat_features,
            cont_features=cont_features,
        )
        X_all = X_noisy.mask(mask, np.nan)
        X_train, X_val = X_all.loc[train_labels], X_all.loc[val_labels]

    return X_train, X_val
