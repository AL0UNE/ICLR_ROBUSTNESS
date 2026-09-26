"""MCAR/MAR/MNAR missing-data mechanisms used by the benchmarks."""

import hashlib
from collections import OrderedDict

import numpy as np
import torch
from scipy import optimize
from sklearn.preprocessing import StandardScaler

_MASK_CACHE = OrderedDict()
_MASK_CACHE_MAXSIZE = 64
_MASK_CACHE_ENABLED = True


def configure_mask_cache(enabled=True, size=64):
    """Sets the MAR/MNAR mask-cache policy.

    Driven by config.yaml from the entry scripts.
    """
    global _MASK_CACHE_ENABLED, _MASK_CACHE_MAXSIZE
    _MASK_CACHE_ENABLED = bool(enabled)
    _MASK_CACHE_MAXSIZE = max(0, int(size))
    _MASK_CACHE.clear()


def _mask_cache_key(
    X_float, noise_level, mechanism, prop_cond_features, rng, torch_gen
):
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(X_float, dtype=np.float32).tobytes())
    h.update(
        repr(
            (mechanism, float(noise_level), float(prop_cond_features))
        ).encode()
    )
    h.update(repr(rng.bit_generator.state).encode())
    h.update(torch_gen.get_state().cpu().numpy().tobytes())
    return h.digest()


# #################### MISSING DATA MECHANISMS ############################# #
# Code directly taken from
# https://raw.githubusercontent.com/BorisMuzellec/MissingDataOT/master/utils.py
# #### Missing At Random


def MAR_mask(X, p, p_obs, rng=None, torch_gen=None):
    """Missing at random mechanism with a logistic masking model.

    First, a subset of variables with no missing values is randomly selected.
    The remaining variables have missing values according to a logistic model
    with random weights, re-scaled to attain the desired proportion of
    missing values.
    """
    n, d = X.shape
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    to_torch = torch.is_tensor(X)
    if not to_torch:
        X = torch.from_numpy(X)

    mask = (
        torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)
    )

    if rng is None:
        rng = np.random.default_rng()
    if torch_gen is None:
        torch_gen = torch.Generator()

    d_obs = max(int(p_obs * d), 1)
    d_na = d - d_obs

    idxs_obs = rng.choice(d, size=d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    coeffs = pick_coeffs(X, idxs_obs, idxs_nas, torch_gen=torch_gen)
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na, generator=torch_gen)
    mask[:, idxs_nas] = ber < ps

    return mask


def MNAR_self_mask_logistic(X, p, rng=None, torch_gen=None):
    """Missing not at random mechanism with a logistic self-masking model."""
    n, d = X.shape

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    to_torch = torch.is_tensor(X)
    if not to_torch:
        X = torch.from_numpy(X)

    if rng is None:
        rng = np.random.default_rng()
    if torch_gen is None:
        torch_gen = torch.Generator()

    coeffs = pick_coeffs(X, self_mask=True, torch_gen=torch_gen)
    intercepts = fit_intercepts(X, coeffs, p, self_mask=True)

    ps = torch.sigmoid(X * coeffs + intercepts)

    ber = torch.rand(n, d, generator=torch_gen)
    return ber < ps


def pick_coeffs(
    X, idxs_obs=None, idxs_nas=None, self_mask=False, torch_gen=None
):
    """Draws random logistic-model coefficients for a masking mechanism.

    Args:
        X: Data tensor of shape (n, d).
        idxs_obs: Indices of the always-observed variables.
        idxs_nas: Indices of the variables that receive missing values.
        self_mask: If True, each variable is masked by itself (MNAR).
        torch_gen: Optional ``torch.Generator`` for reproducibility.

    Returns:
        The coefficient tensor, rescaled to unit-variance linear scores.
    """
    n, d = X.shape
    if torch_gen is None:
        torch_gen = torch.Generator()
    if self_mask:
        coeffs = torch.randn(d, generator=torch_gen)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na, generator=torch_gen)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs


def fit_intercepts(X, coeffs, p, self_mask=False):
    """Finds logistic intercepts that yield a missing proportion ``p``.

    Args:
        X: Data tensor of shape (n, d).
        coeffs: Coefficients from ``pick_coeffs``.
        p: Target proportion of missing values.
        self_mask: If True, fit one intercept per variable (MNAR).

    Returns:
        The tensor of fitted intercepts.
    """
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            wx = X * coeffs[j]

            def f(x, wx=wx):
                return torch.sigmoid(wx + x).mean().item() - p

            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):
            wx = X.mv(coeffs[:, j])

            def f(x, wx=wx):
                return torch.sigmoid(wx + x).mean().item() - p

            intercepts[j] = optimize.bisect(f, -50, 50)
    return intercepts


def add_missingness(
    X_noisy,
    noise_level,
    mechanism="MCAR",
    prop_cond_features=0.5,
    rng=None,
    torch_gen=None,
    cat_features=None,
    cont_features=None,
):
    """Returns a copy of ``X_noisy`` with missing values injected.

    Args:
        X_noisy: Feature frame.
        noise_level: Target proportion of missing values.
        mechanism: One of ``MCAR``, ``MAR`` or ``MNAR``.
        prop_cond_features: Proportion of always-observed variables (MAR).
        rng: Optional NumPy random generator.
        torch_gen: Optional ``torch.Generator``.
        cat_features: Names of the categorical columns.
        cont_features: Names of the continuous columns.
    """
    X_noisy = X_noisy.copy()
    X_noisy = X_noisy.copy()

    X_noisy[cont_features] = X_noisy[cont_features].fillna(
        X_noisy[cont_features].mean()
    )
    X_noisy[cat_features] = X_noisy[cat_features].fillna(
        X_noisy[cat_features].mode().loc[0]
    )

    size = X_noisy.shape
    mask = None

    if rng is None:
        rng = np.random.default_rng()
    if torch_gen is None:
        torch_gen = torch.Generator()

    if mechanism == "MCAR":
        return rng.random(size) < noise_level

    X_float = X_noisy.astype(np.float32).values

    cache_key = None
    if (
        _MASK_CACHE_ENABLED
        and _MASK_CACHE_MAXSIZE > 0
        and mechanism in ("MAR", "MNAR")
    ):
        cache_key = _mask_cache_key(
            X_float, noise_level, mechanism, prop_cond_features, rng, torch_gen
        )
        cached = _MASK_CACHE.get(cache_key)
        if cached is not None:
            _MASK_CACHE.move_to_end(cache_key)
            return cached

    if mechanism == "MAR":
        mask = MAR_mask(
            X_float,
            p=noise_level,
            p_obs=prop_cond_features,
            rng=rng,
            torch_gen=torch_gen,
        )
    elif mechanism == "MNAR":
        mask = MNAR_self_mask_logistic(
            X_float, noise_level, rng=rng, torch_gen=torch_gen
        )

    if cache_key is not None and mask is not None:
        _MASK_CACHE[cache_key] = mask
        while len(_MASK_CACHE) > _MASK_CACHE_MAXSIZE:
            _MASK_CACHE.popitem(last=False)

    if mask is None:
        print("DEBUG : EMPTY MASK")
    return mask
