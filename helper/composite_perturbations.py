"""Shared building blocks for the composite (multi-perturbation) benchmarks.

This module is the single source of truth for the two composite drivers
(``multiperturbation_benchmark.py`` and ``multiperturbation_benchmark_reg.py``):

* the perturbation catalogues (:data:`perturb_dic`, :data:`perturb_dic_reg`),
* the composite sampler (:func:`sample_composite_perturbations`) and report
  writer,
* the per-type level samplers, and
* the variant-string parsers shared by both drivers.
"""

import json

import numpy as np

# Classification catalogue (hospital mortality). Mirrors the single-perturbation
# coverage in mimic_benchmark.py.
perturb_dic = {
    "label_noise": [
        "random",
        "0to1",
        "1to0",
        "conditional",
        "proxy_hospital_death",
        "proxy_icu_death",
        "proxy_overall_death",
    ],
    "input_noise": [
        "train_continuous",
        "train_categorical",
        "train_continuous_and_categorical",
        "val_continuous",
        "val_categorical",
        "val_continuous_and_categorical",
        "train_val_continuous",
        "train_val_categorical",
        "train_val_continuous_and_categorical",
    ],
    "missing_data": [
        "train_mcar",
        "train_mar",
        "train_mnar",
        "val_mcar",
        "val_mar",
        "val_mnar",
        "train_val_mcar",
        "train_val_mar",
        "train_val_mnar",
    ],
    "feature_permutation": [
        "train_feature_shuffle",
        "val_feature_shuffle",
        "train_val_feature_shuffle",
    ],
    "class_imbalance": [
        "negative_class_downsampling",
    ],
    "training_data_regime": [
        "reduced_training_size",
    ],
}


# Regression catalogue (ICU length of stay). No class imbalance / proxy labels:
# the target is continuous.
perturb_dic_reg = {
    "label_noise": [
        "random",
        "conditional",
    ],
    "input_noise": [
        "train_continuous",
        "train_categorical",
        "train_continuous_and_categorical",
        "val_continuous",
        "val_categorical",
        "val_continuous_and_categorical",
        "train_val_continuous",
        "train_val_categorical",
        "train_val_continuous_and_categorical",
    ],
    "missing_data": [
        "train_mcar",
        "train_mar",
        "train_mnar",
        "val_mcar",
        "val_mar",
        "val_mnar",
        "train_val_mcar",
        "train_val_mar",
        "train_val_mnar",
    ],
    "feature_permutation": [
        "train_feature_shuffle",
        "val_feature_shuffle",
        "train_val_feature_shuffle",
    ],
    "training_data_regime": [
        "reduced_training_size",
    ],
}


def sample_level_classification(perturb_type, variant, rng):
    """Samples a perturbation magnitude for the classification composites."""
    if perturb_type == "label_noise":
        if variant.startswith("proxy_"):
            if variant == "proxy_hospital_death":
                return "hospital_death"
            if variant == "proxy_icu_death":
                return "icu_death"
            return "overall_death"
        return float(rng.choice(np.linspace(0.1, 0.5, 5)))

    if perturb_type == "input_noise":
        return float(rng.choice(np.linspace(0.1, 0.5, 5)))

    if perturb_type == "missing_data":
        return float(rng.choice(np.linspace(0.1, 0.5, 5)))

    if perturb_type == "feature_permutation":
        return float(rng.choice(np.linspace(0.1, 0.7, 7)))

    if perturb_type == "class_imbalance":
        return float(rng.choice([0.25, 0.5, 0.75, 1.0]))

    if perturb_type == "training_data_regime":
        return float(rng.choice([0.25, 0.5, 0.75, 1.0]))

    raise ValueError(f"Unknown perturbation type: {perturb_type}")


def sample_level_regression(perturb_type, variant, rng):
    """Samples a perturbation magnitude for the regression composite driver."""
    del variant  # variant does not influence the level in the regression driver

    if perturb_type == "label_noise":
        return float(rng.choice(np.linspace(0.1, 0.5, 5)))

    if perturb_type == "input_noise":
        return float(rng.choice(np.linspace(0.1, 0.5, 5)))

    if perturb_type == "missing_data":
        return float(rng.choice(np.linspace(0.1, 0.5, 5)))

    if perturb_type == "feature_permutation":
        return float(rng.choice(np.linspace(0.1, 0.7, 7)))

    if perturb_type == "training_data_regime":
        return float(rng.choice([0.25, 0.5, 0.75, 1.0]))

    raise ValueError(f"Unknown perturbation type: {perturb_type}")


def sample_composite_perturbations(
    perturb_catalogue,
    sample_level_fn,
    n_composites=1,
    n_types=3,
    random_state=42,
):
    """Samples ``n_composites`` composites of ``n_types`` perturbations.

    ``perturb_catalogue`` is a ``{type: [variant, ...]}`` mapping
    (``perturb_dic`` or ``perturb_dic_reg``) and ``sample_level_fn(perturb_type,
    variant, rng)`` returns the magnitude for a chosen variant. Passing the
    catalogue in (rather than resolving it as a module global) is what makes
    this safe to call from either driver.
    """
    rng = np.random.default_rng(random_state)
    perturb_types = list(perturb_catalogue.keys())
    n_types = min(n_types, len(perturb_types))

    composites = []
    for _ in range(n_composites):
        chosen_types = list(
            rng.choice(perturb_types, size=n_types, replace=False)
        )
        steps = []
        for perturb_type in chosen_types:
            variant = str(rng.choice(perturb_catalogue[perturb_type]))
            level = sample_level_fn(perturb_type, variant, rng)
            steps.append(
                {
                    "type": perturb_type,
                    "variant": variant,
                    "level": level,
                }
            )
        composites.append(steps)

    return composites


def save_composites_report(composites, output_path):
    """Writes the sampled composites to ``output_path`` as JSON.

    Shared by both drivers.
    """
    report_payload = {
        "n_composites": len(composites),
        "composites": composites,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)


def parse_input_noise_variant(variant):
    """Maps an ``input_noise`` variant to ``(which_set, feature_type)``."""
    if variant.startswith("train_val_"):
        which_set = "Train_Val"
        feature_part = variant[len("train_val_") :]
    elif variant.startswith("train_"):
        which_set = "Train"
        feature_part = variant[len("train_") :]
    elif variant.startswith("val_"):
        which_set = "Val"
        feature_part = variant[len("val_") :]
    else:
        raise ValueError(f"Unsupported input_noise variant: {variant}")

    if feature_part == "continuous":
        feature_type = "cont"
    elif feature_part == "categorical":
        feature_type = "cat"
    else:
        feature_type = "cont & cat"

    return which_set, feature_type


def parse_missing_data_variant(variant):
    """Maps a ``missing_data`` variant string to ``(which_set, mechanism)``."""
    if variant.startswith("train_val_"):
        which_set = "Train_Val"
        mechanism_name = variant[len("train_val_") :]
    elif variant.startswith("train_"):
        which_set = "Train"
        mechanism_name = variant[len("train_") :]
    elif variant.startswith("val_"):
        which_set = "Val"
        mechanism_name = variant[len("val_") :]
    else:
        raise ValueError(f"Unsupported missing_data variant: {variant}")

    return which_set, mechanism_name.upper()


def parse_feature_permutation_variant(variant):
    """Maps a ``feature_permutation`` variant string to ``which_set``."""
    if variant == "train_feature_shuffle":
        return "Train"
    if variant == "val_feature_shuffle":
        return "Val"
    return "Train_Val"
