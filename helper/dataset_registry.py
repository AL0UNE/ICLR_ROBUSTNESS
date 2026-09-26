"""Task-agnostic registry for the multi-development-set benchmark layout.

Each benchmark used to train/cross-validate on MIMIC-III only and treat MIMIC-IV
(``m4``) and eICU purely as external test sets. The benchmarks now let *each* of
the three datasets take a turn as the development (cross-validation) set, with
the other two as external validation, writing results into
``<run>/<dev>/`` and ``<run>/<dev>/<test>/``.

This module holds the pieces that do not depend on the task (classification vs
regression): the dataset keys, the development/external pairing, the cohort
restrictions, and the per-test-set stratification / output-file specification.
The task-specific loaders that turn a key into actual frames live in
``m3_handling`` (classification) and ``m3_handling_reg`` (regression), which
call the helpers here.

m3 <-> m4 overlap ----------------- MIMIC-III and MIMIC-IV share no
``subject_id`` (they are independently re-anonymized: the id intersection is
empty) but they overlap in calendar time. MIMIC-IV's ``anchor_year_group``
buckets ``"2008 - 2010"`` and ``"2011 - 2013"`` cover the MIMIC-III era, so a
model trained on one and evaluated on the other would be partly trained and
tested on the same admissions. Because m3 carries no year column and no shared
id, the overlap can only be cut on the **m4 side**, by keeping
``anchor_year_group >= 2014``. This cut is applied whenever m3 and m4 are
train/test partners, in either direction; full m4 is used for m4's own
cross-validation and for the m4 <-> eICU pairing.
"""

DEV_KEYS = ["m3", "m4", "eICU"]

# MIMIC-IV anchor-year buckets that do NOT overlap the MIMIC-III era.
M4_NON_OVERLAP_YEAR_GROUPS = ["2014 - 2016", "2017 - 2019", "2020 - 2022"]

# Default missingness threshold: drop rows with more than this fraction of the
# model features missing. 
MISSINGNESS_THRESHOLD = 0.5


def external_pairs(dev_key):
    """The two external-validation test keys for a given development key."""
    if dev_key not in DEV_KEYS:
        raise KeyError(f"Unknown development dataset key: {dev_key!r}")
    return [key for key in DEV_KEYS if key != dev_key]


def needs_overlap_cut(dev_key, test_key):
    """True when the (dev, test) pair is the m3 <-> m4 temporal-overlap pair."""
    return {dev_key, test_key} == {"m3", "m4"}


def restrict_m4_overlap(frame):
    """Returns the m4 rows that do not overlap the MIMIC-III era.

    Keeps ``anchor_year_group`` in :data:`M4_NON_OVERLAP_YEAR_GROUPS`. Only call
    this on a MIMIC-IV frame (it requires the ``anchor_year_group`` column).
    """
    return frame[frame["anchor_year_group"].isin(M4_NON_OVERLAP_YEAR_GROUPS)]


def apply_missingness_filter(frame, features, threshold=MISSINGNESS_THRESHOLD):
    """Drops rows missing more than ``threshold`` of ``features``."""
    return frame[frame[features].isna().mean(axis=1) <= threshold]


# Per-test-set stratification experiments and the overall output filename. The
# stratification file stems preserve the names the benchmarks already used for
# the m4 and eICU external blocks; the m3 stems follow the same convention.
TEST_SET_SPECS = {
    "m3": {
        "overall_filename": "m3.csv",
        "stratifications": [
            ("gender", "MIMIC_3_GENDER"),
            ("age_group", "MIMIC_3_AGE"),
            ("ICU_unit", "MIMIC_3_ICU_UNIT"),
        ],
    },
    "m4": {
        "overall_filename": "m4.csv",
        "stratifications": [
            ("gender", "MIMIC_4_GENDER"),
            ("age_group", "MIMIC_4_AGE"),
            ("ICU_unit", "MIMIC_4_ICU_UNIT"),
            ("anchor_year_group", "MIMIC_4_YEAR"),
        ],
    },
    "eICU": {
        "overall_filename": "eICU.csv",
        "stratifications": [
            ("gender", "eICU_GENDER"),
            ("age_group", "MIMIC_eICU_AGE"),
            ("ICU_unit", "eICU_ICU_UNIT"),
            ("region", "eICU_region"),
        ],
    },
}
