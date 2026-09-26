# Robustness Benchmark

Code for *Stress-testing prediction models: A systematic robustness evaluation
on large-scale ICU data*.

The benchmark stress-tests tabular machine-learning models under clinically
motivated data perturbations and distribution shifts, on two ICU prediction
tasks:

- **Classification** — in-hospital mortality (`hospital_mortality`).
- **Regression** — ICU length of stay in days (`icu_los`).

Three large-scale intensive-care databases are used: MIMIC-III, MIMIC-IV and
eICU. Each of the three serves in turn as the **development** set
(cross-validated), with the other two used as **external** validation sets, so
every dataset is evaluated both as a development and as an external dataset.

Four model families are compared: linear/regularised baselines, tree ensembles
and gradient boosting, a neural baseline (RealMLP), and tabular foundation
models (TabPFNv3, TabICLv2, TabDPT). Perturbation families cover label noise,
input noise, missing data, feature shuffling, class imbalance, training-set
size, subgroup analysis, external validation, and composite (multi-perturbation)
scenarios.

## Repository Layout

The four benchmark entry points live at the project root; all supporting modules
live in the `helper/` package.

| Path | Purpose |
| --- | --- |
| `mimic_benchmark.py` | Single-perturbation classification benchmark (hospital mortality). |
| `mimic_benchmark_reg.py` | Single-perturbation regression benchmark (ICU length of stay). |
| `multiperturbation_benchmark.py` | Classification benchmark on randomly sampled composite perturbations, plus a paired clean baseline. |
| `multiperturbation_benchmark_reg.py` | Regression benchmark on composite perturbations, plus a paired clean baseline. |
| `config/config.yaml` | The single benchmark configuration (parallelism, tuning, devices, model selection, composites, mask cache). |
| `data/` | Input CSVs (not committed) and the SQL cohort-extraction queries `MIMIC_III.sql` / `MIMIC_IV.sql`. |
| `preset/` | Where optional HPO preset repositories are expected (see `preset_path` / `preset_path_reg`). Empty by default. |
| `result/` | Default output root: one timestamped directory per run. Empty by default. |
| `plotting/` | Placeholder for figure scripts. Empty by default. |
| `helper/perturbation_function.py` | Classification perturbation and evaluation functions. |
| `helper/perturbation_function_reg.py` | Regression perturbation and evaluation functions. |
| `helper/perturbations.py` | Task-agnostic perturbation primitives shared by both tasks and both drivers. |
| `helper/composite_perturbations.py` | Composite-benchmark building blocks: perturbation catalogues, recipe sampler, level samplers, variant parsers. |
| `helper/missing_data_mechanism.py` | MCAR, MAR and MNAR missingness mechanisms, with an optional mask cache. |
| `helper/model_factory.py` | Model-dictionary and device-aware HPO-grid factories. |
| `helper/model_selection.py` | Config-driven selection of which models a benchmark runs. |
| `helper/model_helpers.py` | Classification tuning, preprocessing, calibration, preset loading and calibration metrics. |
| `helper/model_helpers_reg.py` | Regression tuning, batched prediction and metrics. |
| `helper/hpo_grid.py` / `helper/hpo_grid_reg.py` | Classification / regression hyperparameter search spaces. |
| `helper/bench_config.py` | Typed configuration loader for `config/config.yaml`. |
| `helper/run_helpers.py` | Shared parallel run loops (CV, subgroup, external) and result saving. |
| `helper/dataset_registry.py` | Development/external dataset pairing, the MIMIC-III ↔ MIMIC-IV overlap and missingness restrictions, and per-test-set stratification specs. |
| `helper/m3_handling.py` | Classification dataset loading (per-dataset bundles + external-frame builder) and feature definitions. |
| `helper/m3_handling_reg.py` | Regression dataset loading and ICU length-of-stay target construction. |
| `helper/helper.py` | Shared utilities: seeding, batched prediction, thread limits, logging, result saving. |


## Data

The benchmarks expect three CSV files in `data/`:

- `data/m3.csv` — MIMIC-III (v1.4).
- `data/m4.csv` — MIMIC-IV (v3.1).
- `data/eICU.csv` — eICU Collaborative Research Database.

The paths can be overridden with the `M3_CSV_PATH`, `M4_CSV_PATH` and
`EICU_CSV_PATH` environment variables. All of `data/` is gitignored: these are
restricted-access clinical datasets and must not be committed.

`data/MIMIC_III.sql` and `data/MIMIC_IV.sql` are the cohort-extraction queries
that produce `m3.csv` and `m4.csv`. They follow the SAPS II definition (Le Gall
et al., 1993), computed on the first day of each ICU stay.

### Features and targets

Predictors are informed by the variables in SAPS II:

- **Continuous (22)**: `age`; heart rate, systolic blood pressure and
  temperature (min/max); `urineoutput`; BUN, WBC, potassium, sodium,
  bicarbonate and bilirubin (min/max); `mingcs`; `pao2fio2_vent_min`.
- **Categorical (4)**: `aids`, `hem`, `mets`, `admissiontype`.

Targets and auxiliary columns:

- **Classification target**: `hospital_mortality`.
- **Proxy outcomes** for the proxy label-noise scenario: MIMIC-III and MIMIC-IV
  carry both `death_icu` and `death_overall`; eICU carries only `death_icu`. The
  proxy sweep runs each development set with the variants its columns support —
  three for m3/m4 (`hospital_death`, `icu_death`, `overall_death`), two for
  eICU.
- **Regression target**: `icu_los` in days. For MIMIC it is derived from
  `intime` and `outtime`; for eICU the CSV must already contain an `icu_los`
  column. Rows whose target cannot be computed are dropped.
- **Stratification columns**: `gender`, `age_group`, `ICU_unit` for every
  dataset, plus `anchor_year_group` for MIMIC-IV and `region` for eICU.

Columns are read per dataset with an explicit `usecols` list, so a missing
column fails loudly at load time.

### Development sets and the MIMIC-III ↔ MIMIC-IV overlap

Every benchmark evaluates **each** of the three datasets as the development
(cross-validation) set in turn, with the other two as external validation
(`helper/dataset_registry.py`). MIMIC-III and MIMIC-IV share no `subject_id`
(they are independently re-anonymised) but they **overlap in calendar time**:
MIMIC-IV's `anchor_year_group` buckets `2008 - 2010` and `2011 - 2013` cover the
MIMIC-III era. To avoid training and testing on the same admissions, whenever
MIMIC-III and MIMIC-IV are train/test partners — in either direction — the
**MIMIC-IV side is restricted to `anchor_year_group >= 2014`**. The cut can only
be made on the MIMIC-IV side, because MIMIC-III carries no year column and no
shared id. Full MIMIC-IV is used for its own cross-validation and for the
MIMIC-IV ↔ eICU pairing.

External test frames additionally pass through a defensive missingness filter:
rows with more than 50% of the model features missing are dropped.

## Installation

Python `>=3.12` with [uv](https://docs.astral.sh/uv/). The lockfile is
`uv.lock`; PyTorch is selected through mutually exclusive extras:

```bash
uv sync --extra cpu     # CPU-only PyTorch wheels
uv sync --extra cu130   # CUDA 13.0 PyTorch wheels
```

Runtime dependencies (`pyproject.toml`): `scikit-learn==1.8.0`,
`xgboost==3.2.0`, `lightgbm==4.6.0`, `catboost==1.2.10`, `tabicl==2.1.1`,
`tabpfn>=8.0.8`, `tabdpt>=1.2.0`, `pytabkit>=1.5.0`, `pyyaml>=6.0`, plus
`torch`/`torchvision` from the extras above.

`pytabkit` (RealMLP) is imported defensively: if it is absent, "RealMLP" stays
in the model lineup but raises an actionable `ImportError` at fit time rather
than being silently dropped.

## Hardware

The shipped `config/config.yaml` runs on CPU: `xgboost_device: cpu`,
`lightgbm_device: cpu`, `realmlp_device: cpu`. Switching to
`xgboost_device: cuda` / `lightgbm_device: gpu` / `realmlp_device: cuda` needs
no source edits.

One caveat with `hpo: true`: `helper/model_factory.build_param_grids` injects
the device into a search space only if that space already declares a `device`
key. `helper/hpo_grid_reg.py` does; `helper/hpo_grid.py` does not. Since the
tuned XGBoost, LightGBM and CatBoost classifiers are rebuilt from the search
space alone, a tuned **classification** run puts them on CPU whatever these keys
say. Untuned classification runs and all regression runs honour the config.

The tabular foundation models (TabPFNv3, TabICLv2, TabDPT) are **not**
configurable: `model_factory.FOUNDATION_DEVICE` pins them to `cuda`. They need a
visible GPU and can use substantial GPU memory depending on dataset size. TabICL
downloads its checkpoint from the Hugging Face Hub on first use; set `HF_HOME` to
control where it is cached.

The paper's runs used an Intel Core Ultra 9 285K CPU with an NVIDIA RTX 4090 and
64 GB of RAM.

## Running Benchmarks

Run from the repository root.

```bash
uv run python mimic_benchmark.py                 # classification
uv run python mimic_benchmark_reg.py             # regression
uv run python multiperturbation_benchmark.py     # composite classification
uv run python multiperturbation_benchmark_reg.py # composite regression
```

These are experiment entry points: they execute the configured benchmark
immediately when run. There is no command-line interface — every tunable setting
lives in `config/config.yaml` (see [Configuration](#configuration)) or in the
environment variables listed below.

Full runs are expensive: each single-perturbation benchmark trains every selected
model across `n_folds` folds (10 by default) for every level of every scenario,
for all three development sets. For a quick pass, restrict the `models` block
and/or lower `n_folds`.

## Benchmark Coverage

### Single-perturbation benchmarks

Both `mimic_benchmark.py` and `mimic_benchmark_reg.py` run, per development set,
a cross-validated suite followed by external validation on the two other
datasets. Levels are the sweep values passed to each scenario.

| Scenario | Output folder | Files | Levels | Task |
| --- | --- | --- | --- | --- |
| Random label noise | `label_noise/` | `RANDOM_LABEL_NOISE` | 0.0 … 1.0 (step 0.1) | both |
| Targeted label noise (0→1, 1→0) | `label_noise/` | `01_LABEL_NOISE`, `10_LABEL_NOISE` | 0.0 … 0.9 | classification |
| Age-conditional label noise | `label_noise/` | `AGE_LABEL_NOISE` | 0.0 … 1.0 | both |
| Proxy-outcome label noise | `label_noise/` | `PROXY_LABEL_NOISE` | per-dataset variants | classification |
| Input (measurement) noise | `input_noise/` | `INPUT_NOISE_{TRAIN,VAL,ALL}` and their `_CONT` / `_CAT` variants | 0.0 … 1.0 | both |
| Class imbalance | `imbalance_data/` | `IMBALANCED_DATA` | 1.0 … 0.1 | classification |
| Training-set size | `training_size/` | `TRAINING_SIZE` | 0.05, 0.1, 0.25, 0.5, 0.8, 1.0 | both |
| Feature shuffling | `feature_shuffle/` | `SHUFFLED_{TRAIN,VAL,ALL}_DATA` | 0.0 … 1.0 | both |
| Missing data | `missing_data/` | `{MCAR,MAR,MNAR}_{TRAIN,VAL,ALL}` | 0.1 … 0.9 | both |
| Subgroup analysis | `subgroups/` | `SUBGROUP_{GENDER,AGEGROUP,ICU_UNIT}` | — | both |
| External validation | `<test_key>/` | per-stratification files + an overall file | — | both |

Notes on the mechanisms:

- **Label noise (classification)** flips training labels: at random, from 0→1
  only, from 1→0 only, with a flip probability scaled by the patient's age
  percentile (`conditional`), or by replacing the target with a proxy outcome.
- **Label noise (regression)** adds squared-Gaussian noise to the continuous
  target, scaled by `level × 0.5 × std(y)`; the `conditional` variant scales the
  amplitude by the age percentile.
- **Input noise** adds Gaussian noise of scale `level × 2 × std` to continuous
  features and resamples categorical features from their marginal distribution
  with probability `level`. `which_set` selects train, validation or both;
  `feature_type` selects continuous, categorical or both.
- **Class imbalance** downsamples the **positive (minority)** class of the
  training set to the given fraction.
- **Feature shuffling** independently permutes a proportion of the columns. For
  the `Train_Val` variant the same columns are shuffled in both sets.
- **Missing data** completes the matrix (continuous → mean, categorical → mode),
  applies the MCAR/MAR/MNAR mechanism, then re-applies the resulting mask as
  `NaN`. For `Train_Val` the two sets are masked jointly and split back by row
  label. The mechanisms follow Muzellec et al.; the implementation is adapted
  from [MissingDataOT](https://github.com/BorisMuzellec/MissingDataOT).
- **Subgroup analysis** cross-validates on the development set and reports
  metrics per stratum of `gender`, `age_group` and `ICU_unit`.
- **External validation** trains on the full development set and scores on the
  other dataset, both overall and stratified. Stratification columns are
  per-test-set (`helper/dataset_registry.TEST_SET_SPECS`): MIMIC-IV adds
  `anchor_year_group` (temporal shift) and eICU adds `region` (geographic
  shift).

The regression suite is the same minus class imbalance, targeted label noise and
proxy-outcome label noise, all of which are undefined for a continuous target.

### Composite benchmarks

The composite drivers sample `n_composites` recipes, each a sequence of
`n_perturb_types_per_composite` perturbations of **distinct** types applied
sequentially to the same fold. The sampled recipes are written to
`composite_combinations.json` at the root of the run directory, and the same
recipes are reused across development sets.

Catalogues live in `helper/composite_perturbations.py`. The classification
catalogue covers label noise, input noise, missing data, feature permutation,
class imbalance and training-set size; the regression catalogue drops class
imbalance. `multiperturbation_benchmark.py` additionally filters the `proxy_*`
label-noise variants out of its catalogue, so every recipe can run on all three
development sets (eICU has no `death_overall` column).

Composite magnitudes are sampled per type: 0.1–0.5 for label noise, input noise
and missing data; 0.1–0.7 for feature permutation; and {0.25, 0.5, 0.75, 1.0}
for class imbalance and training-set size.

Both composite drivers also run an unperturbed **clean baseline** (an empty
recipe) per development set — the paired reference for robustness comparisons.
Composite results are saved incrementally, so a crash mid-run still leaves every
completed recipe on disk.

## Models

| Family | Classification | Regression |
| --- | --- | --- |
| Linear / regularised | `Logistic`, `LASSO`, `Ridge` | `Linear`, `LASSO`, `Ridge` |
| Trees and boosting | `Random Forest`, `Gradient Boosting`, `XGBoost`, `LightGBM`, `CatBoost` | same |
| Neural | `RealMLP` (pytabkit) | same |
| Tabular foundation | `TabPFNv3`, `TabICLv2`, `TabDPT` | same |

**Preprocessing** (`helper/model_helpers.build_preprocessors`, shared by both
tasks): continuous features are mean-imputed and categorical features
most-frequent-imputed, each with a binary missingness indicator added. Linear
models use the scaled preprocessor (`StandardScaler` on the continuous block);
tree, boosting and neural models use the unscaled one. Foundation models receive
the raw frame and rely on their own missing-value handling.

**Tuning** (`hpo: true`): nested cross-validation — the `n_folds` outer folds
(10 by default) with a 3-fold inner loop, and a budget of `n_search_gs`
(default 15) sampled configurations per model. Search spaces are in
`helper/hpo_grid.py` and `helper/hpo_grid_reg.py`. Non-boosting models go
through `RandomizedSearchCV` (scored by AUC for
classification, negative MSE for regression); XGBoost, LightGBM and CatBoost use
a manual inner loop so early stopping can be used. TabPFNv3, TabICLv2 and TabDPT
have no search space and always run at their library defaults.

**Calibration** (classification only): predicted probabilities are Platt-scaled
on a dedicated holdout of 10% of the outer validation fold; the remaining 90% is
what gets timed and scored. The composite driver fits the same sigmoid directly
from one set of pre-computed scores instead of going through
`CalibratedClassifierCV`, which avoids five redundant `predict_proba` calls per
fold — a real cost for the in-context foundation models.

## Outputs

Each run creates a timestamped directory under `result/`, holding a
`backup_config.yaml` snapshot of the configuration it ran with:

- `result/res_YYYYMMDD_HHMMSS` — classification.
- `result/reg_res_YYYYMMDD_HHMMSS` — regression.
- `result/res_multi_YYYYMMDD_HHMMSS` — classification composites.
- `result/reg_res_multi_YYYYMMDD_HHMMSS` — regression composites.

Set `BENCHMARK_OUTPUT_BASE_DIR` to write elsewhere, or
`BENCHMARK_RESULTS_DIR_NAME` to replace the timestamped directory name.

Result CSVs are organised first by **development set** (`m3`, `m4`, `eICU`),
then by scenario. Each development set holds its cross-validated suite plus one
external-validation subfolder per other dataset:

```text
res_YYYYMMDD_HHMMSS/
  backup_config.yaml
  m3/                      # MIMIC-III as the development set
    label_noise/ input_noise/ imbalance_data/ training_size/
    feature_shuffle/ missing_data/ subgroups/
    m4/                    # train m3, test MIMIC-IV (anchor_year_group >= 2014)
    eICU/                  # train m3, test eICU
  m4/                      # MIMIC-IV as the development set
    label_noise/ ... subgroups/
    m3/                    # train m4 (anchor_year_group >= 2014), test MIMIC-III
    eICU/
  eICU/                    # eICU as the development set
    label_noise/ ... subgroups/
    m3/
    m4/
```

The composite runs write, per development set, a `clean/CLEAN_BASELINE.csv` and
a `multi_perturbation/MULTI_PERTURBATION_COMPOSITE.csv`, plus
`composite_combinations.json` at the top of the run directory. They have no
external-validation block.

Columns per row (one row per model × level × fold):

- **Classification**: `Model`, `Noise level`, `AUC`, `Brier score`, `Intercept`,
  `Slope` (calibration intercept and slope), `Train fit time`, `Test pred time`,
  `Best param`, `HPO`.
- **Regression**: `Model`, `Noise level`, `MAE`, `RMSE`, `R2`, `Train fit time`,
  `Test pred time`, `Best param`, `HPO`.
- **Subgroup and stratified external** files replace `Noise level` with
  `Variable` and `Category`.
- **Overall external** files (`m3.csv`, `m4.csv`, `eICU.csv`) carry neither, and
  no `Best params` column.

`HPO` records whether a search actually ran for that row, so preset-driven and
default-parameter rows stay distinguishable.

Regression filenames carry the `n_training_sample` suffix (`TRAINING_SIZE_1.csv`
with the default `n_training_sample: 1`). The overall external-validation files
are written without it.

Failure logs (`boosting_failures_*.log`, and `calibration_failures_*.log` for
classification) are created in the run directory only if something fails; each
line is a JSON record with the timestamp, the parameters and the error.

## Configuration

All settings live in `config/config.yaml` and are loaded by
`bench_config.load_config`. The single `defaults` block applies identically to
every benchmark (`classification`, `classification_composite`, `regression`,
`regression_composite`); per-benchmark overrides are deliberately unsupported,
so the four experiments always run under the same settings and their results
stay comparable — the loader raises if a `benchmarks` block is added. Unknown
keys are also rejected.

Point `BENCHMARK_CONFIG` at another file to use a different configuration.

Key settings:

- `random_state`: global reproducibility seed. It drives the CV folds, the
  tuning seeds **and** the perturbation noise realisations, so changing this one
  value re-randomises an entire run reproducibly.
- `n_folds`: number of cross-validation folds (default 10). Classification uses
  `StratifiedKFold` on the mortality outcome; regression uses `KFold`.
- `n_jobs`, `n_jobs_train`, `n_jobs_predict`, `n_jobs_gs`: parallelism. `n_jobs`
  is the outer (model × fold) pool; `n_jobs_train` bounds the threads inside
  each fit. Keep `n_jobs × n_jobs_train` at or below the physical core count.
- `split_train_predict`: score each fold with no training in flight, so
  `Test pred time` is not contended by concurrent fitting. The runner trains
  then scores one fold at a time, so peak memory is one fold's models rather
  than `n_models × n_folds`. Leave it on whenever prediction timings matter.
- `train_across_folds`, `fold_n_jobs`: opt into training every (model, fold)
  pair for a level concurrently instead of one fold at a time. Only meaningful
  with `split_train_predict: true`, and it trades the memory guarantee above for
  throughput. Currently read by the classification driver only.
- `hpo`: enable hyperparameter optimisation. Off by default — see the default
  regime below.
- `n_search_gs`: number of sampled hyperparameter configurations (default 15).
- `no_inner_fold`: use a single 67/33 inner split instead of the 3-fold inner
  CV during tuning.
- `xgboost_device` (`cuda`/`cpu`), `lightgbm_device` (`gpu`/`cpu`),
  `realmlp_device` (`cuda`/`cpu`): applied to the model definitions, and to the
  HPO search spaces for regression only (see Hardware above). The foundation
  models are not covered by these keys.
- `models`: which models to run, as a mapping of model name to `true`/`false`.
  Switch a model off by setting it to `false` **or** by commenting out its line;
  deleting the whole block (or leaving it empty) runs every model. The linear
  model is `Logistic` for classification and `Linear` for regression; the other
  names are shared. Names that do not apply to the current benchmark are
  ignored, so one block covers all four. An unknown name raises (typo guard), as
  does a block that leaves no model enabled.
- `preset_enabled`, `preset_strict`, `preset_path`, `preset_path_reg`,
  `preset_top_k`: optional tuned-once preset repository of best hyperparameters,
  replayed from a previous HPO run instead of re-tuning. `preset_path` is used
  by the classification benchmark (candidates ranked by AUC) and
  `preset_path_reg` by the regression one (ranked by RMSE); both are relative to
  the repo root. The composite benchmarks do not use presets.
- `n_composites`, `n_perturb_types_per_composite`: composite sampling.
- `composite_start`: 1-based index of the first composite to train, to resume a
  partially completed composite run. The full recipe set is still sampled and
  reported, so the recipes stay reproducible; only the earlier ones are skipped.
- `mask_cache_enabled`, `mask_cache_size`: reuse the model-independent MAR/MNAR
  masks across models. Disable for multi-threaded execution.
- `batch_size`, `n_training_sample`: regression-only prediction batch size
  (`null` → 100000) and the index appended to regression output filenames.

The committed `config/config.yaml` enables `Logistic`, `LASSO`, `Ridge`,
`Random Forest`, `Gradient Boosting`, `XGBoost` and `CatBoost`, and leaves
`LightGBM`, `RealMLP` and the three foundation models off. Enable them to
reproduce the full paper lineup.

### Default regime

- **Sequential**: every `n_jobs*` defaults to `1`. With the foundation models
  enabled, process-level fan-out risks GPU contention and memory blow-up; on a
  CPU-only or many-core host, raise `n_jobs` first — the threadpool plumbing is
  already wired.
- **HPO off**: models are compared at library-default hyperparameters,
  identically across all four benchmarks. Turn it on in `config.yaml`, or supply
  a tuned-once preset via the `preset_*` keys. This is a protocol choice, not a
  speed knob.

### Environment variables

```bash
export BENCHMARK_CONFIG=/path/to/config.yaml      # alternative config file
export BENCHMARK_OUTPUT_BASE_DIR=/path/to/outputs # default: result/
export BENCHMARK_RESULTS_DIR_NAME=custom_run_name # replaces res_<timestamp>
export M3_CSV_PATH=/path/to/m3.csv                # also M4_CSV_PATH, EICU_CSV_PATH
export PRESET_RESULTS_PATH=/path/to/hpo_results   # overrides preset_path
export PRESET_RESULTS_PATH_REG=/path/to/hpo_reg   # overrides preset_path_reg
```

Composite drivers only — the single-perturbation benchmarks always loop over all
three development sets:

```bash
export BENCHMARK_DEV_KEY=m3          # run one development set (m3 | m4 | eICU)
export BENCHMARK_DEV_KEYS=m3,m4      # regression composite only: restrict the set
export BENCHMARK_BASELINE_ONLY=1     # regression composite only: clean baseline, no composites
```

The `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` variables
are set to `1` by each entry script before importing NumPy, unless already
present in the environment. Set them yourself to override.

## Submission Package Notes

For anonymous review, include the source files, `README.md`, `pyproject.toml`
and `uv.lock`. Do not include local caches, raw clinical CSV files, generated
benchmark outputs, `.vscode/`, `__pycache__/` or other machine-specific files.

MIMIC-III, MIMIC-IV and eICU are restricted-access datasets under a data use
agreement and cannot be redistributed. The SQL cohort-extraction queries in
`data/` and the feature list above document the required schema; reviewers
should be told which results depend on the restricted data and which parts of
the code can be inspected or run independently.
