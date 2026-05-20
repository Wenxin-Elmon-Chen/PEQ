PEQ-Net
============================

This repo contains runnable code and Hydra configs to reproduce the paper experiments on the
semi-synthetic **MIMIC-Extract** generator (`mimic_extract` and `mimic_extract_complex` under `src/data/`).


## Repo layout (high-level)

- `pipelines/`: experiment entrypoints, shared `pipeline_utils.py`, ground-truth scripts, and shell runners
  `pipelines.sh`, plus result aggregation via `aggregate_results.py`
- `config/`: Hydra configs (`config/dataset/`, `config/model/`, root `config/config.yaml`)
- `src/`: models (`peq_net`, `dltmle_correct`, `dltmle_correct_multiQhead`, `deepace`, …) and MIMIC semi-synthetic data code
- `runnables/`: dataset adapters and R runner `ltmle_runner_capo.R` (g-comp / LTMLE via `rpy2`)
- `data/semi-syn/MIMIC_Extract/`: place `all_hourly_data.h5` here; ground-truth JSONs are written alongside it

## Setup

### Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### R dependencies (for G-computation / LTMLE)

The G-comp and LTMLE baselines call R through `rpy2` using `runnables/ltmle_runner_capo.R`. You need:

- an R installation on your `PATH`
- R packages: `ltmle`, `SuperLearner`, `arm`, `xgboost`, `randomForest`
- Python: `rpy2` in the same venv as above

## Data

### MIMIC-III access and required input file

This project uses MIMIC-III (credentialed access via PhysioNet):
[MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/).

After access is granted, obtain the **pre-processed hourly file** described by Wang et al. (2020) and place it at:

- `data/semi-syn/MIMIC_Extract/all_hourly_data.h5`

Semi-synthetic covariates, treatments, and outcomes are generated from this file by
`src/data/mimic_extract/` and `src/data/mimic_extract_complex/` (`dataset.py`, `simulation.py`).

## IMPORTANT: generate ground truth first (must-run)

Before training or evaluation, generate every Monte Carlo ground-truth CAPO JSON file referenced by
`ground_truth_output_file` in `config/dataset/mimic_extract*.yaml`. Run the following from the **repository root**
with your venv activated.

### Full command list (all experiments in this repo)

```bash
# --- Non-stepwise: fixed counterfactual treatment sequences (keys in JSON are tuples) ---
python pipelines/compute_mimic_extract_ground_truth.py +dataset=mimic_extract
python pipelines/compute_mimic_extract_ground_truth.py +dataset=mimic_extract_complex

# --- Func-stepwise: threshold policies (one output file per dataset config) ---
python pipelines/compute_mimic_extract_func_stepwise_ground_truth.py +dataset=mimic_extract_func_stepwise
python pipelines/compute_mimic_extract_func_stepwise_ground_truth.py +dataset=mimic_extract_func_stepwise_04_05_06_all

python pipelines/compute_mimic_extract_func_stepwise_ground_truth.py +dataset=mimic_extract_complex_func_stepwise
python pipelines/compute_mimic_extract_func_stepwise_ground_truth.py +dataset=mimic_extract_complex_func_stepwise_04_05_06_all
python pipelines/compute_mimic_extract_func_stepwise_ground_truth.py +dataset=mimic_extract_complex_func_stepwise_0_05_1
python pipelines/compute_mimic_extract_func_stepwise_ground_truth.py +dataset=mimic_extract_complex_func_stepwise_0_05_1_all
python pipelines/compute_mimic_extract_func_stepwise_ground_truth.py +dataset=mimic_extract_complex_func_stepwise_0_04_05_06_1
python pipelines/compute_mimic_extract_func_stepwise_ground_truth.py +dataset=mimic_extract_complex_func_stepwise_0_04_05_06_1_all
```

Output paths are whatever each YAML sets in `ground_truth_output_file` (under `data/semi-syn/MIMIC_Extract/` by default).

## Run experiments

After ground truth exists, run experiments from the repo root (venv activated). Main bundled launcher:

```bash
bash pipelines/pipelines.sh
```

## Aggregate results

After all result JSONs have been produced under `results/`, aggregate the experiment outputs:

```bash
python3 pipelines/aggregate_results.py
```

## Experiments included (overview)

### Deterministic counterfactual sequence (two DGP × five estimators)

- **DGP**: `mimic_extract`, `mimic_extract_complex`.
- **Models / configs** (representative): `gcomp`, `ltmle`, `deepace`, `DLTMLE_correct`, **PEQ-net**.

### Dynamic policy counterfactual sequence (two DGP × two scenarios × two estimator)

- **DGP**: `mimic_extract`, `mimic_extract_complex`.
- **Models / configs**: **PEQ-net**, dltmle_correct

### Ablation Study
- **DGP**: `mimic_extract_complex`
- **Models / configs**: **PEQ-net**, dltmle_correct with finetuning, dltmle_correct with multi-Q-head 

## Acknowledgments

This repository builds on the code structure adapted from the [G-transformer](https://github.com/konstantinhess/G_transformer) and [DeepLTMLE](https://github.com/shirakawatoru/dltmle-icml-2024). We thank the authors of those projects for making their work available.