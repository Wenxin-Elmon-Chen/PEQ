import os
import json
from typing import Any, Dict, Iterable, Optional

import logging
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from copy import deepcopy
import torch
import numpy as np


def load_ground_truth_capo(args, cf_seqs: Iterable, project_root: str, logger=None) -> Dict[tuple, float]:
    """
    Load ground truth CAPO from the JSON file produced by `pipelines/compute_syn_when2stop_ground_truth.py`.

    Returns a dict keyed by `tuple(cf_seq)` with float values.
    """
    ground_truth_file = args.dataset.ground_truth_output_file
    if not os.path.isabs(ground_truth_file):
        ground_truth_file = os.path.join(project_root, ground_truth_file)

    with open(ground_truth_file, "r") as f:
        ground_truth_data = json.load(f)

    ground_truth_apo: Dict[tuple, float] = {}
    for seq_str, val in ground_truth_data["ground_truth_capo"].items():
        seq_tuple = tuple(eval(seq_str))  # stored as str(tuple/list)
        ground_truth_apo[seq_tuple] = float(val)

    if logger is not None:
        logger.info("Ground truth CAPO loaded from file:")
        for seq in cf_seqs:
            logger.info(f"  {tuple(seq)}: {ground_truth_apo[tuple(seq)]:.4f}")

    return ground_truth_apo

def load_ground_truth_capo_func(args, project_root: str, logger=None) -> Dict[tuple, float]:
    """
    Load ground truth CAPO from the JSON file produced by `pipelines/compute_syn_when2stop_ground_truth.py`.

    Returns a dict keyed by `tuple(cf_seq)` with float values.
    """
    ground_truth_file = args.dataset.ground_truth_output_file
    if not os.path.isabs(ground_truth_file):
        ground_truth_file = os.path.join(project_root, ground_truth_file)

    with open(ground_truth_file, "r") as f:
        ground_truth_data = json.load(f)

    ground_truth_apo: Dict[tuple, float] = {}
    for seq_threshold, val in ground_truth_data["ground_truth_capo"].items():
        ground_truth_apo[seq_threshold] = float(val)

    if logger is not None:
        logger.info("Ground truth CAPO loaded from file:")
        if hasattr(args.dataset, "cf_seq_thresholds_stepwise"):
            for seq_threshold in args.dataset.cf_seq_thresholds_stepwise:
                logger.info(f"  {seq_threshold}: {ground_truth_apo[str(seq_threshold[0])]:.4f}")
        else:
            for seq_threshold in args.dataset.cf_seq_thresholds:
                logger.info(f"  {seq_threshold}: {ground_truth_apo[str(seq_threshold)]:.4f}")

    return ground_truth_apo


def export_capo_results(
    *,
    model_name: str,
    dataset_name: str,
    capo_results: Dict[tuple, Dict[str, Any]],
    ground_truth_apo: Dict[tuple, float],
    cf_seqs: Iterable,
    output_dir,
    args,
    estimate_field: str,
    bias_field: str,
    results_filename: str,
    best_hparams: Optional[Dict[str, Any]] = None,
    outcome_scale: Optional[Dict[str, Any]] = None,
    logger=None,
) -> Dict[str, Any]:
    """
    Export CAPO estimation results to a JSON summary + the resolved Hydra config YAML.

    This standardizes the output format across pipelines (CRN/DeepACE/GNet).
    """
    results_summary: Dict[str, Any] = {
        "model": model_name,
        "exp_config": OmegaConf.to_container(args.exp, resolve=True) if hasattr(args, "exp") else None,
        "dataset_config": OmegaConf.to_container(args.dataset, resolve=True),
        "model_config": OmegaConf.to_container(args.model, resolve=True),
        "hyperparameter_tuning": {
            "enabled": bool(args.model.get("tune_hparams", False)),
            "best_hyperparameters": best_hparams if best_hparams else None,
            "tune_range": args.model.get("tune_range", None),
        },
        "results": {},
    }
    if outcome_scale is not None:
        results_summary["outcome_scale"] = outcome_scale

    for cf_seq in cf_seqs:
        seq_key = str(list(cf_seq))
        results_summary["results"][seq_key] = {
            "ground_truth": float(ground_truth_apo[tuple(cf_seq)]),
            estimate_field: float(capo_results[tuple(cf_seq)][estimate_field]),
            bias_field: float(capo_results[tuple(cf_seq)][bias_field]),
        }

    results_file = output_dir / results_filename
    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    config_file = output_dir / f"{model_name}_{dataset_name}_experiment_config.yaml"
    with open(config_file, "w") as f:
        OmegaConf.save(args, f)

    if logger is not None:
        logger.info(f"Results exported to {output_dir}")

    return results_summary


def export_capo_results_multi(
    *,
    model_name: str,
    dataset_name: str,
    per_sequence_results: Dict[tuple, Dict[str, Any]],
    ground_truth_apo: Dict[tuple, float],
    cf_seqs: Iterable,
    output_dir,
    args,
    results_filename: str,
    best_hparams: Optional[Dict[str, Any]] = None,
    outcome_scale: Optional[Dict[str, Any]] = None,
    logger=None,
) -> Dict[str, Any]:
    """
    Export CAPO results when each sequence has multiple metrics (e.g., gcomp + ltmle).

    `per_sequence_results[(...)]` should be a dict of JSON-serializable scalars, e.g.:
      {"gcomp_estimate": 0.1, "gcomp_bias": 0.01, "ltmle_estimate": 0.09, "ltmle_bias": 0.0}
    """
    results_summary: Dict[str, Any] = {
        "model": model_name,
        "exp_config": OmegaConf.to_container(args.exp, resolve=True) if hasattr(args, "exp") else None,
        "dataset_config": OmegaConf.to_container(args.dataset, resolve=True),
        "model_config": OmegaConf.to_container(args.model, resolve=True),
        "hyperparameter_tuning": {
            "enabled": bool(args.model.get("tune_hparams", False)),
            "best_hyperparameters": best_hparams if best_hparams else None,
            "tune_range": args.model.get("tune_range", None),
        },
        "results": {},
    }
    if outcome_scale is not None:
        results_summary["outcome_scale"] = outcome_scale

    for cf_seq in cf_seqs:
        # if cf_seq is iterable, convert to list
        if isinstance(cf_seq, Iterable):
            seq_key = str(list(cf_seq))
            seq_name = tuple(cf_seq)
            row = {"ground_truth": float(ground_truth_apo[seq_name])}
        else:
            seq_key = str(cf_seq)
            seq_name = cf_seq
            row = {"ground_truth": float(ground_truth_apo[str(cf_seq)])}
        
        # Copy and cast to float where appropriate
        for k, v in per_sequence_results[seq_name].items():
            try:
                row[k] = float(v)
            except Exception:
                row[k] = v
        results_summary["results"][seq_key] = row

    results_file = output_dir / results_filename
    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    config_file = output_dir / f"{model_name}_{dataset_name}_experiment_config.yaml"
    with open(config_file, "w") as f:
        OmegaConf.save(args, f)

    if logger is not None:
        logger.info(f"Results exported to {output_dir}")

    return results_summary

def export_capo_results_multi_stepwise(
    *,
    model_name: str,
    dataset_name: str,
    per_sequence_results: Dict[tuple, Dict[str, Any]],
    ground_truth_apo: Dict[tuple, float],
    cf_seqs: Iterable,
    output_dir,
    args,
    results_filename: str,
    best_hparams: Optional[Dict[str, Any]] = None,
    outcome_scale: Optional[Dict[str, Any]] = None,
    logger=None,
) -> Dict[str, Any]:
    """
    Export CAPO results when each sequence has multiple metrics (e.g., gcomp + ltmle).

    `per_sequence_results[(...)]` should be a dict of JSON-serializable scalars, e.g.:
      {"gcomp_estimate": 0.1, "gcomp_bias": 0.01, "ltmle_estimate": 0.09, "ltmle_bias": 0.0}
    """
    results_summary: Dict[str, Any] = {
        "model": model_name,
        "exp_config": OmegaConf.to_container(args.exp, resolve=True) if hasattr(args, "exp") else None,
        "dataset_config": OmegaConf.to_container(args.dataset, resolve=True),
        "model_config": OmegaConf.to_container(args.model, resolve=True),
        "hyperparameter_tuning": {
            "enabled": bool(args.model.get("tune_hparams", False)),
            "best_hyperparameters": best_hparams if best_hparams else None,
            "tune_range": args.model.get("tune_range", None),
        },
        "results": {},
    }
    if outcome_scale is not None:
        results_summary["outcome_scale"] = outcome_scale

    for cf_seq in cf_seqs:
        seq_name = cf_seq[0]
        row = {"ground_truth": float(ground_truth_apo[str(cf_seq[0])])}
    
        # Copy and cast to float where appropriate
        for k, v in per_sequence_results[seq_name].items():
            try:
                row[k] = float(v)
            except Exception:
                row[k] = v
        results_summary["results"][seq_name] = row

    results_file = output_dir / results_filename
    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    config_file = output_dir / f"{model_name}_{dataset_name}_experiment_config.yaml"
    with open(config_file, "w") as f:
        OmegaConf.save(args, f)

    if logger is not None:
        logger.info(f"Results exported to {output_dir}")

    return results_summary

def _setup_file_logging(output_dir: Path, filename: str) -> Path:
    """
    Write logs to `output_dir/filename` in addition to console.
    """
    log_path = output_dir / filename
    root_logger = logging.getLogger()

    # Avoid adding duplicate file handlers (e.g. if Hydra re-invokes main).
    for h in root_logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_path):
            return log_path

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    root_logger.addHandler(fh)
    return log_path


def infer_y_bounds_from_mimic_splits(*splits, y_index: int = 0):
    """Return (y_max, y_min) from MIMIC arrays shaped (n, T, d), using Y at index `y_index`."""
    arrs = [np.asarray(a) for a in splits if a is not None]
    if not arrs:
        raise ValueError("Could not infer y bounds: no non-NaN Y values found in provided splits.")
    if any(a.ndim != 3 for a in arrs):
        bad = next(a for a in arrs if a.ndim != 3)
        raise ValueError(f"Expected a 3D array (n, T, d) for MIMIC splits, got shape={bad.shape}")

    y = np.concatenate([a[..., y_index].reshape(-1) for a in arrs], axis=0)
    y = y[~np.isnan(y)]
    if y.size == 0:
        raise ValueError("Could not infer y bounds: no non-NaN Y values found in provided splits.")
    return float(y.max() + 0.001), float(y.min() - 0.001) # the small offset is to stabilize LTMLE, because the one-step LTMLE operate on the logit space.

# syn_when2stop function
def create_initiate_func(L_threshold_initiate):
    def f_initiate(L_hist, A_hist):
        # initiate treatment if any observation is below threshold
        for l in L_hist:
            if l <= L_threshold_initiate:
                return 0
        return 1
    return f_initiate

# mimic_extract function
def create_treat_func(cf_seq_threshold=0.5, T=15, lag=8):
    from scipy.special import expit
    """
    Ground-truth-like counterfactual policy with ONE knob `cf_seq_threshold`.

    This mirrors the factual assignment mechanism in `src/data/mimic_extract/simulation.py`,
    except we remove the stochastic noise term `err_A` and replace the 0.5 cutoff with
    a single tunable threshold.

    It reconstructs the latent `treat_level` *exactly like* `generate_data_factual`,
    but computed from observed history (A, Y, vitals) inside the policy:

      treat_level_0 = T/2 - 3
      treat_level update at each step s uses:
        - current A_s
        - previous A_{s-1}
        - current vitals mean
        - previous Y_{s-1}

    Then we apply a single threshold:
        A = 1{ a_contrib > cf_seq_threshold }

    The input `vitals_hist` is expected to contain the full observed covariate history
    available to the policy at time t, i.e. standardized base vitals followed by any
    raw action-affected Z variables.

    Policy signature matches the simulator:
        policy(vitals_hist, Y_hist, A_hist) -> {0,1} vector of shape (n,)
    """
    h = lag
    coef_xa = torch.tensor([((-1) ** i) * (1 / (i + 1)) for i in range(h)], dtype=float)
    coef_ya = torch.tensor([((-1) ** i) * (1 / (i + 1)) for i in range(h)], dtype=float)
    treat_level_mid = T / 2
    treat_level_max = T

    def policy(vitals_hist, Y_hist, A_hist):
        """
        vitals_hist: (n, t+1, p) includes current time t
        Y_hist:      (n, t)     raw outcomes up to t-1
        A_hist:      (n, t)     treatments up to t-1
        """

        vitals_hist = torch.as_tensor(vitals_hist)
        Y_hist = torch.as_tensor(Y_hist)
        A_hist = torch.as_tensor(A_hist)
        t = vitals_hist.shape[1] - 1  # current decision time index (t=0 is allowed)
        n = vitals_hist.shape[0]

        # --- Reconstruct treat_level(t) from scratch using A and Y (exactly as generate_data_factual) ---
        # At t=0: no past A/Y, so treat_level is just the initializer (matches generate_data_factual).
        treat_level = torch.full((n,), treat_level_mid - 3.0, dtype=float)
        if t > 0:
            for s in range(t):
                A_s = A_hist[:, s]
                vitals_s_mean = torch.mean(vitals_hist[:, s, :], axis=1)
                if s > 1:
                    delta = torch.abs(vitals_s_mean * torch.tanh(Y_hist[:, s - 1]))
                else:
                    delta = torch.abs(vitals_s_mean)
                treat_level = treat_level + (2.0 * A_s - 1.0) * delta
                if s > 0:
                    treat_level = treat_level + (2.0 * A_hist[:, s - 1] - 1.0)
                treat_level = torch.clip(treat_level, 0.0, float(treat_level_max))

        # --- Compute GT-like assignment score at time t (but without err_A) ---
        t_start = max(0, t - h)
        hist = vitals_hist[:, t_start : (t + 1), :]          # (n, window_len, p)
        hist_mean = torch.mean(hist, axis=2)                    # (n, window_len)
        # At t=0, this is an empty (n,0) array, matching generate_data_factual.
        hist_y = torch.tanh(Y_hist[:, t_start:t] / 2.0)         # (n, window_len-1)

        a_contrib = -torch.tanh(treat_level - treat_level_mid)

        # lagged hist_mean terms
        max_i_x = min(h, hist_mean.shape[1])
        for i in range(max_i_x):
            a_contrib += coef_xa[i] * hist_mean[:,  -1 - i]

        # lagged outcome-history terms
        max_i_y = min(h - 1, hist_y.shape[1])
        for i in range(max_i_y):
            a_contrib += coef_ya[i] * hist_y[:, -1 - i]

        prob_a = expit(a_contrib)

        return torch.where(prob_a > cf_seq_threshold, 1.0, 0.0)

    return policy
