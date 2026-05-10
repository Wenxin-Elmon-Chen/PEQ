import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

import json
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import seed_everything

import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from pipelines.pipeline_utils import create_treat_func


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)


def compute_ground_truth_capo_mimic_extract(args: DictConfig) -> dict:
    """
    Compute ground truth CAPO for each counterfactual sequence for the MIMIC-Extract semi-synthetic generator.

    Ground truth is estimated by Monte Carlo averaging terminal outcome under the counterfactual treatment
    sequence in `dataset_collection.test_cf_treatment_seq`.
    """
    ground_truth_capo = {}
    data_seed = int(args.dataset.get("seed", args.exp.seed))

    for i, cf_thresholds_seq in enumerate(args.dataset.cf_seq_thresholds_stepwise):
        logger.info(
            f"Computing ground truth for cf_seq {i+1}/{len(args.dataset.cf_seq_thresholds_stepwise)}: {cf_thresholds_seq}"
        )
        cf_seq = [create_treat_func(threshold) for threshold in cf_thresholds_seq]

        h5file_path = os.path.join(project_root, args.dataset.h5file_path)
        config = args.dataset
        config.update(
            {
                "noise_Y": 0.0,
                "noise_A": 0.0,
            }
        )
        if "noise_Z" in config:
            config["noise_Z"] = 0

        if args.dataset.name.startswith("mimic_extract_complex_func_stepwise"):
            from src.data.mimic_extract_complex.dataset import MIMICSynDatasetCollection

            dataset_collection = MIMICSynDatasetCollection(
                num_patients={"train": 100, "val": 100, "eval": 100, "test": 25000},
                config=config,
                h5file_path=h5file_path,
                cf_treatment_sequence=cf_seq,
                seed=data_seed,
            )
        elif args.dataset.name.startswith("mimic_extract_func_stepwise"):
            from src.data.mimic_extract.dataset import MIMICSynDatasetCollection

            dataset_collection = MIMICSynDatasetCollection(
                num_patients={"train": 100, "val": 100, "eval": 100, "test": 25000},
                config=config,
                h5file_path=h5file_path,
                cf_treatment_sequence=cf_seq,
                seed=data_seed,
            )
        else:
            raise ValueError(
                f"Unsupported dataset for this script: {args.dataset.name}. "
                "Expected mimic_extract_*_func_stepwise or mimic_extract_complex_*_func_stepwise."
            )

        # test_cf_treatment_seq: shape (n, T, features) where Y is in channel 0
        y_terminal = np.asarray(dataset_collection.test_cf_treatment_seq)[:, -1, 0]
        gt = y_terminal.mean()
        ground_truth_capo[cf_thresholds_seq[0]] = gt

        logger.info(f"Ground truth CAPO for {cf_thresholds_seq}: {gt:.6f}")

    return ground_truth_capo


@hydra.main(config_name="config.yaml", config_path="../config/", version_base=None)
def main(args: DictConfig):
    """
    Compute and export ground truth CAPO for all counterfactual sequences for mimic_extract func-stepwise configs.
    """
    OmegaConf.set_struct(args, False)
    logger.info("\n" + OmegaConf.to_yaml(args, resolve=True))

    name = str(getattr(args.dataset, "name", ""))
    if not (
        name.startswith("mimic_extract_complex_func_stepwise")
        or name.startswith("mimic_extract_func_stepwise")
    ):
        raise ValueError(
            f"Unsupported dataset: {name}. "
            "Use mimic_extract_func_stepwise* or mimic_extract_complex_func_stepwise* configs."
        )

    seed_everything(int(args.dataset.get("seed", args.exp.seed)))

    ground_truth_capo = compute_ground_truth_capo_mimic_extract(args)

    ground_truth_results = {
        "dataset_config": OmegaConf.to_container(args.dataset, resolve=True),
        "ground_truth_capo": ground_truth_capo,
    }

    out_file = args.dataset.get("ground_truth_output_file", None)
    if out_file is None:
        raise ValueError("dataset.ground_truth_output_file must be set to export ground truth.")

    if not os.path.isabs(out_file):
        out_file = os.path.join(project_root, out_file)

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(ground_truth_results, f, indent=2)

    logger.info(f"Ground truth CAPO exported to {out_path}")


if __name__ == "__main__":
    main()
