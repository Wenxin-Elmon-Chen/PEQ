import sys
import os
import gc

# Force CPU-only execution for this pipeline.
# Must be set before importing torch so CUDA isn't even visible.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "runnables"))

import logging
from pathlib import Path

import numpy as np
from pytorch_lightning import seed_everything
import hydra
from omegaconf import DictConfig, OmegaConf

from pipelines.pipeline_utils import load_ground_truth_capo, _setup_file_logging
import json
from typing import Any, Dict, Iterable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_capo_results(
        model_name: str,
        capo_results: Dict[tuple, Dict[str, Any]],
        ground_truth_capo: Dict[tuple, float],
        cf_seqs: Iterable,
        output_dir,
        args,
        estimate_field: str,
        bias_field: str,
        results_filename: str,
        logger=None):
    """
    Export CAPO estimation results to a JSON summary.
    """
    
    results_summary: Dict[str, Any] = {
        "model": model_name,
        "exp_config": OmegaConf.to_container(args.exp, resolve=True) if hasattr(args, "exp") else None,
        "dataset_config": OmegaConf.to_container(args.dataset, resolve=True),
        "model_config": OmegaConf.to_container(args.model, resolve=True),
        "results": {},
    }

    for cf_seq in cf_seqs:
        seq_key = str(list(cf_seq))
        results_summary["results"][seq_key] = {
            "ground_truth": float(ground_truth_capo[tuple(cf_seq)]),
            estimate_field: float(capo_results[tuple(cf_seq)][estimate_field]),
            bias_field: float(capo_results[tuple(cf_seq)][bias_field]),
        }

    results_file = output_dir / results_filename
    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    if logger is not None:
        logger.info(f"Results exported to {output_dir}")

    return results_summary


def _run_ltmle_apo_rpy2(*, data: np.ndarray, cf_seq, method: str = "ltmle_super", v_folds: int = 3) -> float:
    """
    Call `runnables/ltmle_runner_capo.R::estimate_apo(data, a_int, method, v_folds)` via rpy2.
    Returns float APO estimate E[Y^a] for intervention `cf_seq`.
    """
    from rpy2 import robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import STAP

    data = np.asarray(data)
    
    r_script_path = os.path.join(project_root, "runnables", "ltmle_runner_capo.R")
    with open(r_script_path, "r") as f:
        r_code = f.read()

    mod = STAP(r_code, "ltmle_runner")

    a_int = np.asarray(cf_seq, dtype=float)
    if a_int.ndim != 1 or a_int.shape[0] != data.shape[1]:
        raise ValueError(f"cf_seq length must match T={data.shape[1]}, got {a_int.shape}")

    # rpy2's global activate/deactivate is deprecated; use an explicit conversion context instead.
    with localconverter(ro.default_converter + numpy2ri.converter):
        est = mod.estimate_apo(data, a_int, method=method, v_folds=int(v_folds))

    # rpy2 returns an R vector; cast to float
    try:
        return float(est[0])
    except Exception:
        return float(est)


def _make_dataset_collection(args: DictConfig, cf_seq, exp_seed: int):
    h5file_path = os.path.join(project_root, args.dataset.h5file_path)

    if args.dataset.name.startswith("mimic_extract_complex"):
        from src.data.mimic_extract_complex.dataset import MIMICSynDatasetCollection
    elif args.dataset.name.startswith("mimic_extract"):
        from src.data.mimic_extract.dataset import MIMICSynDatasetCollection
    else:
        raise ValueError(f"Invalid dataset kind: {args.dataset.name}")

    return MIMICSynDatasetCollection(
        num_patients=args.dataset.num_patients,
        config=args.dataset,
        h5file_path=h5file_path,
        cf_treatment_sequence=cf_seq,
        seed=exp_seed,
    )


def _close_mimic_data_cache(dataset_name: str, logger=None) -> None:
    if not dataset_name.startswith("mimic_extract"):
        return

    try:
        if dataset_name.startswith("mimic_extract_complex"):
            from src.data.mimic_extract_complex.dataset import MIMICDataCache
        else:
            from src.data.mimic_extract.dataset import MIMICDataCache
        MIMICDataCache.close_all()
    except Exception as e:
        if logger is not None:
            logger.warning(f"Failed to close MIMIC H5 cache cleanly: {e}")


@hydra.main(config_name="config.yaml", config_path="../config/", version_base=None)
def main(args: DictConfig):
    """
    Benchmark LTMLE (R ltmle + SuperLearner) for CAPO estimation.
    Mirrors the dataset/ground-truth/export flow used by `pipelines/train_deepace.py`.
    """
    capo_results = {}

    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    logger.info("\n" + OmegaConf.to_yaml(args, resolve=True))

    exp_seed = int(args.exp.seed)
    seed_everything(exp_seed)

    cf_seqs = [tuple(list(seq)) for seq in args.dataset.cf_seqs]

    # Output directory + file logging
    base_output_dir = Path(os.path.join(project_root, args.exp.output_dir))
    output_dir = base_output_dir / f"exp_seed={exp_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = _setup_file_logging(output_dir, f"{args.model.name}_{args.dataset.name}_seed={exp_seed}_log.txt")
    logger.info(f"Logging to {log_path}")
    # Keep rpy2 "R callback write-console:" noise visible in the terminal (console handlers),
    # but prevent it from being written to the experiment log file.
    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_path):
            class _DropRpy2ConsoleNoise(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
                    name = getattr(record, "name", "")
                    if isinstance(name, str) and name.startswith("rpy2.rinterface_lib.callbacks"):
                        return False
                    return True
            h.addFilter(_DropRpy2ConsoleNoise())
            break

    # Load ground truth CAPO per sequence
    logger.info("Loading ground truth CAPO...")
    ground_truth_apo = load_ground_truth_capo(args, cf_seqs, project_root=project_root, logger=logger)

    # Estimate CAPO for each sequence. Downstream code can difference regimens later if needed.
    method = args.model.method
    v_folds = args.model.v_folds
    if method == "iptw_only":
        logger.info("Running IPTW-only diagnostic for each counterfactual sequence (g-model only)...")
    else:
        logger.info(f"Running `{method}` for each counterfactual sequence...")

    for cf_seq in cf_seqs:
        dataset_collection = None
        data = None
        try:
            dataset_collection = _make_dataset_collection(args, cf_seq, exp_seed)
            logger.info(f"Dataset loaded for cf_seq: {cf_seq}")
            logger.info(f"Train samples: {len(dataset_collection.train_f)}")
            logger.info(f"Val samples: {len(dataset_collection.val_f)}")
            logger.info(f"Test samples: {len(dataset_collection.test_cf_treatment_seq)}")

            data = np.concatenate(
                [dataset_collection.train_f, dataset_collection.val_f],
                axis=0,
            )
            gt = float(ground_truth_apo[tuple(cf_seq)])
            est = _run_ltmle_apo_rpy2(
                data=data,
                cf_seq=cf_seq,
                method=method,
                v_folds=v_folds,
            )
            bias = float(est - gt)
            logger.info(f"cf_seq={cf_seq}: CAPO gt={gt:.6f}, {args.model.name}={est:.6f}, bias={bias:.6f}")
            capo_results[tuple(cf_seq)] = {
                f"{args.model.name}_estimate": float(est),
                f"{args.model.name}_bias": float(bias),
                "ground_truth": gt,
            }
        finally:
            if data is not None:
                del data
            if dataset_collection is not None:
                del dataset_collection
            _close_mimic_data_cache(args.dataset.name, logger=logger)
            gc.collect()

    logger.info("Exporting raw CAPO results...")
    export_capo_results(
        model_name=args.model.name,
        capo_results=capo_results,
        ground_truth_capo=ground_truth_apo,
        cf_seqs=args.dataset.cf_seqs,
        output_dir=output_dir,
        args=args,
        estimate_field=f"{args.model.name}_estimate",
        bias_field=f"{args.model.name}_bias",
        results_filename=f"{args.model.name}_{args.dataset.name}_capo_results.json",
        logger=logger,
    )

    return capo_results


if __name__ == "__main__":
    argv_rest = sys.argv[1:]
    if not any(a.startswith("+dataset=") or a.startswith("dataset=") for a in argv_rest):
        sys.argv.append("+dataset=mimic_extract_v3_tau15_numz0")
    if not any(a.startswith("+model=") or a.startswith("model=") for a in argv_rest):
        sys.argv.append("+model=gcomp")
    main()


