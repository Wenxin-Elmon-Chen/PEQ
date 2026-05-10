import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "runnables"))
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, ConcatDataset
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

from src.models.dltmle_correct import dltmle_correct

from runnables.deepsdr_utils import MIMICDGPAdapter_CF

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from pipelines.pipeline_utils import load_ground_truth_capo_func, export_capo_results_multi_stepwise, infer_y_bounds_from_mimic_splits, _setup_file_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)

torch.set_num_threads(12)          # intra-op
torch.set_num_interop_threads(1) 

def _infer_run_device(trainer):
    return getattr(getattr(trainer, "strategy", None), "root_device", None)


def _move_to_device(obj, device: torch.device):
    """Recursively move tensors in a nested batch (dict/list/tuple) onto `device`."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [_move_to_device(v, device) for v in obj]
        return tuple(moved) if isinstance(obj, tuple) else moved
    return obj
    
def _combine_train_val(train_split, val_split, dataset_kind: str):
    """Concatenate MIMIC numpy train/val splits along the sample axis."""
    if dataset_kind.startswith("mimic_extract"):
        return np.concatenate([train_split, val_split], axis=0)
    raise ValueError(f"Invalid dataset kind: {dataset_kind}")



def _adapt_dataset(split_data, cf_seq, y_max, y_min, dataset_kind: str):
    """
    Convert raw dataset split to the batch format expected by dltmle_correct.
    """
    if dataset_kind.startswith("mimic_extract"):
        return MIMICDGPAdapter_CF(data=split_data, cf_seq=cf_seq, Y_max=y_max, Y_min=y_min)
    raise ValueError(f"Invalid dataset kind: {dataset_kind}")


def tune_hyperparameters(train_dataset, val_dataset, args):
    """
    Hyperparameter tuning for dltmle_correct using simple random search
    """
    logger.info(f"Running simple random search with {args.model.tune_range} trials")
    
    # Convert hyperparameter grid to simple lists
    hparams_grid = {}
    for k, v in args.model.hparams_grid.items():
        if len(v) > 0:
            hparams_grid[k] = v
    
    if not hparams_grid:
        logger.warning("No hyperparameter grid specified. Skipping tuning.")
        return None
    
    logger.info(f"Tuning hyperparameters: {list(hparams_grid.keys())}")
    
    best_loss = float('inf')
    best_config = None
    
    # Simple random search without Ray
    import random
    for trial in range(args.model.tune_range):
        # Sample random hyperparameters
        trial_config = {}
        for k, v in hparams_grid.items():
            trial_config[k] = random.choice(v)
        
        logger.info(f"Trial {trial + 1}/{args.model.tune_range}: {trial_config}")
        
        # Create a copy of args and update with trial config
        trial_args = deepcopy(args)
        for k, v in trial_config.items():
            if hasattr(trial_args.model, k):
                setattr(trial_args.model, k, v)
        
        # Train and evaluate
        try:
            # Create model with trial hyperparameters
            model = dltmle_correct(
                dim_static=trial_args.dataset.dim_static,
                dim_dynamic=trial_args.dataset.dim_dynamic,
                projection_horizon=trial_args.dataset.T - 1,
                dim_treatments=trial_args.dataset.dim_treatments,
                dim_emb=trial_args.model.dim_emb,
                dim_emb_time=trial_args.model.dim_emb_time,
                dim_emb_type=trial_args.model.dim_emb_type,
                hidden_size=trial_args.model.hidden_size,
                num_layers=trial_args.model.num_layers,
                nhead=trial_args.model.nhead,
                dropout=trial_args.model.dropout,
                learning_rate=trial_args.model.learning_rate,
                alpha=trial_args.model.alpha,
                beta=trial_args.model.beta,
                outcome_type=trial_args.model.outcome_type,
                q_head=trial_args.model.q_head,
                sdr_transformation=trial_args.model.sdr_transformation,
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
            
            # Train the model
            trainer = Trainer(
                accelerator="gpu" if (len(args.exp.gpus) > 0) and (torch.cuda.is_available()) else "cpu",
                max_epochs=int(trial_args.exp.max_epochs),
                gradient_clip_val=1.0,
                logger=False,
            )
            
            trainer.fit(model, train_loader, val_loader)
            
            # Evaluate on validation set
            model.eval()
            total_loss = 0.0
            num_batches = 0
            device = _infer_run_device(trainer)
            model = model.to(device)

            with torch.no_grad():
                for batch in val_loader:
                    batch = _move_to_device(batch, device)
                    S_hat = model(**batch)
                    loss_dict = model.loss(S_hat, batch)
                    total_loss += loss_dict['Q_last'].item() + loss_dict['G'].item() # factual loss
                    num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            logger.info(f"  Trial loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_config = trial_config.copy()
                logger.info(f"  New best loss: {best_loss:.4f}")
                
        except Exception as e:
            logger.warning(f"  Trial failed: {e}")
            continue
    
    if best_config:
        logger.info(f"Best hyperparameters found: {best_config} (loss: {best_loss:.4f})")
    else:
        logger.warning("No successful trials completed.")
    
    return best_config



@hydra.main(config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):
    """
    Training / evaluation script for dltmle_correct with CATE estimation
    Args:
        args: arguments of run as DictConfig

    Returns: dict with results (estimates, standard errors, and evaluation metrics)
    """
    results = {}

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    # Split randomness: keep data generation fixed while training varies by experiment seed.
    exp_seed = int(args.exp.seed)
    # data_seed = int(args.dataset.seed)

    if not (
        args.dataset.name.startswith("mimic_extract_func_stepwise")
        or args.dataset.name.startswith("mimic_extract_complex_func_stepwise")
    ):
        raise ValueError(
            f"Unsupported dataset (expected mimic_extract_*_func_stepwise): {args.dataset.name}"
        )
    from pipelines.pipeline_utils import create_treat_func

    # Keep experiment randomness tied to exp_seed; dataset construction uses data_seed below.
    seed_everything(exp_seed)

    dataset_collections = {}
    for cf_thresholds_seq in args.dataset.cf_seq_thresholds_stepwise:
        # Instantiate dataset per CF sequence
        if args.dataset.name.startswith("mimic_extract_complex_func_stepwise"):
            from src.data.mimic_extract_complex.dataset import MIMICSynDatasetCollection
        else:
            from src.data.mimic_extract.dataset import MIMICSynDatasetCollection

        cf_seq = [create_treat_func(threshold) for threshold in cf_thresholds_seq]
        h5file_path = os.path.join(project_root, args.dataset.h5file_path)
        dataset_collections[cf_thresholds_seq] = MIMICSynDatasetCollection(
            num_patients=args.dataset.num_patients, 
            config=args.dataset,
            h5file_path=h5file_path, 
            cf_treatment_sequence=cf_seq, 
            seed=exp_seed
        )

    # Log dataset information
    for cf_seq, dataset_collection in dataset_collections.items():
        logger.info(f"Dataset loaded for cf_seq: {cf_seq}")
        logger.info(f"Train samples: {len(dataset_collection.train_f)}")
        logger.info(f"Val samples: {len(dataset_collection.val_f)}")
        logger.info(f"Test samples: {len(dataset_collection.test_cf_treatment_seq)}")
    
    # Close any open H5 handles (MIMIC uses a cached HDFStore during dataset creation).
    if args.dataset.name.startswith("mimic_extract"):
        try:
            if args.dataset.name.startswith("mimic_extract_complex_func"):
                from src.data.mimic_extract_complex.dataset import MIMICDataCache
            else:
                from src.data.mimic_extract.dataset import MIMICDataCache

            MIMICDataCache.close_all()
        except Exception as e:
            logger.warning(f"Failed to close MIMIC H5 cache cleanly: {e}")

    # Infer outcome scaling bounds from *all* generated (train/val) splits, then reuse everywhere.
    if args.dataset.name.startswith("mimic_extract"):
        _splits = []
        for _dc in dataset_collections.values():
            _splits.extend([_dc.train_f, _dc.val_f])
        y_max, y_min = infer_y_bounds_from_mimic_splits(*_splits)
        logger.info(
            f"Inferred global y-bounds from generated data: y_min={y_min:.6g}, y_max={y_max:.6g}"
        )
    else:
        raise ValueError(f"Unsupported dataset (expected mimic_extract_func): {args.dataset.name}")

    # Re-seed for experiment/training randomness (hyperparameter tuning, model init, dataloader shuffles, etc.)
    seed_everything(exp_seed)
    
    
    # Create output directory
    # use root project dir for output directory
    base_output_dir = Path(os.path.join(project_root, args.exp.output_dir))
    output_dir = base_output_dir / f"exp_seed={exp_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = _setup_file_logging(output_dir, f"{args.model.name}_{args.dataset.name}_seed={exp_seed}_log.txt")
    logger.info(f"Logging to {log_path}")
    
    # Step 1: Load ground truth CAPO for each cf_seq
    logger.info("Loading ground truth CAPO...")
    ground_truth_apo = load_ground_truth_capo_func(args, project_root=project_root, logger=logger)
    
    # Step 2: Hyperparameter tuning (if enabled)
    best_hparams = None
    if args.model.get("tune_hparams", False):
        cf_thresholds_seq = args.dataset.cf_seq_thresholds_stepwise[0] # use the first cf_seq_threshold for tuning
        cf_seq = [create_treat_func(threshold) for threshold in cf_thresholds_seq]
        
        logger.info(f"Tuning hyperparameters for cf_thresholds_seq={cf_thresholds_seq} ...")
        train_dataset_adapted = _adapt_dataset(dataset_collections[cf_thresholds_seq].train_f, cf_seq, y_max, y_min, args.dataset.name)
        val_dataset_adapted = _adapt_dataset(dataset_collections[cf_thresholds_seq].val_f, cf_seq, y_max, y_min, args.dataset.name)
        best_hparams = tune_hyperparameters(train_dataset_adapted, val_dataset_adapted, args)
    
    # Step 3: Train dltmle_correct model
    logger.info("Training a separate dltmle_correct model for each counterfactual sequence...")
    for cf_thresholds_seq in args.dataset.cf_seq_thresholds_stepwise:
        cf_seq = [create_treat_func(threshold) for threshold in cf_thresholds_seq]
        # Per-sequence args (apply tuned hyperparameters for this cf_seq only)
        seq_args = deepcopy(args)
        if best_hparams:
            for k, v in best_hparams.items():
                if hasattr(seq_args.model, k):
                    setattr(seq_args.model, k, v)

        # combine train and val datasets to make final estimates
        concat_dataset = _combine_train_val(
            dataset_collections[cf_thresholds_seq].train_f, dataset_collections[cf_thresholds_seq].val_f, args.dataset.name
        )
        concat_dataset_adapted = _adapt_dataset(concat_dataset, cf_seq, y_max, y_min, args.dataset.name)
        model, trainer = train_dltmle_correct_model_separate(concat_dataset_adapted, None, seq_args)

        # No sample splitting: evaluate on the same factual population used for final fitting.
        eval_dataset_adapted = _adapt_dataset(concat_dataset, cf_seq, y_max, y_min, args.dataset.name)
        capo_results = estimate_capo_for_one_sequence(
            model,
            trainer,
            eval_dataset_adapted,
            cf_thresholds_seq[0],
            ground_truth_apo,
            y_max=y_max,
            y_min=y_min,
        )
        results[cf_thresholds_seq[0]] = capo_results
    
    # Step 4: Export results
    logger.info("Exporting results...")
    per_seq_results = {}
    for cf_thresholds_seq in args.dataset.cf_seq_thresholds_stepwise:
        seq_name = cf_thresholds_seq[0]
        per_seq_results[seq_name] = {
            "gcomp_estimate": results[seq_name]["gcomp_results"],
            "gcomp_bias": results[seq_name]["gcomp_bias"],
            "ltmle_estimate": results[seq_name]["ltmle_results"],
            "ltmle_bias": results[seq_name]["ltmle_bias"],
        }

    export_capo_results_multi_stepwise(
        model_name=args.model.name,
        dataset_name=args.dataset.name,
        per_sequence_results=per_seq_results,
        ground_truth_apo=ground_truth_apo,
        cf_seqs=args.dataset.cf_seq_thresholds_stepwise,
        output_dir=output_dir,
        args=args,
        results_filename=f"{args.model.name}_{args.dataset.name}_capo_results.json",
        best_hparams=best_hparams,
        outcome_scale={"y_min": y_min, "y_max": y_max},
        logger=logger,
    )
    
    return results


 
def train_dltmle_correct_model_separate(train_dataset, val_dataset, args):
    """Train the dltmle_correct model"""

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = None if val_dataset is None else DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Get model configuration
    model_config = args.model
    
    # Initialize DeepSDR_policy model with configuration
    model = dltmle_correct(
        dim_static=args.dataset.dim_static,
        dim_dynamic=args.dataset.dim_dynamic,
        projection_horizon=args.dataset.T - 1,
        dim_treatments=args.dataset.dim_treatments,
        dim_emb=model_config.dim_emb,
        dim_emb_time=model_config.dim_emb_time,
        dim_emb_type=model_config.dim_emb_type,
        hidden_size=model_config.hidden_size,
        num_layers=model_config.num_layers,
        nhead=model_config.nhead,
        dropout=model_config.dropout,
        learning_rate=model_config.learning_rate,
        alpha=model_config.alpha,
        beta=model_config.beta,
        outcome_type=model_config.outcome_type,
        q_head=model_config.q_head,
        sdr_transformation=model_config.sdr_transformation
    )

    # Train the main model
    logger.info("Training a separate dltmle_correct model for each counterfactual sequence...")
    trainer = Trainer(
        accelerator="gpu" if (len(args.exp.gpus) > 0) and (torch.cuda.is_available()) else "cpu",
        max_epochs=int(args.exp.max_epochs),
        gradient_clip_val=1.0,
        logger=False,
        enable_checkpointing=False,
    )
    
    if val_dataloader is None:
        trainer.fit(model, train_dataloader)
    else:
        trainer.fit(model, train_dataloader, val_dataloader)
    
    return model, trainer


def estimate_capo_for_one_sequence(model, trainer, eval_dataset, cf_seq_threshold, ground_truth_apo, y_max: float, y_min: float):
    """Estimate CAPO for each counterfactual sequence"""
    logger.info(f"Estimating CAPO for cf_seq {cf_seq_threshold}: {cf_seq_threshold}")
    
    eval_dataloader_cf_seq = DataLoader(eval_dataset, batch_size=128, shuffle=False)
        
    # G-computation estimate
    model.eps.data = torch.zeros_like(model.eps.data)
    preds = trainer.predict(model, eval_dataloader_cf_seq)
    est_gcomp, se_gcomp, pnic_gcomp, eic_gcomp = model.get_estimates_from_prediction(
        preds, eval_dataloader_cf_seq, verbose=False
    )
    
    scale = (y_max - y_min)
    est_gcomp = float(est_gcomp) * scale + y_min
    gcomp_results = est_gcomp
    gcomp_bias = est_gcomp - ground_truth_apo[str(cf_seq_threshold)]
    
    # LTMLE estimate
    model.solve_canonical_gradient(trainer, eval_dataloader_cf_seq, model.projection_horizon+1)
    preds = trainer.predict(model, eval_dataloader_cf_seq)
    est_ltmle, se_ltmle, pnic_ltmle, eic_ltmle = model.get_estimates_from_prediction(
        preds, eval_dataloader_cf_seq, verbose=False
    )
    
    est_ltmle = float(est_ltmle) * scale + y_min
    ltmle_results = est_ltmle
    ltmle_bias = est_ltmle - ground_truth_apo[str(cf_seq_threshold)]
    
    logger.info(f"Results for {cf_seq_threshold}:")
    logger.info(f"  Ground truth: {ground_truth_apo[str(cf_seq_threshold)]:.4f}")
    logger.info(f"  G-comp: {est_gcomp:.4f} (bias: {gcomp_bias:.4f})")
    logger.info(f"  LTMLE: {est_ltmle:.4f} (bias: {ltmle_bias:.4f})")

    return {
        'gcomp_results': gcomp_results,
        'gcomp_bias': gcomp_bias,
        'ltmle_results': ltmle_results,
        'ltmle_bias': ltmle_bias
    }

if __name__ == "__main__":
    # Add default arguments only if not provided via CLI (Hydra errors on duplicates).
    argv_rest = sys.argv[1:]
    if not any(a.startswith("+dataset=") or a.startswith("dataset=") for a in argv_rest):
        sys.argv.append("+dataset=mimic_extract_complex_func_stepwise")
    if not any(a.startswith("+model=") or a.startswith("model=") for a in argv_rest):
        sys.argv.append("+model=dltmle_correct_func_tuned_numz0")
    main()
    