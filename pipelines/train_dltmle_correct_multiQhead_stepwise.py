import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "runnables"))
import numpy as np
from pathlib import Path
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

from src.models.dltmle_correct_multiQhead import DLTMLE_correct_multiQhead
from runnables.deepsdr_utils import MIMICDGPAdapter_CF, MIMICDGPAdapter_CF_with_policy_embedding

from torch.utils.data import ConcatDataset
from src.models.utils_policy_embed import compute_mds_coordinates_per_timestep_mmd, create_policy_features

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from pipelines.pipeline_utils import load_ground_truth_capo_func, export_capo_results_multi_stepwise, infer_y_bounds_from_mimic_splits, _setup_file_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)

torch.set_num_threads(1)          # intra-op
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

def _combine_train_val(train_split, val_split):
    """
    Combine factual training + validation splits into one `ConcatDataset`.
    If both inputs are already `ConcatDataset`, flatten to avoid nesting.
    """
    if isinstance(train_split, ConcatDataset) and isinstance(val_split, ConcatDataset):
        return ConcatDataset(list(train_split.datasets) + list(val_split.datasets))

    return ConcatDataset([train_split, val_split])

def tune_hyperparameters(train_dataset, val_dataset, args):
    """
    Hyperparameter tuning for DeepSDR_separate using simple random search
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
            model, trainer, train_loader, val_loader = train_multiqhead_model(train_dataset, val_dataset, trial_args)
            
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
    Training / evaluation script for DeepSDR with CATE estimation
    Args:
        args: arguments of run as DictConfig

    Returns: dict with results (estimates, standard errors, and evaluation metrics)
    """
    results = {}

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    # Split randomness: exp seed for training randomness.
    exp_seed = int(args.exp.seed)

    if not (
        args.dataset.name.startswith("mimic_extract_func_stepwise")
        or args.dataset.name.startswith("mimic_extract_complex_func_stepwise")
    ):
        raise ValueError(
            f"Unsupported dataset (expected mimic_extract_*_func_stepwise): {args.dataset.name}"
        )
    from pipelines.pipeline_utils import create_treat_func

    # Initialisation of data (deterministic dataset generation)
    seed_everything(exp_seed)
    dataset_collections = {}
    for cf_thresholds_seq in args.dataset.cf_seq_thresholds_stepwise:
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
    for cf_seq_threshold, dataset_collection in dataset_collections.items():
        logger.info(f"Dataset loaded for cf_seq_threshold: {cf_seq_threshold}")
        logger.info(f"Train samples: {len(dataset_collection.train_f)}")
        logger.info(f"Val samples: {len(dataset_collection.val_f)}")
        logger.info(f"Test samples: {len(dataset_collection.test_cf_treatment_seq)}")
    
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
    _splits = []
    for _dc in dataset_collections.values():
        _splits.extend([_dc.train_f, _dc.val_f])
    y_max, y_min = infer_y_bounds_from_mimic_splits(*_splits)
    logger.info(
        f"Inferred global y-bounds from generated data: y_min={y_min:.6g}, y_max={y_max:.6g}"
    )

    # Re-seed for experiment/training randomness (hyperparameter tuning, model init, dataloader shuffles, etc.)
    seed_everything(exp_seed)
    
    # Create output directory
    # use root project dir for output directory
    base_output_dir = Path(os.path.join(project_root, args.exp.output_dir))
    output_dir = base_output_dir / f"exp_seed={exp_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = _setup_file_logging(output_dir, f"{args.model.name}_{args.dataset.name}_seed={exp_seed}_log.txt")
    logger.info(f"Logging to {log_path}")
    
    # Step 1: Load ground truth CATE for each cf_seq
    logger.info("Loading ground truth CATE...")
    ground_truth_apo = load_ground_truth_capo_func(args, project_root=project_root, logger=logger)
    
    # Step 2: Generate datasets and policy embeddings
    logger.info("Generating datasets and policy embeddings...")
    if args.dataset.name.startswith("mimic_extract"):
        train_datasets, val_datasets, eval_datasets = prepare_policy_embeddings_fake(
            dataset_collections, args, policy_func=create_treat_func, y_max=y_max, y_min=y_min
        )
    else:
        raise ValueError(f"Unsupported dataset (expected mimic_extract_func): {args.dataset.name}")

    # Combine all datasets
    combined_train_dataset = ConcatDataset(list(train_datasets.values()))
    combined_val_dataset = ConcatDataset(list(val_datasets.values()))

    # Step 2: Hyperparameter tuning (if enabled)
    best_hparams = None
    if args.model.get('tune_hparams', False):
        best_hparams = tune_hyperparameters(combined_train_dataset, combined_val_dataset, args)
    
    # Step 3: Train DLTMLE_correct_multiQhead model
    logger.info("Training DLTMLE_correct_multiQhead model...")
    args_tuned = deepcopy(args)
    if best_hparams:
        for k, v in best_hparams.items():
            if hasattr(args_tuned.model, k):
                setattr(args_tuned.model, k, v)
    # combine train and val datasets to make final estimates
    combined_all_dataset = _combine_train_val(combined_train_dataset, combined_val_dataset)
    model, trainer, _, _ = train_multiqhead_model(combined_all_dataset, None, args_tuned)
    
    # Step 4: Estimate CATE for each cf_seq
    logger.info("Estimating CATE for each counterfactual sequence...")
    capo_results = estimate_capo_for_all_sequences(
        model, trainer, eval_datasets, args.dataset.cf_seq_thresholds_stepwise, ground_truth_apo, y_max=y_max, y_min=y_min
    )
    
    # Step 5: Export results
    logger.info("Exporting results...")
    per_seq_results = {}
    for cf_thresholds_seq in args.dataset.cf_seq_thresholds_stepwise:
        per_seq_results[cf_thresholds_seq[0]] = {
            "gcomp_estimate": capo_results["gcomp_results"][cf_thresholds_seq[0]],
            "gcomp_bias": capo_results["gcomp_bias"][cf_thresholds_seq[0]],
            "ltmle_estimate": capo_results["ltmle_results"][cf_thresholds_seq[0]],
            "ltmle_bias": capo_results["ltmle_bias"][cf_thresholds_seq[0]],
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
    
    results.update(capo_results)
    return results


def prepare_policy_embeddings_fake(dataset_collections, args, policy_func, y_max: float, y_min: float):
    """Generate datasets for all counterfactual sequences, using fake policy embeddings (all 0)"""
    train_datasets = {}
    val_datasets = {}
    eval_datasets = {}

    # Stable indexing for per-policy diagnostics/logging.
    sorted_thresholds = list(dataset_collections.keys())
    policy_idx_by_id = {pid: i for i, pid in enumerate(sorted_thresholds)}

    # Generate dataset for each counterfactual sequence
    for cf_thresholds_seq, dataset_collection in dataset_collections.items():
        cf_seq = [policy_func(threshold) for threshold in cf_thresholds_seq]
        policy_embeddings = []
        for t in range(args.dataset.T):
            policy_embeddings.append(np.zeros(8))
        policy_embeddings = np.vstack(policy_embeddings)
        
        if args.dataset.name.startswith("mimic_extract"):
            train_datasets[cf_thresholds_seq] = MIMICDGPAdapter_CF_with_policy_embedding(
                dataset_collection.train_f,
                cf_seq,
                policy_embeddings,
                y_max,
                y_min,
                policy_id=cf_thresholds_seq[0],
                policy_idx=policy_idx_by_id[cf_thresholds_seq],
            )
            val_datasets[cf_thresholds_seq] = MIMICDGPAdapter_CF_with_policy_embedding(
                dataset_collection.val_f,
                cf_seq,
                policy_embeddings,
                y_max,
                y_min,
                policy_id=cf_thresholds_seq[0],
                policy_idx=policy_idx_by_id[cf_thresholds_seq],
            )
            # No sample splitting: evaluate on the same factual population used for final fitting.
            eval_datasets[cf_thresholds_seq] = _combine_train_val(
                train_datasets[cf_thresholds_seq],
                val_datasets[cf_thresholds_seq],
            )
        else:
            raise ValueError(f"Unsupported dataset (expected mimic_extract_func): {args.dataset.name}")
    
    return train_datasets, val_datasets, eval_datasets


def train_multiqhead_model(train_datasets, val_datasets, args):
    """Train the DLTMLE_correct_multiQhead model"""
    # # Combine all datasets
    # combined_train_dataset = ConcatDataset(list(train_datasets.values()))
    # combined_val_dataset = ConcatDataset(list(val_datasets.values()))
    
    train_dataloader = DataLoader(train_datasets, batch_size=128, shuffle=True)
    val_dataloader = None
    if val_datasets is not None:
        val_dataloader = DataLoader(val_datasets, batch_size=128, shuffle=False)

    # Get model configuration
    model_config = deepcopy(args.model)
    
    # Initialize DLTMLE_correct_multiQhead model with configuration
    num_q_heads = int(model_config.get("num_q_heads", len(args.dataset.cf_seq_thresholds_stepwise)))
    model = DLTMLE_correct_multiQhead(
        dim_static=args.dataset.dim_static,
        dim_dynamic=args.dataset.dim_dynamic,
        projection_horizon=args.dataset.T - 1,
        num_q_heads=num_q_heads,
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
        sdr_transformation=model_config.sdr_transformation,
    )
    
    # # Pretrain seqAE with policy embeddings
    # logger.info("Pretraining seqAE...")
    # _ = model.pretrain_seqAE(
    #     train_loader=cf_seq_train_loader,
    #     num_epochs=model_config.seqAE_epochs,
    #     learning_rate=model_config.seqAE_learning_rate,
    #     patience=model_config.seqAE_patience,
    #     device='cpu',
    #     verbose=True
    # )
    
    # Train the main model
    logger.info("Training main DLTMLE_correct_multiQhead model...")
    trainer = Trainer(
        accelerator="gpu" if (len(args.exp.gpus) > 0) and (torch.cuda.is_available()) else 'cpu', 
        max_epochs=int(args.exp.max_epochs),
        gradient_clip_val=1.0,
        logger=False,
        enable_checkpointing=False,
    )
    
    if val_dataloader is None:
        trainer.fit(model, train_dataloaders=train_dataloader)
    else:
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    return model, trainer, train_dataloader, val_dataloader


def estimate_capo_for_all_sequences(model, trainer, eval_datasets, cf_seq_thresholds_stepwise, ground_truth_apo, y_max: float, y_min: float):
    """Estimate CAPO for each counterfactual sequence"""
    gcomp_results = {}
    gcomp_bias = {}
    ltmle_results = {}
    ltmle_bias = {}
    
    for cf_thresholds_seq in cf_seq_thresholds_stepwise:
        logger.info(f"Estimating CAPO for cf_seq_threshold {cf_thresholds_seq[0]}: {cf_thresholds_seq}")
        
        eval_dataset_cf_seq = eval_datasets[cf_thresholds_seq]
        eval_dataloader_cf_seq = DataLoader(eval_dataset_cf_seq, batch_size=128, shuffle=False)
        
        # G-computation estimate
        model.eps.data = torch.zeros_like(model.eps.data)
        preds = trainer.predict(model, eval_dataloader_cf_seq)
        est_gcomp, se_gcomp, pnic_gcomp, eic_gcomp = model.get_estimates_from_prediction(
            preds, eval_dataloader_cf_seq, verbose=False
        )
        
        scale = (y_max - y_min)
        est_gcomp = float(est_gcomp) * scale + y_min
        gcomp_results[cf_thresholds_seq[0]] = est_gcomp
        gcomp_bias[cf_thresholds_seq[0]] = est_gcomp - ground_truth_apo[str(cf_thresholds_seq[0])]
        
        # LTMLE estimate
        model.solve_canonical_gradient(trainer, eval_dataloader_cf_seq, model.projection_horizon+1)
        preds = trainer.predict(model, eval_dataloader_cf_seq)
        est_ltmle, se_ltmle, pnic_ltmle, eic_ltmle = model.get_estimates_from_prediction(
            preds, eval_dataloader_cf_seq, verbose=False
        )
        
        est_ltmle = float(est_ltmle) * scale + y_min
        ltmle_results[cf_thresholds_seq[0]] = est_ltmle
        ltmle_bias[cf_thresholds_seq[0]] = est_ltmle - ground_truth_apo[str(cf_thresholds_seq[0])]
        
        logger.info(f"Results for {cf_thresholds_seq[0]}:")
        logger.info(f"  Ground truth: {ground_truth_apo[str(cf_thresholds_seq[0])]:.4f}")
        logger.info(f"  G-comp: {est_gcomp:.4f} (bias: {gcomp_bias[cf_thresholds_seq[0]]:.4f})")
        logger.info(f"  LTMLE: {est_ltmle:.4f} (bias: {ltmle_bias[cf_thresholds_seq[0]]:.4f})")
    
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
        sys.argv.append("+model=dltmle_correct_multiQhead_func_numz5")
    main()
    