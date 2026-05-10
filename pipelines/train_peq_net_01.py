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
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

from src.models.peq_net import peq_net
from runnables.deepsdr_utils import MIMICDGPAdapter_CF_with_policy_embedding

from torch.utils.data import ConcatDataset
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from pipelines.pipeline_utils import load_ground_truth_capo, export_capo_results_multi, _setup_file_logging, infer_y_bounds_from_mimic_splits

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

def _combine_train_val(train_split, val_split, dataset_kind: str):
    """
    Combine factual training + validation splits for final fitting.

    Notes:
    - For `mimic_extract`, splits are numpy arrays, so we concatenate along the sample axis.
    - For torch `Dataset`s (including `ConcatDataset`), we return a `ConcatDataset`.
      If both inputs are `ConcatDataset`, we flatten to avoid nested concat datasets.
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
            model, trainer, train_loader, val_loader = train_peq_net_model(train_dataset, val_dataset, trial_args)
            
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

    # Split randomness: keep data generation fixed while training varies by experiment seed.
    exp_seed = int(args.exp.seed)
    # data_seed = int(args.dataset.seed)

    # Keep experiment randomness tied to exp_seed; dataset construction uses data_seed below.
    seed_everything(exp_seed)
    dataset_collections = {}
    for cf_seq in args.dataset.cf_seqs:
        if args.dataset.name.startswith("mimic_extract_complex"):
            from src.data.mimic_extract_complex.dataset import MIMICSynDatasetCollection

            h5file_path = os.path.join(project_root, args.dataset.h5file_path)
            dataset_collections[cf_seq] = MIMICSynDatasetCollection(
                num_patients=args.dataset.num_patients,
                config=args.dataset,
                h5file_path=h5file_path,
                cf_treatment_sequence=cf_seq,
                seed=exp_seed,
            )
        elif args.dataset.name.startswith("mimic_extract"):
            from src.data.mimic_extract.dataset import MIMICSynDatasetCollection

            h5file_path = os.path.join(project_root, args.dataset.h5file_path)
            dataset_collections[cf_seq] = MIMICSynDatasetCollection(
                num_patients=args.dataset.num_patients,
                config=args.dataset,
                h5file_path=h5file_path,
                cf_treatment_sequence=cf_seq,
                seed=exp_seed,
            )
        else:
            raise ValueError(f"Invalid dataset kind: {args.dataset.name}")

    # Log dataset information
    for cf_seq, dataset_collection in dataset_collections.items():
        logger.info(f"Dataset loaded for cf_seq: {cf_seq}")
        logger.info(f"Train samples: {len(dataset_collection.train_f)}")
        logger.info(f"Val samples: {len(dataset_collection.val_f)}")
        logger.info(f"Test samples: {len(dataset_collection.test_cf_treatment_seq)}")
    
    if args.dataset.name.startswith("mimic_extract"):
        try:
            if args.dataset.name.startswith("mimic_extract_complex"):
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
        raise ValueError(f"Unsupported dataset (expected mimic_extract*): {args.dataset.name}")

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
    ground_truth_apo = load_ground_truth_capo(args, args.dataset.cf_seqs, project_root=project_root, logger=logger)
    
    # Step 2: Generate datasets and policy embeddings
    logger.info("Generating datasets and policy embeddings...")
    train_datasets, val_datasets, eval_datasets = prepare_policy_embeddings(
        dataset_collections, args, y_max=y_max, y_min=y_min
    )

    # Combine all datasets
    combined_train_dataset = ConcatDataset(list(train_datasets.values()))
    combined_val_dataset = ConcatDataset(list(val_datasets.values()))

    # Step 2: Hyperparameter tuning (if enabled)
    best_hparams = None
    if args.model.get('tune_hparams', False):
        best_hparams = tune_hyperparameters(combined_train_dataset, combined_val_dataset, args)
    
    # Step 3: Train peq_net model
    logger.info("Training peq_net model...")
    args_tuned = deepcopy(args)
    if best_hparams:
        for k, v in best_hparams.items():
            if hasattr(args_tuned.model, k):
                setattr(args_tuned.model, k, v)
    # combine train and val datasets to make final estimates
    combined_all_dataset = _combine_train_val(combined_train_dataset, combined_val_dataset, args.dataset.name)
    model, trainer, _, _ = train_peq_net_model(combined_all_dataset, None, args_tuned)
    
    # Step 4: Estimate CATE for each cf_seq
    logger.info("Estimating CATE for each counterfactual sequence...")
    capo_results = estimate_capo_for_all_sequences(
        model, trainer, eval_datasets, args.dataset.cf_seqs, ground_truth_apo, y_max=y_max, y_min=y_min
    )
    
    # Step 5: Export results
    logger.info("Exporting results...")
    per_seq_results = {}
    for cf_seq in args.dataset.cf_seqs:
        seq_t = tuple(cf_seq)
        per_seq_results[seq_t] = {
            "gcomp_estimate": capo_results["gcomp_results"][seq_t],
            "gcomp_bias": capo_results["gcomp_bias"][seq_t],
            "ltmle_estimate": capo_results["ltmle_results"][seq_t],
            "ltmle_bias": capo_results["ltmle_bias"][seq_t],
        }

    export_capo_results_multi(
        model_name=args.model.name,
        dataset_name=args.dataset.name,
        per_sequence_results=per_seq_results,
        ground_truth_apo=ground_truth_apo,
        cf_seqs=args.dataset.cf_seqs,
        output_dir=output_dir,
        args=args,
        results_filename=f"{args.model.name}_{args.dataset.name}_capo_results.json",
        best_hparams=best_hparams,
        outcome_scale={"y_min": y_min, "y_max": y_max},
        logger=logger,
    )
    
    results.update(capo_results)
    return results


def prepare_policy_embeddings(dataset_collections, args, y_max: float, y_min: float):
    """Generate train/val datasets and no-split evaluation datasets for each policy."""
    train_datasets = {}
    val_datasets = {}
    eval_datasets = {}

    for cf_seq, dataset_collection in dataset_collections.items():
        policy_embeddings = np.vstack([np.array([cf_seq[t]]) for t in range(len(cf_seq))])
        if args.dataset.name.startswith("mimic_extract"):
            train_datasets[cf_seq] = MIMICDGPAdapter_CF_with_policy_embedding(
                dataset_collection.train_f, cf_seq, policy_embeddings, y_max, y_min
            )
            val_datasets[cf_seq] = MIMICDGPAdapter_CF_with_policy_embedding(
                dataset_collection.val_f, cf_seq, policy_embeddings, y_max, y_min
            )
            # No sample splitting: evaluate on the same factual population used for final fitting.
            eval_datasets[cf_seq] = _combine_train_val(
                train_datasets[cf_seq],
                val_datasets[cf_seq],
                args.dataset.name,
            )
        else:
            raise ValueError(f"Unsupported dataset (expected mimic_extract*): {args.dataset.name}")
    
    return train_datasets, val_datasets, eval_datasets


def train_peq_net_model(train_datasets, val_datasets, args):
    """Train the peq_net model"""
    batch_size = int(args.model.get("batch_size", 128))
    train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    val_dataloader = None
    if val_datasets is not None:
        val_dataloader = DataLoader(val_datasets, batch_size=batch_size, shuffle=False)

    # Get model configuration
    model_config = deepcopy(args.model)
    
    # Initialize peq_net model with configuration
    model = peq_net(
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
        policy_embedding_dim=model_config.policy_embedding_dim,
        policy_seq_embedding_dim=model_config.policy_seq_embedding_dim,
        use_target_network=model_config.use_target_network,
        target_polyak_tau=model_config.target_polyak_tau
    )
    
    # Train the main model
    logger.info("Training main peq_net model...")
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


def estimate_capo_for_all_sequences(model, trainer, eval_datasets, cf_seqs, ground_truth_apo, y_max: float, y_min: float):
    """Estimate CAPO for each counterfactual sequence"""
    gcomp_results = {}
    gcomp_bias = {}
    ltmle_results = {}
    ltmle_bias = {}
    
    for i, cf_seq in enumerate(cf_seqs):
        logger.info(f"Estimating CAPO for cf_seq {i+1}/{len(cf_seqs)}: {cf_seq}")
        
        eval_dataset_cf_seq = eval_datasets[cf_seq]
        eval_dataloader_cf_seq = DataLoader(eval_dataset_cf_seq, batch_size=128, shuffle=False)
        
        # G-computation estimate
        model.eps.data = torch.zeros_like(model.eps.data)
        preds = trainer.predict(model, eval_dataloader_cf_seq)
        est_gcomp, se_gcomp, pnic_gcomp, eic_gcomp = model.get_estimates_from_prediction(
            preds, eval_dataloader_cf_seq, verbose=False
        )
        
        scale = (y_max - y_min)
        est_gcomp = float(est_gcomp) * scale + y_min
        gcomp_results[tuple(cf_seq)] = est_gcomp
        gcomp_bias[tuple(cf_seq)] = est_gcomp - ground_truth_apo[tuple(cf_seq)]
        
        # LTMLE estimate
        model.solve_canonical_gradient(trainer, eval_dataloader_cf_seq, model.projection_horizon+1)
        preds = trainer.predict(model, eval_dataloader_cf_seq)
        est_ltmle, se_ltmle, pnic_ltmle, eic_ltmle = model.get_estimates_from_prediction(
            preds, eval_dataloader_cf_seq, verbose=False
        )
        
        est_ltmle = float(est_ltmle) * scale + y_min
        ltmle_results[tuple(cf_seq)] = est_ltmle
        ltmle_bias[tuple(cf_seq)] = est_ltmle - ground_truth_apo[tuple(cf_seq)]
        
        logger.info(f"Results for {cf_seq}:")
        logger.info(f"  Ground truth: {ground_truth_apo[tuple(cf_seq)]:.4f}")
        logger.info(f"  G-comp: {est_gcomp:.4f} (bias: {gcomp_bias[tuple(cf_seq)]:.4f})")
        logger.info(f"  LTMLE: {est_ltmle:.4f} (bias: {ltmle_bias[tuple(cf_seq)]:.4f})")
    
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
        sys.argv.append("+dataset=mimic_extract_complex")
    if not any(a.startswith("+model=") or a.startswith("model=") for a in argv_rest):
        sys.argv.append("+model=peq_net_01_tuned_numz5")
    main()
    