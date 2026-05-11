import sys
import os

# Force CPU-only execution for this pipeline.
# Must be set before importing torch / pytorch_lightning so CUDA isn't even visible.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
from torch.utils.data import TensorDataset, ConcatDataset
from pytorch_lightning import seed_everything
import pytorch_lightning as pl

# DeepACE specific imports
from src.models.deepace import DeepACE
from runnables.deepace_utils import DeepACEDataAdapter_MIMICExtract

import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pipelines.pipeline_utils import load_ground_truth_capo, export_capo_results, _setup_file_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)

torch.set_num_threads(10)          # intra-op
torch.set_num_interop_threads(1) 

def _combine_train_val(train_split, val_split, dataset_kind: str):
    """
    Combine factual training + validation splits for final fitting.

    Notes:
    - Splits are numpy arrays (`mimic_extract*`), so we concatenate along the sample axis.
    """
    if dataset_kind.startswith("mimic_extract"):
        import numpy as np
        return np.concatenate([train_split, val_split], axis=0)
    # Generic fallback for other dataset types
    return ConcatDataset([train_split, val_split])

def _adapt_dataset(data, dataset_kind: str, y_scaler=None):
    """
    Convert raw dataset split to the batch format expected by DeepACE.
    """
    if dataset_kind.startswith("mimic_extract"):
        return DeepACEDataAdapter_MIMICExtract(data, y_scaler=y_scaler)
    else:
        raise ValueError(f"Invalid dataset kind: {dataset_kind}")


def tune_hyperparameters(train_dataset, val_dataset, cf_seq,args):
    """
    Hyperparameter tuning for DeepACE using simple random search
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
        trial_config = {k: random.choice(v) for k, v in hparams_grid.items()}
        logger.info(f"Trial {trial + 1}/{args.model.tune_range}: {trial_config}")
    
        trial_args = deepcopy(args)
        for k, v in trial_config.items():
            if hasattr(trial_args.model, k):
                setattr(trial_args.model, k, v)
        
        # Train and evaluate
        try:
            train_loader = DataLoader(train_dataset, batch_size=trial_args.model.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=trial_args.model.batch_size, shuffle=False)
            
            # Train DeepACE
            _, loss_GQ = train_deepace_model(
                train_loader=train_loader,
                val_loader=val_loader,
                cf_seq=cf_seq,
                args=trial_args
            )
            
            val_loss = loss_GQ
            logger.info(f"  Trial loss: {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
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
    Training / evaluation script for DeepACE with CATE estimation
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
    # data_seed = int(args.dataset.seed)

    # Initialisation of data (deterministic dataset generation)
    seed_everything(exp_seed)
    
    dataset_collections = {}
    cf_seqs = [tuple(list(seq)) for seq in args.dataset.cf_seqs]
    for cf_seq in cf_seqs:
        # Instantiate dataset per CF sequence
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

    # Close any open H5 handles (MIMIC uses a cached HDFStore during dataset creation).
    if args.dataset.name.startswith("mimic_extract"):
        try:
            if args.dataset.name.startswith("mimic_extract_complex"):
                from src.data.mimic_extract_complex.dataset import MIMICDataCache
            else:
                from src.data.mimic_extract.dataset import MIMICDataCache
            MIMICDataCache.close_all()
        except Exception as e:
            logger.warning(f"Failed to close MIMIC H5 cache cleanly: {e}")

    # Fit a global Y StandardScaler from all generated (train/val) splits, then reuse everywhere.
    y_scaler = None
    if args.dataset.name.startswith("mimic_extract"):
        _splits = []
        for _dc in dataset_collections.values():
            _splits.extend([_dc.train_f, _dc.val_f])
        y_scaler = DeepACEDataAdapter_MIMICExtract.fit_y_scaler(*_splits, y_index=0)
        logger.info(
            "Fitted global Y StandardScaler from generated data: "
            f"mean={float(y_scaler.mean_[0]):.6g}, std={float(y_scaler.scale_[0]):.6g}"
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
    logger.info("Loading ground truth CAPO...")
    ground_truth_apo = load_ground_truth_capo(args, cf_seqs, project_root=project_root, logger=logger)
    
    # Step 2: Hyperparameter tuning (if enabled)
    best_hparams = None
    if args.model.get('tune_hparams', False):
        cf_seq = cf_seqs[0]
        logger.info(f"Tuning hyperparameters for cf_seq={cf_seq} ...")
        train_dataset_adapted = _adapt_dataset(
            deepcopy(dataset_collections[cf_seq].train_f), args.dataset.name, y_scaler=y_scaler
        )
        val_dataset_adapted = _adapt_dataset(
            deepcopy(dataset_collections[cf_seq].val_f), args.dataset.name, y_scaler=y_scaler
        )
        best_hparams = tune_hyperparameters(train_dataset_adapted, val_dataset_adapted, cf_seq, args)

    
    # Step 3: Train DeepACE model for each counterfactual sequence
    logger.info("Training a separate DeepACE model for each counterfactual sequence...")
    for cf_seq in cf_seqs:
        # Per-sequence args (apply tuned hyperparameters for this cf_seq only)
        seq_args = deepcopy(args)
        if best_hparams:
            for k, v in best_hparams.items():
                if hasattr(seq_args.model, k):
                    setattr(seq_args.model, k, v)
        # combine train and val datasets to make final estimates
        concat_dataset = _combine_train_val(
            deepcopy(dataset_collections[cf_seq].train_f), deepcopy(dataset_collections[cf_seq].val_f), args.dataset.name
        )
        concat_dataset_adapted = _adapt_dataset(deepcopy(concat_dataset), args.dataset.name, y_scaler=y_scaler)
        train_loader = DataLoader(concat_dataset_adapted, batch_size=seq_args.model.batch_size, shuffle=True)
        model, _ = train_deepace_model(train_loader, None, cf_seq, seq_args)
        # No sample splitting: evaluate on the same factual population used for final fitting.
        eval_dataset_adapted = _adapt_dataset(
            deepcopy(concat_dataset), args.dataset.name, y_scaler=y_scaler
        )
        loader_for_eval = DataLoader(eval_dataset_adapted, batch_size=len(eval_dataset_adapted), shuffle=False)
        capo_results = estimate_capo_for_one_sequence(
            model, loader_for_eval, cf_seq, ground_truth_apo, y_scaler=y_scaler
        )
        results[cf_seq] = capo_results
    
    # Step 4: Export results
    logger.info("Exporting results...")
    export_capo_results(
        model_name=args.model.name,
        dataset_name=args.dataset.name,
        capo_results=results,
        ground_truth_apo=ground_truth_apo,
        cf_seqs=args.dataset.cf_seqs,
        output_dir=output_dir,
        args=args,
        estimate_field="deepace_estimate",
        bias_field="deepace_bias",
        results_filename=f"{args.model.name}_{args.dataset.name}_capo_results.json",
        best_hparams=best_hparams,
        outcome_scale={
            "type": "standardize",
            "mean": float(y_scaler.mean_[0]) if y_scaler is not None else None,
            "std": float(y_scaler.scale_[0]) if y_scaler is not None else None,
        },
        logger=logger,
    )
    
    return results


def train_deepace_model(train_loader, val_loader, cf_seq, args):
    """Train the DeepACE model"""

    # Get model configuration
    model_config = args.model
    
    # Create DeepACE config
    config = {
        "hidden_size_lstm":model_config.hidden_size_lstm,
        "hidden_size_body":model_config.hidden_size_body,
        "hidden_size_head":model_config.hidden_size_head,
        "batch_size":model_config.batch_size,
        "lr":model_config.learning_rate,
        "dropout":model_config.dropout
    }

    deepace = DeepACE(config=config, input_size=train_loader.dataset[0].shape[1] - 1, a_int=np.array(cf_seq), alpha=args.model.alpha, beta=args.model.beta)
    deepace.set_tune_mode(False)

    # from pytorch_lightning.loggers import TensorBoardLogger
    # tb_logger = TensorBoardLogger(save_dir=str(Path(project_root) / "logs"), name="deepace")

    # Train DeepACE
    logger.info(f"Training DeepACE model for sequence: {cf_seq}")
    # Trainer1 = pl.Trainer(max_epochs=int(args.exp.max_epochs), logger=tb_logger,log_every_n_steps=10)
    Trainer1 = pl.Trainer(max_epochs=int(args.exp.max_epochs), logger=False, enable_checkpointing=False)
    Trainer1.fit(deepace, train_loader, val_loader)
    if val_loader is not None:
        val_results = Trainer1.validate(model=deepace, dataloaders=val_loader, verbose=False)
        loss_GQ = val_results[0]['val_loss_last'] + val_results[0]['val_loss_a']
    else:
        loss_GQ = None
    return deepace, loss_GQ

def estimate_capo_for_one_sequence(model, train_dataloader, cf_seq, ground_truth_apo, y_scaler=None):
    """Estimate CAPO for each counterfactual sequence using DeepACE.

    For MIMIC datasets we standardize Y (and rebuild Y_prev) before training/eval.
    DeepACE can invert-transform the estimated outcome back to the raw Y scale via `y_scaler`.
    """
    
    logger.info(f"Estimating CAPO for cf_seq {cf_seq}: {cf_seq}")
    
    # Retrieve all samples in train_dataloader as a numpy array.
    all_samples = torch.cat([x.unsqueeze(0) for x in train_dataloader.dataset], dim=0)
    deepace_estimate = model.estimate_avg_outcome(all_samples.detach().numpy(), y_scaler=y_scaler)

    # Calculate bias
    ground_truth_val = ground_truth_apo[tuple(cf_seq)]
    deepace_bias = deepace_estimate - ground_truth_val
    
    logger.info(f"Results for {cf_seq}:")
    logger.info(f"  Ground truth: {ground_truth_val:.4f}")
    logger.info(f"  DeepACE: {deepace_estimate:.4f} (bias: {deepace_bias:.4f})")

    return {
        'deepace_estimate': deepace_estimate,
        'deepace_bias': deepace_bias,
        'ground_truth': ground_truth_val
    }


if __name__ == "__main__":
    # Add default arguments only if not provided via CLI (Hydra errors on duplicates).
    argv_rest = sys.argv[1:]
    if not any(a.startswith("+dataset=") or a.startswith("dataset=") for a in argv_rest):
        sys.argv.append("+dataset=mimic_extract_complex_tau15_h8")
    if not any(a.startswith("+model=") or a.startswith("model=") for a in argv_rest):
        sys.argv.append("+model=deepace")
    main()
