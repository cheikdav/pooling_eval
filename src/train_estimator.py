"""Train a single value estimator for one (batch, n_episodes) unit.

This module is the per-estimator training entry point. Each invocation trains
exactly ONE estimator — one method on one data batch for one episode count.
Fan-out over methods, batches, and episode counts is handled by the Snakemake
workflow at the repo root.
"""

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset

from src.config import (
    ExperimentConfig, BaseEstimatorConfig, resolve_param_for_episodes,
    FeatureExtractorType,
)
from src.estimators import ESTIMATOR_REGISTRY
from src.data_preprocessing import preprocess_episodes, TransitionDataset


METHOD_ABBREV = {'monte_carlo': 'MC', 'td': 'TD', 'td_lambda': 'TDλ'}


def get_method_abbreviation(method_name: str) -> str:
    return METHOD_ABBREV.get(method_name, method_name)


def load_batch_data(batch_path: Path, max_episodes: int = None) -> dict:
    data = np.load(batch_path, allow_pickle=True)
    batch = {}
    for key in data.keys():
        if key in ['observations', 'actions', 'rewards', 'dones', 'next_observations']:
            eps = [arr for arr in data[key]]
            batch[key] = eps if max_episodes is None else eps[:max_episodes]
        else:
            batch[key] = data[key]
    return batch


def apply_tuned_hyperparams(method_config: BaseEstimatorConfig, tuned_path: Path):
    """Return a copy of method_config with values from tuned_hyperparams.json merged in."""
    if not tuned_path.exists():
        return method_config
    with open(tuned_path, 'r') as f:
        tuned = json.load(f)
    updated = copy.deepcopy(method_config)
    for key, value in tuned.items():
        if value is None:
            continue
        if key == 'rbf_n_components' and updated.feature_extractor is not None:
            updated.feature_extractor.n_components = int(value)
        elif key == 'rbf_gamma' and updated.feature_extractor is not None:
            updated.feature_extractor.gamma = float(value)
        elif hasattr(updated, key):
            current = getattr(updated, key)
            if isinstance(current, bool):
                setattr(updated, key, bool(value))
            elif isinstance(current, int):
                setattr(updated, key, int(value))
            else:
                setattr(updated, key, value)
    return updated


def _resolve_method_config(method_config, num_episodes, experiment_config):
    """Resolve per-episode hyperparams and auto-set policy_path if needed."""
    resolved = method_config.resolve_for_episodes(num_episodes)
    if (resolved.feature_extractor
            and resolved.feature_extractor.type == FeatureExtractorType.POLICY_REPRESENTATION
            and resolved.feature_extractor.policy_path is None):
        policy_path = experiment_config.get_policy_dir() / "policy_final.zip"
        resolved.feature_extractor.policy_path = str(policy_path)
    return resolved


def train_single_estimator(
    method_config: BaseEstimatorConfig,
    train_batch: dict,
    validation_dataset,
    config: ExperimentConfig,
    method_name: str,
    batch_name: str,
    num_episodes: int,
    use_wandb: bool,
    checkpoint_dir: Path,
    sweep_mode: bool = False,
    log_frequency: int = 10,
):
    """Train one estimator via pl.Trainer and return (best_mc_loss, trained_module)."""
    gamma = config.value_estimators.training.gamma
    training_config = config.value_estimators.training
    obs_dim = train_batch['observations'].shape[-1]

    resolved_config = _resolve_method_config(method_config, num_episodes, config)

    ModuleClass = ESTIMATOR_REGISTRY[type(resolved_config)]
    module = ModuleClass.from_config(resolved_config, config.network, obs_dim, gamma)

    # Resolve batch_size and max_epochs for this episode count
    batch_size_raw = (resolved_config.batch_size if resolved_config.batch_size is not None
                     else training_config.batch_size)
    batch_size = resolve_param_for_episodes(batch_size_raw, num_episodes)
    max_epochs_raw = (resolved_config.max_epochs if resolved_config.max_epochs is not None
                     else training_config.max_epochs)
    max_epochs = resolve_param_for_episodes(max_epochs_raw, num_episodes)

    # Reward centering (method override takes precedence)
    reward_centering = (resolved_config.reward_centering
                       if resolved_config.reward_centering is not None
                       else training_config.reward_centering)

    train_dataset = TransitionDataset(train_batch)
    module.configure_pretraining(
        train_dataset=train_dataset,
        val_dataset=validation_dataset,
        batch_size=batch_size,
        reward_centering=reward_centering,
    )

    # Callbacks
    monitor = 'val/mc_loss' if validation_dataset is not None else 'train/mc_loss'
    callbacks = [
        EarlyStopping(monitor=monitor, mode='min',
                      patience=training_config.convergence_patience,
                      min_delta=training_config.convergence_threshold),
    ]
    ckpt_cb = None
    if not sweep_mode:
        ckpt_cb = ModelCheckpoint(
            dirpath=str(checkpoint_dir), filename='estimator',
            monitor=monitor, mode='min', save_top_k=1,
            save_weights_only=False, enable_version_counter=False,
        )
        callbacks.append(ckpt_cb)

    # Logger
    if use_wandb and config.logging.use_wandb and not sweep_mode:
        run_name = f"{get_method_abbreviation(method_name)} ({config.environment.name}, #{batch_name}, #ep {num_episodes})"
        save_dir = str(config.get_estimator_dir(method_config) / "wandb_offline") \
            if config.logging.wandb_mode == "offline" else None
        logger = WandbLogger(
            project=config.logging.get_project_name(config.environment.name),
            entity=config.logging.wandb_entity,
            name=run_name,
            group=config.experiment_id,
            mode=config.logging.wandb_mode,
            save_dir=save_dir,
            tags=[method_name, f"batch_{batch_name}", f"{num_episodes}_episodes",
                  config.environment.name],
        )
    else:
        logger = False

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=False,
        log_every_n_steps=max(1, log_frequency),
        num_sanity_val_steps=0,
        enable_checkpointing=not sweep_mode,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )

    train_active_idx = train_dataset.active.nonzero(as_tuple=True)[0].tolist()
    train_loader = DataLoader(Subset(train_dataset, train_active_idx),
                              batch_size=batch_size, shuffle=True)
    if validation_dataset is not None:
        val_active_idx = validation_dataset.active.nonzero(as_tuple=True)[0].tolist()
        val_loader = DataLoader(Subset(validation_dataset, val_active_idx),
                                batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    trainer.fit(module, train_loader, val_loader)

    # Recover best loss + load best weights into the in-memory module
    if ckpt_cb is not None and ckpt_cb.best_model_score is not None:
        best_mc_loss = float(ckpt_cb.best_model_score.item())
        if ckpt_cb.best_model_path:
            best = torch.load(ckpt_cb.best_model_path, map_location=module.device,
                              weights_only=False)
            module.load_state_dict(best['state_dict'], strict=False)
            module.on_load_checkpoint(best)
    else:
        metric = trainer.callback_metrics.get(monitor)
        best_mc_loss = float(metric.item()) if metric is not None else float('inf')

    return best_mc_loss, module


def train_one_estimator(
    config: ExperimentConfig,
    method_config: BaseEstimatorConfig,
    batch_path: Path,
    output_dir: Path,
    batch_name: str,
    n_episodes: int,
    use_wandb: bool = True,
    sweep_mode: bool = False,
    log_frequency: Optional[int] = None,
) -> dict:
    """Train one estimator for a single (batch, n_episodes) unit.

    Atomic training operation. Fan-out over methods, batches, and episode counts
    is the caller's responsibility.
    """
    gamma = config.value_estimators.training.gamma
    truncation_coefficient = config.value_estimators.training.truncation_coefficient
    method_name = method_config.name
    log_freq = log_frequency if log_frequency is not None else config.logging.log_frequency

    if not sweep_mode:
        tuned_path = config.get_estimator_dir(method_config) / "sweeps" / "tuned_hyperparams.json"
        method_config = apply_tuned_hyperparams(method_config, tuned_path)

    train_seed = (config.value_estimators.training.seed
                  + hash((method_name, batch_name, n_episodes)) % (2**31))
    pl.seed_everything(train_seed, workers=True)

    # Validation batch
    validation_batch_path = batch_path.parent / f"{batch_path.stem}_validation.npz"
    validation_dataset = None
    if validation_batch_path.exists():
        print(f"Found validation batch at {validation_batch_path}")
        val_raw = load_batch_data(validation_batch_path)
        val_proc = preprocess_episodes(val_raw, gamma, truncation_coefficient)
        validation_dataset = TransitionDataset(val_proc)
    else:
        print(f"No validation batch at {validation_batch_path}; training without validation.")

    # Training batch (capped to n_episodes for subset ablation)
    train_raw = load_batch_data(batch_path, max_episodes=n_episodes)
    train_batch = preprocess_episodes(train_raw, gamma, truncation_coefficient)

    episodes_dir = output_dir / str(n_episodes) / f"batch_{batch_name}"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = episodes_dir / "estimator.pt"

    print(f"\nTraining {method_name}  batch={batch_name}  n_ep={n_episodes}")

    best_mc_loss, module = train_single_estimator(
        method_config, train_batch, validation_dataset, config, method_name, batch_name,
        n_episodes, use_wandb=use_wandb, checkpoint_dir=episodes_dir,
        sweep_mode=sweep_mode, log_frequency=log_freq,
    )

    if not sweep_mode:
        # Rename Lightning's .ckpt to the expected .pt
        lightning_ckpt = episodes_dir / "estimator.ckpt"
        if lightning_ckpt.exists():
            lightning_ckpt.rename(checkpoint_path)
        print(f"Saved estimator (best MC loss={best_mc_loss:.6f}) to {checkpoint_path}")

        stats = {
            'method': method_name,
            'batch_name': batch_name,
            'n_episodes': n_episodes,
            'final_mc_loss': best_mc_loss,
        }
        with open(episodes_dir / "training_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        data_metadata = {}
        data_metadata_path = batch_path.parent / "data_metadata.json"
        if data_metadata_path.exists():
            with open(data_metadata_path, 'r') as f:
                data_metadata = json.load(f)

        estimator_metadata = {
            'method': method_name,
            'batch_name': batch_name,
            'n_episodes': n_episodes,
            'batch_path': str(batch_path),
            'gamma': gamma,
            'final_mc_loss': best_mc_loss,
            'seed': config.value_estimators.training.seed,
            'max_epochs': config.value_estimators.training.max_epochs,
            'convergence_threshold': config.value_estimators.training.convergence_threshold,
            'convergence_patience': config.value_estimators.training.convergence_patience,
            'estimator_config': asdict(method_config),
            'network_config': {
                'hidden_sizes': config.network.hidden_sizes,
                'activation': config.network.activation,
            },
            'data_metadata': data_metadata,
        }
        with open(episodes_dir / "estimator_metadata.json", 'w') as f:
            json.dump(estimator_metadata, f, indent=2)

    print(f"Training complete for {method_name} batch_{batch_name} n_ep={n_episodes}.")
    return {'num_episodes': n_episodes, 'best_mc_loss': best_mc_loss}


def main():
    parser = argparse.ArgumentParser(
        description="Train one value estimator for a single (batch, n_episodes) unit."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--batch-idx", type=int, required=True)
    parser.add_argument("--n-episodes", type=int, required=True)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)

    method_config = None
    for mc in config.value_estimators.method_configs:
        if mc.name == args.method:
            method_config = mc
            break
    if method_config is None:
        available = [mc.name for mc in config.value_estimators.method_configs]
        parser.error(f"Method '{args.method}' not found. Available: {available}")

    batch_name = str(args.batch_idx)
    batch_path = config.get_data_dir() / f"batch_{batch_name}.npz"
    output_dir = config.get_estimator_dir(method_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    train_one_estimator(
        config=config,
        method_config=method_config,
        batch_path=batch_path,
        output_dir=output_dir,
        batch_name=batch_name,
        n_episodes=args.n_episodes,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
