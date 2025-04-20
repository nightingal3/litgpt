# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader, Dataset

from litgpt.model import Block, GPT, Config
from litgpt.tokenizer import Tokenizer
from litgpt.args import EvalArgs
from litgpt.utils import (
    chunked_cross_entropy,
    extend_checkpoint_dir,
    load_checkpoint,
    num_parameters,
    parse_devices,
)


class JsonDataset(Dataset):
    """Dataset for loading data from a JSON file."""
    
    def __init__(self, json_path, tokenizer, max_seq_length):
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Load the data
        with open(json_path) as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("text", "")
        
        # Tokenize text
        tokens = self.tokenizer.encode(text, device="cpu")
        
        # Handle sequence length
        if len(tokens) > self.max_seq_length + 1:
            tokens = tokens[:self.max_seq_length + 1]
        
        return tokens


def evaluate_on_datasets(
    checkpoint_dir: Path,
    dataset_paths: List[Path],
    out_dir: Optional[Path] = None,
    precision: Optional[str] = None,
    devices: Union[int, str] = 1,
    eval: EvalArgs = EvalArgs(max_iters=100),
    seed: int = 1337,
) -> None:
    """Evaluate a model on multiple datasets.

    Arguments:
        checkpoint_dir: The path to the model's checkpoint directory.
        dataset_paths: List of paths to directories containing val.json files.
        out_dir: Directory in which to save the evaluation results.
        precision: The precision to use for evaluation.
        devices: How many devices/GPUs to use.
        eval: Evaluation-related arguments.
        seed: The random seed to use for reproducibility.
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    print(f"Evaluating model from {checkpoint_dir} on {len(dataset_paths)} datasets")
    
    devices = parse_devices(devices)
    if out_dir is None:
        out_dir = checkpoint_dir / "multi_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    if devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"
        
    from lightning.fabric.loggers import CSVLogger
    logger = CSVLogger(out_dir, name="multi_eval")
    
    fabric = L.Fabric(
        devices=devices, strategy=strategy, precision=precision, loggers=[logger]
    )
    fabric.launch(main, config, dataset_paths, checkpoint_dir, out_dir, eval, seed)


def main(
    fabric: L.Fabric,
    config: Config,
    dataset_paths: List[Path],
    checkpoint_dir: Path,
    out_dir: Path,
    eval: EvalArgs,
    seed: int,
) -> None:
    """Run evaluation on multiple datasets."""
    fabric.seed_everything(seed)
    tokenizer = Tokenizer(checkpoint_dir)
    
    fabric.print(f"Loading model from {checkpoint_dir}")
    t0 = time.perf_counter()
    
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)
    
    load_checkpoint(fabric, model, checkpoint_dir / "lit_model.pth")
    fabric.print(f"Time to load model: {time.perf_counter() - t0:.2f} seconds")
    fabric.print(f"Number of parameters: {num_parameters(model):,}")
    
    model = fabric.setup(model)
    
    # Track results
    results = {}
    
    # Evaluate on each dataset
    for dataset_path in dataset_paths:
        val_json_path = dataset_path / "val.json"
        if not val_json_path.exists():
            fabric.print(f"Warning: val.json not found in {dataset_path}, skipping...")
            continue
            
        dataset_name = dataset_path.name
        fabric.print(f"\nEvaluating on {dataset_name}...")
        
        # Create dataset and dataloader
        dataset = JsonDataset(val_json_path, tokenizer, model.max_seq_length)
        val_dataloader = DataLoader(
            dataset, 
            batch_size=1,  # Can be increased if memory allows
            shuffle=False,
            pin_memory=True
        )
        val_dataloader = fabric.setup_dataloaders(val_dataloader)
        
        # Evaluate
        t0 = time.perf_counter()
        val_loss = validate(fabric, model, val_dataloader, eval)
        val_ppl = math.exp(val_loss)
        eval_time = time.perf_counter() - t0
        
        # Log results
        fabric.print(
            f"Dataset: {dataset_name}, "
            f"Loss: {val_loss:.4f}, "
            f"Perplexity: {val_ppl:.4f}, "
            f"Time: {eval_time:.2f}s"
        )
        
        # Save metrics
        metrics = {
            f"{dataset_name}_loss": val_loss,
            f"{dataset_name}_ppl": val_ppl,
            f"{dataset_name}_time": eval_time,
        }
        results[dataset_name] = metrics
        fabric.log_dict(metrics)
    
    # Save results to a summary file
    import json
    with open(out_dir / "results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    fabric.print(f"\nEvaluation complete. Results saved to {out_dir}")


@torch.no_grad()
def validate(
    fabric: L.Fabric, model: GPT, val_dataloader: DataLoader, eval: EvalArgs
) -> torch.Tensor:
    """Run validation on a dataset using the same approach as in LitGPT training code."""
    fabric.print("Validating...")
    model.eval()
    
    losses = torch.zeros(min(len(val_dataloader), eval.max_iters))
    for k, batch in enumerate(val_dataloader):
        if k >= eval.max_iters:
            break
        
        # Process batch similar to pretrain/finetune code
        if batch.size(1) <= model.max_seq_length + 1:
            # Pad if necessary
            if batch.size(1) < model.max_seq_length + 1:
                padding = torch.zeros(
                    (batch.size(0), model.max_seq_length + 1 - batch.size(1)),
                    dtype=batch.dtype,
                    device=batch.device
                )
                batch = torch.cat([batch, padding], dim=1)
            
            input_ids = batch[:, 0:model.max_seq_length].contiguous()
            targets = batch[:, 1:(model.max_seq_length + 1)].contiguous()
        else:
            # Truncate if necessary
            input_ids = batch[:, 0:model.max_seq_length].contiguous()
            targets = batch[:, 1:(model.max_seq_length + 1)].contiguous()
        
        # Forward pass and loss calculation
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., :-1], chunk_size=0)
    
    val_loss = losses.mean()
    model.train()
    return val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on multiple JSON datasets")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the model checkpoint directory")
    parser.add_argument("--dataset_dirs", type=str, required=True, 
                       help="Comma-separated list of paths to directories containing val.json files")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for evaluation results")
    parser.add_argument("--precision", type=str, default=None, help="Precision for computation")
    parser.add_argument("--devices", type=str, default="1", help="Number of devices to use")
    parser.add_argument("--max_iters", type=int, default=100, help="Maximum number of validation iterations per dataset")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    
    args = parser.parse_args()
    
    # Parse dataset directories
    dataset_paths = [Path(path.strip()) for path in args.dataset_dirs.split(",")]
    
    # Create EvalArgs
    eval_args = EvalArgs(max_iters=args.max_iters)
    
    # Run evaluation
    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir) if args.out_dir else None
    
    evaluate_on_datasets(
        checkpoint_dir=checkpoint_dir,
        dataset_paths=dataset_paths,
        out_dir=out_dir,
        precision=args.precision,
        devices=args.devices,
        eval=eval_args,
        seed=args.seed,
    )