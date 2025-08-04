# pretrain_datasets.py
import argparse
import os
import logging
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
import fsspec
import tempfile
import json

from litdata.streaming import (
    CombinedStreamingDataset,
    StreamingDataLoader,
    StreamingDataset,
    TokensLoader,
)
from litgpt import Tokenizer
from litgpt.data import DataModule
from torch.utils.data import DataLoader

MAX_SHARDS = 50

# Dataset type configurations
DATASET_TYPE_CONFIGS = {
    'main': {'prefix': '', 'description': 'Main web data'},
    'code': {'prefix': 'c', 'description': 'Code datasets'},
    'math': {'prefix': 'm', 'description': 'Math datasets'},
    'instruct': {'prefix': 'i', 'description': 'Instruction datasets'},
    'qa': {'prefix': 'q', 'description': 'QA datasets'},
    'books': {'prefix': 'b', 'description': 'Book datasets'},
    'news': {'prefix': 'n', 'description': 'News datasets'},
    'papers': {'prefix': 'p', 'description': 'Academic papers'},
    'web': {'prefix': 'w', 'description': 'Web datasets'}, 
}

def process_and_tokenize(dataset, tokenizer, idx):
    """Process a dataset item and tokenize it."""
    try:
        example = dataset[idx]
        text = example.get("text", "")
        if text and isinstance(text, str):
            tokens = tokenizer.encode(text.strip(), eos=True)
            if tokens is not None:
                yield tokens
    except Exception as e:
        logging.warning(f"Error processing index {idx}: {str(e)}")

@dataclass
class DatasetMixConfig:
    """Configuration for dataset mixing."""
    weights: Dict[str, float] = field(default_factory=dict)
    proportional_sampling: bool = False
    literal_weights: Dict[str, float] = field(default_factory=dict)  # Direct shard weights
    
    def __post_init__(self):
        """Validate and normalize weights."""
        if not self.weights and not self.literal_weights:
            self.weights = {'main': 1.0}
        
        # If using literal weights, validate they sum to reasonable values
        if self.literal_weights:
            total = sum(self.literal_weights.values())
            if total <= 0:
                raise ValueError("Total literal weights must be positive")
            # Normalize literal weights to sum to 1.0
            self.literal_weights = {k: v / total for k, v in self.literal_weights.items()}
            return
        
        # Only normalize type weights if not using proportional sampling
        if not self.proportional_sampling:
            total = sum(self.weights.values())
            if total <= 0:
                raise ValueError("Total weight must be positive")
            self.weights = {k: v / total for k, v in self.weights.items()}
        
        # Validate dataset types
        invalid_types = set(self.weights.keys()) - set(DATASET_TYPE_CONFIGS.keys())
        if invalid_types:
            raise ValueError(f"Invalid dataset types: {invalid_types}. "
                           f"Valid types: {list(DATASET_TYPE_CONFIGS.keys())}")
    
    @classmethod
    def from_string(cls, weights_str: str, proportional_sampling: bool = False) -> 'DatasetMixConfig':
        """Create config from string like 'main:0.7,code:0.2,math:0.1'"""
        weights = {}
        for pair in weights_str.split(','):
            if ':' not in pair:
                raise ValueError(f"Invalid weight format: {pair}. Expected 'type:weight'")
            dataset_type, weight_str = pair.strip().split(':')
            weights[dataset_type.strip()] = float(weight_str.strip())
        return cls(weights=weights, proportional_sampling=proportional_sampling)
    
    @classmethod  
    def from_literal_string(cls, literal_str: str) -> 'DatasetMixConfig':
        """Create config from literal shard weights like 'main/1:0.8,q1:0.1,q2:0.05,q3:0.05'"""
        literal_weights = {}
        for pair in literal_str.split(','):
            if ':' not in pair:
                raise ValueError(f"Invalid weight format: {pair}. Expected 'shard:weight'")
            shard_name, weight_str = pair.strip().split(':')
            literal_weights[shard_name.strip()] = float(weight_str.strip())
        return cls(literal_weights=literal_weights)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'weights': self.weights,
            'proportional_sampling': self.proportional_sampling,
            'literal_weights': self.literal_weights
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'DatasetMixConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

@dataclass
class BaseStreamingDataset(DataModule):
    """Enhanced base class for streaming pretraining datasets with flexible mixing."""
    
    # Core paths
    data_path: str
    val_data_path: Optional[str] = None
    
    # Dataset mixing configuration
    mix_config: Optional[DatasetMixConfig] = None
    mix_config_path: Optional[str] = None
    mix_weights_str: Optional[str] = None
    literal_weights_str: Optional[str] = None  # New option for direct shard weights
    proportional_sampling: bool = False
    
    # Training parameters
    val_split_fraction: float = 0.003
    seed: int = 42
    num_workers: int = 8
    fast_dev_run: bool = False
    
    # Internal fields
    tokenizer: Optional[Tokenizer] = field(default=None, repr=False, init=False)
    batch_size: int = field(default=1, repr=False, init=False)
    seq_length: int = field(default=2048, repr=False, init=False)
    is_gcs_fs: bool = field(default=False, repr=False, init=False)
    local_cache_dir: Optional[str] = field(default=None, repr=False, init=False)
    fs: Optional[fsspec.AbstractFileSystem] = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        # Setup mix configuration
        self._setup_mix_config()
        
        # Define training and validation directories
        if not self.val_data_path:
            self.data_path_train = os.path.join(self.data_path, "train")
            self.data_path_val = os.path.join(self.data_path, "val")
        else:
            self.data_path_train = self.data_path
            self.data_path_val = self.val_data_path

        # Setup filesystem
        self._setup_filesystem()
    
    def _setup_mix_config(self) -> None:
        """Setup dataset mixing configuration from various sources."""
        config_sources = [
            self.mix_config,
            self.mix_config_path and DatasetMixConfig.load(self.mix_config_path),
            self.literal_weights_str and DatasetMixConfig.from_literal_string(self.literal_weights_str),
            self.mix_weights_str and DatasetMixConfig.from_string(self.mix_weights_str, self.proportional_sampling)
        ]
        
        # Use the first non-None configuration
        self.mix_config = next((config for config in config_sources if config), 
                              DatasetMixConfig(proportional_sampling=self.proportional_sampling))
        
        if self.mix_config.literal_weights:
            print(f"Using literal shard weights: {self.mix_config.literal_weights}")
        else:
            print(f"Dataset mixing configuration: {self.mix_config.weights}")
            print(f"Proportional sampling: {self.mix_config.proportional_sampling}")
    
    def _setup_filesystem(self) -> None:
        """Setup filesystem for GCS or local storage."""
        if self.data_path_train.startswith("gs://"):
            if self.data_path_val and not self.data_path_val.startswith("gs://"):
                raise ValueError("Both train and val paths should be GCS or both local")
            
            self.is_gcs_fs = True
            self.data_path_train = str(self.data_path_train)
            self.data_path_val = str(self.data_path_val) if self.data_path_val else None
            self.fs = fsspec.filesystem("gcs")
            self.local_cache_dir = tempfile.mkdtemp(prefix="streaming_data_cache_")

    def connect(self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, 
                max_seq_length: Optional[int] = 512) -> None:
        """Connect tokenizer and set training parameters."""
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1  # +1 for next token prediction

    def _path_exists(self, path: str) -> bool:
        """Check if path exists (works for both GCS and local)."""
        if self.is_gcs_fs:
            return self.fs.exists(path)
        return os.path.exists(path)
    
    def _is_directory(self, path: str) -> bool:
        """Check if path is a directory (works for both GCS and local)."""
        if self.is_gcs_fs:
            return self.fs.isdir(path)
        return os.path.isdir(path)
    
    def _list_directory(self, path: str) -> List[str]:
        """List directory contents (works for both GCS and local)."""
        if self.is_gcs_fs:
            return [os.path.basename(p) for p in self.fs.ls(path)]
        return os.listdir(path)

    def _discover_dataset_shards(self, base_dir: str) -> Dict[str, List[str]]:
        """Discover all dataset shards organized by type."""
        discovered = {dataset_type: [] for dataset_type in DATASET_TYPE_CONFIGS.keys()}
        
        if not self._path_exists(base_dir):
            logging.warning(f"Base directory {base_dir} does not exist")
            return discovered
        
        # Discover main dataset shards (numbered directories)
        for i in range(1, MAX_SHARDS + 1):
            shard_dir = os.path.join(base_dir, str(i))
            index_path = os.path.join(shard_dir, "index.json")
            if self._path_exists(index_path):
                discovered['main'].append(shard_dir)
        
        # Discover typed dataset shards
        try:
            dir_contents = self._list_directory(base_dir)
        except Exception as e:
            logging.warning(f"Could not list directory {base_dir}: {e}")
            return discovered
        
        for item in dir_contents:
            item_path = os.path.join(base_dir, item)
            if not self._is_directory(item_path):
                continue
                
            # Check if this matches any dataset type prefix pattern
            for dataset_type, config in DATASET_TYPE_CONFIGS.items():
                if dataset_type == 'main':
                    continue
                    
                prefix = config['prefix']
                # Match patterns like 'c1', 'c2', 'm1', 'm2', 'q1', 'q2', etc.
                if (item.startswith(prefix) and 
                    len(item) > len(prefix) and 
                    item[len(prefix):].isdigit()):
                    
                    index_path = os.path.join(item_path, "index.json")
                    if self._path_exists(index_path):
                        discovered[dataset_type].append(item_path)
        
        # Log discovery results
        for dataset_type, shards in discovered.items():
            if shards:
                print(f"Found {len(shards)} {dataset_type} shards")
        
        return discovered

    def _create_streaming_dataset(self, shard_path: str) -> StreamingDataset:
        """Create a StreamingDataset for a single shard."""
        return StreamingDataset(
            input_dir=shard_path,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
            drop_last=True,
        )

    def _get_shard_size(self, shard_path: str) -> int:
        """Get the number of samples in a shard from its index.json"""
        index_path = os.path.join(shard_path, "index.json")
        try:
            with open(index_path, 'r') as f:
                index_data = json.load(f)
                
                # For litgpt format, try to estimate samples from chunk data
                if 'chunks' in index_data:
                    # Method 1: Sum chunk_size (tokens) and divide by sequence length
                    total_tokens = sum(chunk.get('chunk_size', 0) for chunk in index_data['chunks'])
                    if total_tokens > 0:
                        # Assume sequence length of 2048 (adjust if different)
                        estimated_samples = total_tokens // self.seq_length if self.seq_length > 1 else total_tokens // 2048
                        return max(estimated_samples, 1)
                    
                    # Method 2: Count number of chunks as rough estimate
                    return len(index_data['chunks'])
                
                # Fallback to other possible size indicators
                return index_data.get('config', {}).get('data_spec', {}).get('length', 
                       index_data.get('length', 
                       index_data.get('num_samples', 
                       index_data.get('num_items', 1))))
        except Exception as e:
            print(f"Warning: Could not read shard size from {index_path}: {e}")
            return 1  # Fallback

    def _create_combined_dataset(self, base_dir: str) -> CombinedStreamingDataset:
        """Create a combined dataset with proper mixing weights."""
        discovered_shards = self._discover_dataset_shards(base_dir)
        
        # Filter to only include dataset types that have shards
        available_types = {k: v for k, v in discovered_shards.items() if v}
        
        if not available_types:
            raise ValueError(f"No shards found in {base_dir}")
        
        # Create datasets and calculate weights
        all_datasets = []
        all_weights = []
        
        if self.mix_config.literal_weights:
            # Use literal shard weights - direct mapping
            print("Using literal shard weights")
            
            # Create a mapping of shard paths to their identifiers
            shard_path_to_id = {}
            for dataset_type, shard_paths in available_types.items():
                for shard_path in shard_paths:
                    shard_name = os.path.basename(shard_path)
                    # Handle main shards (numbered directories)
                    if dataset_type == 'main':
                        shard_id = f"main/{shard_name}"  # e.g., "main/1"
                    else:
                        shard_id = shard_name  # e.g., "q1", "c1", "m1"
                    shard_path_to_id[shard_path] = shard_id
            
            # Load datasets with literal weights
            for shard_path, shard_id in shard_path_to_id.items():
                if shard_id in self.mix_config.literal_weights:
                    weight = self.mix_config.literal_weights[shard_id]
                    print(f"Loading shard: {shard_path} (ID: {shard_id}, weight: {weight:.4f})")
                    dataset = self._create_streaming_dataset(shard_path)
                    all_datasets.append(dataset)
                    all_weights.append(weight)
                else:
                    print(f"Warning: No weight specified for shard {shard_id}, skipping")
            
        elif self.mix_config.proportional_sampling:
            # Calculate weights proportional to actual dataset sizes (number of samples)
            type_sizes = {}
            shard_sizes = {}
            
            # First pass: get sizes for each dataset type
            for dataset_type, shard_paths in available_types.items():
                type_total = 0
                for shard_path in shard_paths:
                    shard_size = self._get_shard_size(shard_path)
                    shard_sizes[shard_path] = shard_size
                    type_total += shard_size
                type_sizes[dataset_type] = type_total
                print(f"{dataset_type} total samples: {type_total:,}")
            
            total_samples = sum(type_sizes.values())
            print(f"Total samples across all types: {total_samples:,}")
            
            # Second pass: calculate weights and create datasets
            for dataset_type, shard_paths in available_types.items():
                # Base weight proportional to dataset size
                base_type_weight = type_sizes[dataset_type] / total_samples
                
                # Apply any type-specific multiplier if specified in config
                multiplier = self.mix_config.weights.get(dataset_type, 1.0)
                adjusted_type_weight = base_type_weight * multiplier
                
                for shard_path in shard_paths:
                    # Weight this shard proportional to its share of the dataset type
                    shard_size = shard_sizes[shard_path]
                    shard_proportion = shard_size / type_sizes[dataset_type]
                    shard_weight = adjusted_type_weight * shard_proportion
                    
                    print(f"Loading {dataset_type} shard: {shard_path}")
                    print(f"  Samples: {shard_size:,}, Weight: {shard_weight:.4f}")
                    
                    dataset = self._create_streaming_dataset(shard_path)
                    all_datasets.append(dataset)
                    all_weights.append(shard_weight)
        else:
            # Original logic for specified weights
            available_types = {k: v for k, v in available_types.items() 
                              if k in self.mix_config.weights}
            
            if not available_types:
                raise ValueError(f"No shards found for any configured dataset types in {base_dir}")
            
            for dataset_type, shard_paths in available_types.items():
                type_weight = self.mix_config.weights[dataset_type]
                weight_per_shard = type_weight / len(shard_paths)
                
                for shard_path in shard_paths:
                    print(f"Loading {dataset_type} shard: {shard_path} (weight: {weight_per_shard:.4f})")
                    dataset = self._create_streaming_dataset(shard_path)
                    all_datasets.append(dataset)
                    all_weights.append(weight_per_shard)
        
        # Normalize weights to ensure they sum to 1.0 (unless using literal weights which are already normalized)
        if not self.mix_config.literal_weights:
            weight_sum = sum(all_weights)
            all_weights = [w / weight_sum for w in all_weights]
        
        sampling_mode = "literal" if self.mix_config.literal_weights else ("proportional" if self.mix_config.proportional_sampling else "weighted")
        print(f"Created combined dataset with {len(all_datasets)} shards")
        print(f"Sampling mode: {sampling_mode}")
        print(f"Final weights: {[f'{w:.4f}' for w in all_weights]}")
        
        return CombinedStreamingDataset(
            datasets=all_datasets,
            seed=self.seed,
            iterate_over_all=False,  # Use sampling based on weights
            weights=all_weights,
        )

    def _has_any_shards(self, base_dir: str) -> bool:
        """Check if directory has any valid shards."""
        discovered = self._discover_dataset_shards(base_dir)
        return any(shards for shards in discovered.values())

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self._has_any_shards(self.data_path_train):
            train_dataset = self._create_combined_dataset(self.data_path_train)
        else:
            # Fallback to single dataset
            train_dataset = self._create_streaming_dataset(self.data_path_train)
        
        return StreamingDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        val_dataset = self._create_streaming_dataset(self.data_path_val)
        
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def prepare_data(self) -> None:
        """Prepare data - to be implemented by subclasses."""
        raise NotImplementedError("prepare_data() must be implemented in subclasses")

    def save_mix_config(self, path: Optional[str] = None) -> None:
        """Save the current mixing configuration."""
        if path is None:
            path = os.path.join(self.data_path, "mix_config.json")
        self.mix_config.save(path)
        print(f"Saved mixing configuration to {path}")


@dataclass
class ShardedMixedDataset(BaseStreamingDataset):
    """Data module for sharded mixed datasets with flexible mixing."""

    data_split: str = "default"
    dataset_name: str = "allenai/olmo-mix-1124"
    num_samples: int = 10_000_000
    num_val_samples: int = 10_000

    def prepare_data(self) -> None:
        """Prepare Olmo2 dataset."""
        from datasets import load_dataset
        from litdata import optimize
        from itertools import islice

        if self._check_data_exists():
            return

        # Load dataset
        hf_cache_dir = os.getenv("HF_HOME", None)
        print(f"Loading Olmo2 dataset: {self.dataset_name}")
        
        dataset = load_dataset(
            self.dataset_name, 
            name=self.data_split, 
            cache_dir=hf_cache_dir, 
            split="train", 
            streaming=True
        )
        
        # Shuffle and split
        dataset = dataset.shuffle(seed=self.seed, buffer_size=10000)
        total_samples = self.num_samples + self.num_val_samples
        dataset_list = list(islice(dataset, total_samples))
        
        split_dataset = {
            "train": dataset_list[:self.num_samples],
            "val": dataset_list[self.num_samples:],
        }

        self._process_and_save_splits(split_dataset)

    def _check_data_exists(self) -> bool:
        """Check if processed data already exists."""
        train_exists = self._path_exists(self.data_path_train)
        val_exists = self._path_exists(self.data_path_val)
        
        if train_exists and val_exists:
            print(f"Found existing data at {self.data_path}. Skipping preprocessing.")
            return True
        return False

    def _process_and_save_splits(self, split_dataset: Dict) -> None:
        """Process and save train/val splits."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call connect() first.")

        from litdata import optimize

        for split_name, data in split_dataset.items():
            output_dir = self.data_path_train if split_name == "train" else self.data_path_val
            print(f"Processing {split_name} split to {output_dir}")
            
            optimize(
                fn=partial(process_and_tokenize, data, self.tokenizer),
                inputs=list(range(len(data))),
                output_dir=output_dir,
                num_workers=self.num_workers,
                chunk_bytes="200MB",
                fast_dev_run=self.fast_dev_run,
            )
        
        print(f"Finished preprocessing Olmo2 data at {self.data_path}")


def create_dataset_from_config(dataset_type: str, **kwargs) -> BaseStreamingDataset:
    """Factory function to create datasets based on type."""
    dataset_classes = {
        'olmo2': ShardedMixedDataset,
    }
    
    if dataset_type not in dataset_classes:
        raise ValueError(f"Unknown dataset type: {dataset_type}. "
                        f"Available: {list(dataset_classes.keys())}")
    
    return dataset_classes[dataset_type](**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and manage pretraining datasets")
    
    # Dataset selection
    parser.add_argument("--data_path", type=str, required=True,
                       help="Base directory for data storage")
    parser.add_argument("--dataset_type", type=str, choices=["dolmino", "olmo2"], 
                       default="olmo2", help="Dataset type to use")
    parser.add_argument("--data_split", type=str, default="default",
                       help="Data split to use")
    
    # Mixing configuration (mutually exclusive)
    mix_group = parser.add_mutually_exclusive_group()
    mix_group.add_argument("--mix_weights", type=str,
                          help="Dataset mixing weights (e.g., 'main:0.7,code:0.2,math:0.1')")
    mix_group.add_argument("--mix_config_path", type=str,
                          help="Path to JSON file with mixing configuration")
    
    # Sampling options
    parser.add_argument("--proportional_sampling", action="store_true",
                       help="Use proportional sampling based on dataset sizes")
    
    # Other options
    parser.add_argument("--fast_dev_run", action="store_true",
                       help="Run in fast development mode")
    parser.add_argument("--save_config", type=str,
                       help="Save mixing configuration to specified path")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of workers for data processing")
    
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')

    # Create dataset configuration
    dataset_kwargs = {
        'data_path': args.data_path,
        'data_split': args.data_split,
        'fast_dev_run': args.fast_dev_run,
        'num_workers': args.num_workers,
        'proportional_sampling': args.proportional_sampling,
    }
    
    # Add mixing configuration if provided
    if args.mix_weights:
        dataset_kwargs['mix_weights_str'] = args.mix_weights
    elif args.mix_config_path:
        dataset_kwargs['mix_config_path'] = args.mix_config_path

    # Create and setup dataset
    data_module = create_dataset_from_config(args.dataset_type, **dataset_kwargs)
    
    # Save configuration if requested
    if args.save_config:
        data_module.save_mix_config(args.save_config)
        print(f"Configuration saved to {args.save_config}")
    
    # Example usage (you would normally connect tokenizer before calling prepare_data)
    print("Dataset module created successfully")
    print(f"Mix configuration: {data_module.mix_config.weights}")
    print(f"Proportional sampling: {data_module.mix_config.proportional_sampling}")
    
    # Uncomment these lines when you have a tokenizer:
    # tokenizer = Tokenizer(...)
    # data_module.connect(tokenizer=tokenizer, batch_size=1, max_seq_length=2048)
    # data_module.prepare_data()