# pretrain_datasets.py
import argparse
import os
import logging
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional, Union

from litdata.streaming import (
    CombinedStreamingDataset,
    StreamingDataLoader,
    StreamingDataset,
    TokensLoader,
)
from litgpt import Tokenizer
from litgpt.data import DataModule
from torch.utils.data import DataLoader

# Common tokenization functions (if you need slight differences, you can override in the subclass)
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
class BaseOlmoDataset(DataModule):
    """Base class for pretraining datasets."""
    data_path: Union[str, Path]
    val_data_path: Optional[Union[str, Path]] = None
    val_split_fraction: float = 0.003
    seed: int = 42
    num_workers: int = 8
    fast_dev_run: bool = False

    tokenizer: Optional[Tokenizer] = field(default=None, repr=False, init=False)
    batch_size: int = field(default=1, repr=False, init=False)
    seq_length: int = field(default=2048, repr=False, init=False)

    def __post_init__(self) -> None:
        # Define training and validation directories.
        if not self.val_data_path:
            self.data_path_train = os.path.join(self.data_path, "train")
            self.data_path_val = os.path.join(self.data_path, "val")
        else:
            self.data_path_train = self.data_path
            self.data_path_val = self.val_data_path

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        # Increase by one because we need the next token as well.
        self.seq_length = max_seq_length + 1

    def _has_sharded_structure(self, base_dir: Union[str, Path]) -> bool:
        """Check if the directory has numbered subdirectories (0-3) with index.json files."""
        for i in range(4):
            shard_dir = os.path.join(base_dir, str(i))
            if os.path.isdir(shard_dir) and os.path.exists(os.path.join(shard_dir, "index.json")):
                return True
        return False

    def _create_combined_dataset(self, base_dir: Union[str, Path]) -> CombinedStreamingDataset:
        datasets = []
        for i in range(4):
            shard_dir = os.path.join(base_dir, str(i))
            if os.path.isdir(shard_dir) and os.path.exists(os.path.join(shard_dir, "index.json")):
                print(f"Loading shard {i} from {shard_dir}")
                dataset = StreamingDataset(
                    input_dir=shard_dir,
                    item_loader=TokensLoader(block_size=self.seq_length),
                    shuffle=True,
                    drop_last=True,
                )
                datasets.append(dataset)
            else:
                print(f"Warning: Shard {i} at {shard_dir} not found or missing index.json")
        if not datasets:
            raise ValueError(f"No valid shards found in {base_dir}")
        print(f"Created combined dataset from {len(datasets)} shards")
        return CombinedStreamingDataset(datasets=datasets, seed=self.seed, iterate_over_all=True)

    def train_dataloader(self) -> DataLoader:
        if self._has_sharded_structure(self.data_path_train):
            train_dataset = self._create_combined_dataset(self.data_path_train)
        else:
            train_dataset = StreamingDataset(
                input_dir=self.data_path_train,
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=True,
                drop_last=True,
            )
        return StreamingDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        val_dataset = StreamingDataset(
            input_dir=self.data_path_val,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
            drop_last=True,
        )
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def prepare_data(self) -> None:
        # The base version could be abstract here.
        raise NotImplementedError("prepare_data() must be implemented in the subclass.")


@dataclass
class DolminoDataset(BaseOlmoDataset):
    """
    Data module for the Dolmino dataset.
    This version is meant for training with an annealing schedule.
    """
    data_split: str = "default"  # if there are different splits for Dolmino, adjust as needed

    def prepare_data(self) -> None:
        from datasets import load_dataset
        from litdata import optimize
        from itertools import islice
        from tqdm import tqdm

        if Path(self.data_path_train).is_dir() and Path(self.data_path_val).is_dir():
            print(
                f"Found Olmo train and val dir: {self.data_path}. Skipping preprocessing."
            )
            return

        hf_cache_dir = os.getenv("HF_HOME", None)
        print(f"Using Dolmino dataset from {self.data_path}")
        # Example: load your dolmino dataset (adjust dataset name and parameters as needed)
        dataset = load_dataset("allenai/dolmino-mix-1124", name=self.data_split, cache_dir=hf_cache_dir, split="train", streaming=True)

        print("Loaded Dolmino dataset.")

        # shuffle
        print("Shuffling Dolmino dataset...")
        dataset = dataset.shuffle(seed=self.seed, buffer_size=10000)
        print("Shuffled Dolmino dataset.")

        dataset_list = list(islice(dataset, 10_010_000))
        split_dataset = {
            "train": dataset_list[:10_000_000],
            "val": dataset_list[10_000_000:],
        }

        if self.tokenizer is None:
            raise ValueError("Tokenizer is not set. Please call connect() first.")

        # Split dataset into training and validation sets
        # split_dataset = dataset.train_test_split(
        #     test_size=self.val_split_fraction, seed=self.seed, shuffle=True
        # )
        # split_dataset["val"] = split_dataset.pop("test")

        # # Optionally restrict for fast development runs:
        # split_dataset["train"] = split_dataset["train"].select(range(10000))
        # split_dataset["val"] = split_dataset["val"].select(range(1000))

        # If annealing is required, you might want to modify the tokenization process here.
        # For example, you could schedule different token dropout or add a temperature parameter.
        print("Preprocessing Dolmino train split...")
        optimize(
            fn=partial(process_and_tokenize, split_dataset["train"], self.tokenizer),
            inputs=list(range(len(split_dataset["train"]))),
            output_dir=self.data_path_train,
            num_workers=self.num_workers,
            chunk_bytes="200MB",
            fast_dev_run=self.fast_dev_run,
        )
        print("Preprocessing Dolmino val split...")
        optimize(
            fn=partial(process_and_tokenize, split_dataset["val"], self.tokenizer),
            inputs=list(range(len(split_dataset["val"]))),
            output_dir=self.data_path_val,
            num_workers=self.num_workers,
            chunk_bytes="200MB",
            fast_dev_run=self.fast_dev_run,
        )
        print(f"Finished preprocessing of Dolmino data at {self.data_path}")


@dataclass
class Olmo2Dataset(BaseOlmoDataset):
    """
    Data module for the main Olmo2 dataset.
    This version is structured for the main dataset (non-annealing).
    """
    data_split: str = "default"  # adjust as needed for the main dataset

    def prepare_data(self) -> None:
        from datasets import load_dataset
        from litdata import optimize
        from itertools import islice
        from tqdm import tqdm

        if Path(self.data_path_train).is_dir() and Path(self.data_path_val).is_dir():
            print(
                f"Found Olmo train and val dir: {self.data_path}. Skipping preprocessing."
            )
            return

        hf_cache_dir = os.getenv("HF_HOME", None)
        print(f"Using Olmo2 dataset from {self.data_path}")
        # Load the Olmo2 dataset (ensure the dataset name and parameters match your source)
        dataset = load_dataset("allenai/olmo-mix-1124", name=self.data_split, cache_dir=hf_cache_dir, split="train", streaming=True)
        print("Loaded Olmo2 dataset.")

        # shuffle
        print("Shuffling Olmo2 dataset...")
        dataset = dataset.shuffle(seed=self.seed, buffer_size=10000)
        print("Shuffled Olmo2 dataset.")

        dataset_list = list(islice(dataset, 10_010_000))
        split_dataset = {
            "train": dataset_list[:10_000_000],
            "val": dataset_list[10_000_000:],
        }
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not set. Please call connect() first.")

        # Split the dataset for training and validation
        # split_dataset = dataset.train_test_split(
        #     test_size=self.val_split_fraction, seed=self.seed, shuffle=True
        # )
        # split_dataset["val"] = split_dataset.pop("test")

        # # Optionally restrict example counts for development:
        # split_dataset["train"] = split_dataset["train"].select(range(10000))
        # split_dataset["val"] = split_dataset["val"].select(range(1000))

        print("Preprocessing Olmo2 train split...")
        optimize(
            fn=partial(process_and_tokenize, split_dataset["train"], self.tokenizer),
            inputs=list(range(len(split_dataset["train"]))),
            output_dir=self.data_path_train,
            num_workers=self.num_workers,
            chunk_bytes="200MB",
            fast_dev_run=self.fast_dev_run,
        )
        print("Preprocessing Olmo2 val split...")
        optimize(
            fn=partial(process_and_tokenize, split_dataset["val"], self.tokenizer),
            inputs=list(range(len(split_dataset["val"]))),
            output_dir=self.data_path_val,
            num_workers=self.num_workers,
            chunk_bytes="200MB",
            fast_dev_run=self.fast_dev_run,
        )
        print(f"Finished preprocessing of Olmo2 data at {self.data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess and check pretraining data."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Base directory where data is or will be stored.",
        required=True
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        help="Select dataset type: 'dolmino' for annealing version or 'olmo2' for main dataset.",
        choices=["dolmino", "olmo2"],
        default="olmo2",
    )
    parser.add_argument("--data_split", type=str, help="Data split (if applicable)")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run a quick development mode")
    args = parser.parse_args()

    # Choose the proper dataset module based on the command-line argument.
    if args.dataset_type == "dolmino":
        data_module = DolminoDataset(
            data_path=args.data_path,
            data_split=args.data_split if args.data_split else "default",
            fast_dev_run=args.fast_dev_run,
        )
    else:
        data_module = Olmo2Dataset(
            data_path=args.data_path,
            data_split=args.data_split if args.data_split else "default",
            fast_dev_run=args.fast_dev_run,
        )

    # When using the dataset module, ensure that you connect your tokenizer first.
    # For example:
    # tokenizer = Tokenizer(...your tokenizer args...)
    # data_module.connect(tokenizer=tokenizer, batch_size=1, max_seq_length=2048)

    data_module.prepare_data()
