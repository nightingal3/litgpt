import json
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, random_split

from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer


def load_split(json_path: Path) -> List[Dict[str, Any]]:
    if json_path.suffix == ".json":
        return json.loads(json_path.read_text())
    elif json_path.suffix == ".jsonl":
        return [json.loads(line) for line in json_path.open()]
    else:
        raise ValueError(f"Unsupported file format: {json_path.suffix}")


@dataclass
class MultiJSON(DataModule):
    json_path: Path
    mask_prompt: bool = False
    prompt_style: Union[str, PromptStyle] = "alpaca"
    ignore_index: int = -100
    seed: int = 42
    num_workers: int = 4

    # internal
    tokenizer: Optional[Tokenizer] = field(default=None, init=False)
    batch_size: int = field(default=1, init=False)
    max_seq_length: int = field(default=-1, init=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False)
    val_datasets: Dict[str, SFTDataset] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)
        if not self.json_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.json_path}")

    def connect(self, tokenizer: Tokenizer, batch_size: int, max_seq_length: Optional[int] = None) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def setup(self, stage: str = "") -> None:
        # Load train.json
        train_file = self.json_path / "train.json" if self.json_path.is_dir() else self.json_path
        train_data = load_split(train_file)
        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        # Load all val_*.json/jsonl under directory or single val
        val_files = []
        if self.json_path.is_dir():
            for p in sorted(self.json_path.glob("val*.json")) + sorted(self.json_path.glob("val*.jsonl")):
                val_files.append(p)
        else:
            # if single file, require val_split_fraction already handled by parent
            return
        for vf in val_files:
            name = vf.stem  # e.g. 'val_c4', 'val_sft'
            data = load_split(vf)
            ds = SFTDataset(
                data=data,
                tokenizer=self.tokenizer,
                prompt_style=self.prompt_style,
                max_seq_length=self.max_seq_length,
                mask_prompt=self.mask_prompt,
                ignore_index=self.ignore_index,
            )
            self.val_datasets[name] = ds

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length,
                ignore_index=self.ignore_index
            ),
        )

    def val_dataloaders(self) -> Dict[str, DataLoader]:
        loaders = {}
        for name, ds in self.val_datasets.items():
            loaders[name] = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=get_sft_collate_fn(
                    max_seq_length=self.max_seq_length,
                    ignore_index=self.ignore_index
                ),
            )
        return loaders
