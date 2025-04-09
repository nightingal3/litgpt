import argparse
import os
import shutil
import tempfile
from functools import partial
from itertools import islice
from pathlib import Path
from tqdm import tqdm

from datasets import load_dataset
from litdata import optimize
from litgpt import Tokenizer

# Use the same tokenization function as in your class-based code.
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
        print(f"Error processing index {idx}: {e}")

def setup_directories(job_id: str):
    """Set up and clean temporary and cache directories for a specific job."""
    temp_dir = f"/scratch/nightingal3/tmp_job{job_id}"
    cache_dir = f"/scratch/nightingal3/tmp/huggingface_cache_job{job_id}"
    
    # Remove the directories if they already exist.
    for dir_path in [temp_dir, cache_dir]:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Cleaned up existing directory: {dir_path}")
            except Exception as e:
                print(f"Error cleaning {dir_path}: {e}")
    
    # Create fresh directories.
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set environment variables.
    os.environ["HF_HOME"] = cache_dir
    os.environ["TMPDIR"] = temp_dir
    os.environ["TEMP"] = temp_dir
    os.environ["TMP"] = temp_dir
    tempfile.tempdir = temp_dir
    
    print(f"Set up fresh directories for job {job_id}:")
    print(f"Temp dir: {temp_dir}")
    print(f"Cache dir: {cache_dir}")
    
    return cache_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone data optimization script with chunked processing using islice."
    )
    parser.add_argument("--data_path", type=str, default=".", help="Output directory for optimized data")
    parser.add_argument("--data_split", type=str, default="default", help="Dataset configuration (if applicable)")
    parser.add_argument("--job_id", type=str, required=True, help="Unique identifier for this job")
    parser.add_argument("--chunk", type=int, default=1, help="Chunk index (1-10) to process (each chunk is 1M items)")
    args = parser.parse_args()

    if args.chunk < 1 or args.chunk > 10:
        raise ValueError("Chunk index must be between 1 and 10.")

    # Set up temporary and cache directories.
    cache_dir = setup_directories(args.job_id)
    temp_dir = f"/scratch/nightingal3/tmp_job{args.job_id}"
    os.environ["TMPDIR"] = temp_dir
    os.environ["TEMP"] = temp_dir
    os.environ["TMP"] = temp_dir
    tempfile.tempdir = temp_dir
    print(f"Temporary directory is: {tempfile.gettempdir()}")

    # Initialize the tokenizer.
    tokenizer = Tokenizer("/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/litgpt/checkpoints/allenai/OLMo-2-1124-7B")
    print("Tokenizer initialized.")

    hf_cache_dir = os.getenv("HF_HOME")
    
    # For training, load the streaming dataset, shuffle, then islice to get only the desired 1M examples.
    print("Loading training dataset in streaming mode...")
    train_dataset = load_dataset(
        "allenai/dolmino-mix-1124",
        name=args.data_split,
        cache_dir=hf_cache_dir,
        split="train",
        streaming=True
    )
    print("Training dataset loaded.")
    print("Shuffling training dataset...")
    train_dataset = train_dataset.shuffle(seed=42, buffer_size=10000)
    print("Training dataset shuffled.")

    # Calculate start and end index for the given chunk (each chunk is 1,000,000 examples).
    start_index = (args.chunk - 1) * 1_000_000 + 10000
    end_index = args.chunk * 1_000_000 + 10000
    print(f"Consuming training data: seeking to examples {start_index} to {end_index}...")
    current_train_chunk = [item for item in tqdm(islice(train_dataset, start_index, end_index),
                                                   total=(end_index - start_index),
                                                   desc=f"Consuming chunk {args.chunk}")]
    print(f"Finished consuming training chunk {args.chunk}: {len(current_train_chunk)} examples.")

    # For validation, load the streaming dataset separately and take a fixed small number (e.g., 10,000 examples).
    print("Loading validation dataset in streaming mode...")
    val_dataset = load_dataset(
        "allenai/olmo-mix-1124",
        name=args.data_split,
        cache_dir=hf_cache_dir,
        split="train",  # using the same split because original script splits off validation later
        streaming=True
    )
    print("Validation dataset loaded.")
    print("Shuffling validation dataset...")
    val_dataset = val_dataset.shuffle(seed=42, buffer_size=10000)
    print("Validation dataset shuffled.")
    print("Consuming first 10,000 examples for validation...")
    current_val = [item for item in tqdm(islice(val_dataset, 10_000),
                                         total=10_000,
                                         desc="Consuming validation data")]
    print(f"Finished consuming validation data: {len(current_val)} examples.")

    output_dir = Path(args.data_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process the training chunk.
    train_chunk_output = f"{args.data_path}/train/chunk_{args.chunk}"
    Path(train_chunk_output).mkdir(parents=True, exist_ok=True)

    print(f"Optimizing training data for chunk {args.chunk} (processing {len(current_train_chunk)} examples)...")
    optimize(
        fn=partial(process_and_tokenize, current_train_chunk, tokenizer),
        inputs=list(range(len(current_train_chunk))),
        output_dir=train_chunk_output,
        num_workers=8,
        chunk_bytes="200MB",
        fast_dev_run=False
    )

    # Process the validation data.
    print("Optimizing validation data...")
    optimize(
        fn=partial(process_and_tokenize, current_val, tokenizer),
        inputs=list(range(len(current_val))),
        output_dir=f"{args.data_path}/val",
        num_workers=8,
        chunk_bytes="200MB",
        fast_dev_run=False
    )
    print("Data optimization complete.")
