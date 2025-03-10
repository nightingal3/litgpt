#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=sanity_check_myenv
#SBATCH --gres=gpu:6000Ada:8                
#SBATCH --output=sanitycheck_%J.out
#SBATCH --time=3:00:00
#SBATCH --mem=200G
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email at job start, end, and failure
#SBATCH --mail-user=emmy@cmu.edu  # Replace with your email address
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --exclude=babel-4-5

# Usage
# cd Lightning-Pretrain
# conda activate myenv
# sbatch scripts/pretrain_decay.sh

#export NCCL_P2P_DISABLE=1
echo "after torch upgrade to 2.4.1, nvidia-cudnn, nvjitlink, and lightning-utilities upgrade, comment out p2p disable"

method=random
resume=10000

echo -e "\n=== GPU Information ==="
echo "Available GPUs:"
nvidia-smi -L
echo -e "\nGPU Topology:"
nvidia-smi topo -m

set -a
source /data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/configs/.env
set +a

source ${MINICONDA_PATH}
conda activate llm_env_dev_copy

srun python -m litgpt pretrain OLMo2-7B-hf-stage2 \
  --data FineWebDataset \
  --data.num_workers 1 \
  --tokenizer_dir /data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/litgpt/checkpoints/allenai/OLMo-2-1124-7B \
  --data.data_path  ./fineweb_sample/10BT \
  --data.data_split "sample-10BT" \
  --train.save_interval 10 \
  --train.micro_batch_size 1 \
  --train.lr_scheduler decay \
  --train.log_interval 1 \
  --eval.interval 10 \
  --out_dir ./testing \
  --seed 1337 \
  --train.max_additional_steps 10