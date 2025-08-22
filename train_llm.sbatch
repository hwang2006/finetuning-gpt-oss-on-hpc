#!/bin/bash
#SBATCH --comment=tensorflow
##SBATCH --partition=mig_amd_a100_4
#SBATCH --partition=amd_a100nv_8
##SBATCH --partition=eme_h200nv_8
#SBATCH --time=4:00:00        # walltime
#SBATCH --nodes=1             # the number of nodes
#SBATCH --ntasks-per-node=2   # number of tasks per node
#SBATCH --gres=gpu:2          # number of gpus per node
#SBATCH --cpus-per-task=8     # number of cpus per task
#SBATCH -o slurm-%j.out       # Stdout+stderr to file (job ID in name)

set -euo pipefail

# (Optional) load the module so the *host* has `singularity` in PATH
#module load singularity/4.1.0 2>/dev/null || true

# Repo + paths (host-side)
REPO="${REPO:-/scratch/$USER/finetuning-gpt-oss-on-hpc}"
SIF="${SIF:-/scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif}"
VENV="${VENV:-/scratch/$USER/finetuning-gpt-oss-on-hpc/venv}"

cd "$REPO"

# Fast caches on scratch (visible to host and bind-mounted into container)
export HF_HOME=${HF_HOME:-/scratch/$USER/.huggingface}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/scratch/$USER/.cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/scratch/$USER/.cache/pip}
export TMPDIR=${TMPDIR:-/scratch/$USER/tmp}
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

# Minimal locale + tokenizer noise
export LC_ALL=C LANG=C
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "== Node: $(hostname)"
echo "GPUs requested: ${SLURM_GPUS_ON_NODE:-unset}"
echo "CUDA_VISIBLE_DEVICES (host): ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Singularity: $(command -v singularity || echo 'not found on host')"

# Call the host-side wrapper (it will `singularity exec` for you)
./run_train.sh \
  --sif "$SIF" \
  --venv "$VENV" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset yahma/alpaca-cleaned --split 'train[:1%]' \
  --multi-gpu 1 \
  --max-seq-len 4096 --packing 1 --4bit 1 \
  --bs 1 --ga 16 --epochs 1 --lr 1e-4 \
  --log-steps 10 --save-steps 500
