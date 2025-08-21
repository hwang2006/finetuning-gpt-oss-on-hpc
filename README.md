# Finetuning GPT-OSS on a Supercomputer (SLRUM + Singularity)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.9](https://img.shields.io/badge/CUDA-12.9-brightgreen.svg)]()
[![Singularity](https://img.shields.io/badge/Container-Singularity%2FApptainer-indigo.svg)]()

This repository contains **end-to-end recipes to _finetune_ and _infer_ GPT-OSS and similar HF models on HPC systems** using **Singularity/Apptainer only**. All steps (env setup, inference, training, and upload to Hugging Face) run **inside the container**.

> These instructions work with **Singularity _or_ Apptainer**. On many clusters, `singularity` is a symlink to Apptainer—commands below work unchanged.

Tested on multi-GPU nodes (A100/H200) with the **PyTorch 2.8.0 CUDA 12.9 devel** docker image.

> **Repo name:** `finetuning-gpt-oss-on-supercomputer`

---

## Table of Contents

- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [1) Pull the Container](#1-pull-the-container)
- [2) Create a Python venv **inside** the container](#2-create-a-python-venv-inside-the-container)
- [3) Sanity Check (inside the container)](#3-sanity-check-inside-the-container)
- [4) Inference (inside the container)](#4-inference-inside-the-container)
  - [Single GPU](#single-gpu)
  - [Multi-GPU](#multi-gpu)
  - [Streaming & Long Outputs](#streaming--long-outputs)
- [5) Training (inside the container)](#5-training-inside-the-container)
  - [Single-GPU (wrapper)](#single-gpu-wrapper)
  - [Multi-GPU (wrapper)](#multi-gpu-wrapper)
  - [Flexible trainer with torchrun](#flexible-trainer-with-torchrun)
  - [SLURM batch template (example)](#slurm-batch-template-example)
- [6) Upload LoRA Adapter to Hugging Face (inside the container)](#6-upload-lora-adapter-to-hugging-face-inside-the-container)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Quick Start

> **Prereqs**: NVIDIA GPU node (Ampere/Hopper), outbound HTTPS to `huggingface.co`, and Singularity or Apptainer.

```bash
# login node
cd /scratch/$USER
git clone https://github.com/hwang2006/finetuning-gpt-oss-on-hpc.git
cd finetuning-gpt-oss-on-supercomputer

# (Optional) On SLURM clusters, get an interactive **GPU** shell first (login nodes usually have no GPUs):
# (replace <account> and <gpu-partition> for your site)
srun -A <account> -p <gpu-partition> --gres=gpu:1 --pty bash

# 1) Pull container (once)
singularity pull pt-2.8.0-cu129-devel.sif docker://ghcr.io/pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

# 2) Everything below runs IN the container
export SIF=$PWD/pt-2.8.0-cu129-devel.sif
# Tip: pass --env LC_ALL=C.UTF-8 --env LANG=C.UTF-8 to avoid startup locale warnings.
singularity exec --nv --env LC_ALL=C.UTF-8 --env LANG=C.UTF-8 "$SIF" bash -lc 'echo "CUDA ok"; nvidia-smi | head -10'
```

---

## Repository Layout

```
.
├── run_infer.sh              # Singularity/Apptainer-first chat/infer wrapper (streaming supported)
├── run_train.sh              # Single/Multi-GPU training wrapper (torchrun inside)
├── infer_unsloth.py          # Direct Unsloth inference
├── infer_with_peft_v4.py     # Inference with saved LoRA adapter (PEFT)
├── train_unsloth.py          # Minimal SFT example (single GPU)
├── train_unsloth_flex.py     # Flexible SFT (packing, JSONL/HF datasets, torchrun-ready)
├── upload_lora_to_hf.py      # Push LoRA + tokenizer to Hugging Face
└── README.md
```

---

## 1) Pull the Container

```bash
# One-time pull to your scratch (recommended)
export SINGULARITY_TMPDIR=/scratch/qualis/.singularity/tmp
export SINGULARITY_CACHEDIR=/scratch/qualis/.singularity

mkdir -p /scratch/$USER/sifs
singularity pull /scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif docker://ghcr.io/pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
```

> **Why not "runtime" but “devel” image?** Unsloth compiles Triton kernels at first run; the devel image already includes GCC & build tools, avoiding host bind gymnastics.

> **GPU architectures:** CUDA 12.8/12.9 builds target **Volta+** (V100), **Ampere** (A100) and **Hopper** (H100/H200).

---

## 2) Create a Python venv **inside** the container

```bash
# Create venv on scratch and install deps INSIDE the container
export VENV=/scratch/$USER/finetuning-gpt-oss-on-hpc/venv
export SIF=/scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif

# (Recommended) set caches to scratch and enable HF accelerated transfers
export HF_HOME=/scratch/$USER/.huggingface
export XDG_CACHE_HOME=/scratch/$USER/.cache
export PIP_CACHE_DIR=/scratch/$USER/.cache/pip
export TMPDIR=/scratch/$USER/tmp
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

singularity exec --nv --env LC_ALL=C.UTF-8 --env LANG=C.UTF-8 "$SIF" bash -lc '
  set -e
  python3 -V
  python3 -m venv "'"$VENV"'"
  source "'"$VENV"'/bin/activate"
  pip install -q --upgrade pip
  # Unsloth + zoo from PyPI (recommended)
  pip install -q "unsloth[base]" "unsloth_zoo[base]"
  # Optional: faster Hub I/O on fat pipes
  pip install -q "huggingface_hub[hf_transfer]>=0.24.0"
  echo "✅ venv ready @ '"$VENV"'"
'

# Optional: enable accelerated Hub transfers
export HF_HUB_ENABLE_HF_TRANSFER=1
```

**Notes**
- If your site has strict egress, pre-stage models into `$HF_HOME` or mirror them internally.
- If PyPI versions momentarily drift, you can fall back to Git installs (inside the container):
  ```bash
  pip uninstall -y unsloth unsloth_zoo
  pip install -q --no-deps git+https://github.com/unslothai/unsloth_zoo.git
  pip install -q --no-deps git+https://github.com/unslothai/unsloth.git
  ```

---

## 3) Sanity Check (inside the container)

> **Import order matters:** import **Unsloth first**, then `torch`/`transformers`.

```bash
singularity exec --nv --env LC_ALL=C.UTF-8 --env LANG=C.UTF-8 "$SIF" bash -lc '
  export LC_ALL=C.UTF-8 LANG=C.UTF-8
  source "'"$VENV"'/bin/activate"
  python - <<PY
import unsloth, unsloth_zoo
import torch, transformers
print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU OK:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("unsloth:", unsloth.__version__, "| unsloth_zoo:", unsloth_zoo.__version__)
print("transformers:", transformers.__version__)
PY
'
```

---

## 4) Inference (inside the container)

> **Tip:** `./run_infer.sh --help` lists all options. To avoid surprises, pass your paths explicitly (`--sif`, `--venv`, `--pyfile`).

### Single GPU
```bash
export SIF=/scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif
export HF_HOME=/scratch/$USER/.huggingface   # optional but recommended

# Minimal example (Qwen 0.5B)
./run_infer.sh \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --user "Tell me a fun space fact in one sentence."
```

### Multi GPU
> For >7B models or low VRAM cards, you can shard across GPUs.
```bash
./run_infer.sh \
  --multi-gpu 1
  --model Qwen/Qwen2.5-7B-Instruct \
  --user "Summarize the causes of the aurora borealis."
```

### Streaming & Long Outputs
```bash
# Stream tokens as they generate
./run_infer.sh \
  --model openai/gpt-oss-20b \
  --stream 1 \
  --max-new 3200 --max-seq-len 8192 \
  --system "You write long-form, comprehensive answers. Produce at least 2,000 words." \
  --user "Explain quantum computing in detail (>2000 words)."
```

## 5) Training (inside the container)

> **Tip:** `./run_train.sh --help` lists all options (LoRA, seq len, BS/GA/LR, etc.).

### Single-GPU (wrapper)

```bash
singularity exec --nv --env LC_ALL=C.UTF-8 --env LANG=C.UTF-8 "$SIF" bash -lc '
  export LC_ALL=C.UTF-8 LANG=C.UTF-8
  source "'"$VENV"'/bin/activate"
  ./run_train.sh \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset yahma/alpaca-cleaned \
    --out /scratch/$USER/test/unsloth-out
'
```

*(Or use a local JSONL instead of a Hub dataset: `--jsonl /path/data.jsonl`)*

### Multi-GPU (wrapper)

```bash
singularity exec --nv --env LC_ALL=C.UTF-8 --env LANG=C.UTF-8 "$SIF" bash -lc '
  export LC_ALL=C.UTF-8 LANG=C.UTF-8
  source "'"$VENV"'/bin/activate"
  ./run_train.sh \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset yahma/alpaca-cleaned \
    --gpus 0,1 --multi-gpu 1 \
    --out /scratch/$USER/test/unsloth-out-7b
'
```

### Flexible trainer with torchrun

Use the flexible trainer (`train_unsloth_flex.py`) for more control:

```bash
# Example: Qwen 7B, Alpaca 1% split, 2 GPUs data parallel
singularity exec --nv --env LC_ALL=C.UTF-8 --env LANG=C.UTF-8 "$SIF" bash -lc '
  export LC_ALL=C.UTF-8 LANG=C.UTF-8
  source "'"$VENV"'/bin/activate"
  MODEL_ID="Qwen/Qwen2.5-7B-Instruct" \
  DATASET="yahma/alpaca-cleaned" DATASET_SPLIT="train[:1%]" \
  OUTPUT_DIR="/scratch/$USER/test/unsloth-out-7b" \
  MAX_SEQ_LEN=4096 USE_4BIT=1 PACKING=1 \
  BATCH_SIZE=1 GRAD_ACCUM=16 EPOCHS=1 LR=1e-4 LOG_STEPS=10 SAVE_STEPS=500 \
  torchrun --standalone --nproc_per_node=2 train_unsloth_flex.py
'
```

> **Note:** `--nproc_per_node` should match the number of GPUs you intend to use. `torchrun` spawns 1 process per GPU.

### SLURM batch template (example)

This template trains on **all GPUs allocated** to the job:

```bash
#!/bin/bash
#SBATCH -J sft-qwen7b
#SBATCH -A <account>
#SBATCH -p <gpu-partition>
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 04:00:00
set -euo pipefail

SIF="$PWD/pt-2.8.0-cu129-devel.sif"
VENV="/scratch/$USER/test/venv"

export HF_HOME=/scratch/$USER/.huggingface
export XDG_CACHE_HOME=/scratch/$USER/.cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

singularity exec --nv --env LC_ALL=C.UTF-8 --env LANG=C.UTF-8 "$SIF" bash -lc "
  export LC_ALL=C.UTF-8 LANG=C.UTF-8
  source '$VENV/bin/activate'
  MODEL_ID='Qwen/Qwen2.5-7B-Instruct' \
  DATASET='yahma/alpaca-cleaned' DATASET_SPLIT='train[:1%]' \
  OUTPUT_DIR='/scratch/$USER/test/unsloth-out-7b' \
  MAX_SEQ_LEN=4096 USE_4BIT=1 PACKING=1 \
  BATCH_SIZE=1 GRAD_ACCUM=16 EPOCHS=1 LR=1e-4 LOG_STEPS=10 SAVE_STEPS=500 \
  torchrun --standalone --nproc_per_node=\$SLURM_GPUS_ON_NODE train_unsloth_flex.py
"
```

---

## 6) Upload LoRA Adapter to Hugging Face (inside the container)

```bash
# Set where your adapter lives and your target repo
export ADAPTER_DIR=/scratch/$USER/test/unsloth-out-7b
export REPO_ID=hwang2006/qwen2.5-7b-alpaca-1pct-lora
export HF_TOKEN=hf_********************************

singularity exec --nv --env LC_ALL=C.UTF-8 --env LANG=C.UTF-8 "$SIF" bash -lc '
  export LC_ALL=C.UTF-8 LANG=C.UTF-8
  source "'"$VENV"'/bin/activate"
  pip install -q "huggingface_hub[hf_transfer]>=0.24.0"
  python upload_lora_to_hf.py \
    --adapter-dir "'"$ADAPTER_DIR"'" \
    --repo-id "'"$REPO_ID"'" \
    --token "'"$HF_TOKEN"'" \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --datasets yahma/alpaca-cleaned \
    --language en \
    --tags lora,unsloth,peft,qwen,instruction-tuning \
    --ignore checkpoint-* \
    --license apache-2.0
'
```

This script:
- Fixes `adapter_config.json` fields if needed
- Generates a README model card
- Creates the repo (public by default) and uploads `adapter_model.safetensors`, tokenizer files, and metadata.
- Uses `hf_transfer` for faster uploads when available.

---

## Troubleshooting

- **No GPU on login:** use `srun -A <account> -p <gpu-partition> --gres=gpu:1 --pty bash` first.
- **Locale warning at shell startup:** prepend `--env LC_ALL=C.UTF-8 --env LANG=C.UTF-8` to every `singularity exec` call that launches `bash`.
- **First-run compile (Triton/GCC):** the **devel** image includes compilers; no extra bind mounts required.
- **Tokenizers “fork” warning:** harmless under `torchrun`; set `TOKENIZERS_PARALLELISM=false` to silence.
- **Very slow long generations:** enable `--stream 1` in `run_infer.sh` to see tokens as they’re produced.
- **Adapter mismatch:** ensure the base model matches what the adapter was trained on (same family/shape). See the “Adapter ↔ Base compatibility” note.
- **CUDA arch support:** CUDA 12.8/12.9 builds target Volta/Ampere/Hopper. Very old GPUs may be unsupported.
- **Hub auth:** gated/private models or pushing to private repos require `HF_TOKEN` or a prior `huggingface-cli login`.
- **Slow downloads/uploads:** `pip install "huggingface_hub[hf_transfer]"` and `export HF_HUB_ENABLE_HF_TRANSFER=1`.
- **Multi-GPU training:** `--nproc_per_node` = number of GPUs; `torchrun` launches 1 process/GPU.

---

## License

MIT — see [LICENSE](LICENSE). If you use this work in research or production, a citation or a star ⭐ is appreciated!
