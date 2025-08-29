# Finetuning GPT-OSS on a Supercomputer (SLURM + Singularity)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.9](https://img.shields.io/badge/CUDA-12.9-brightgreen.svg)]()
[![Singularity](https://img.shields.io/badge/Container-Singularity-indigo.svg)](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html)
[![Unsloth](https://img.shields.io/badge/Unsloth-2025.x-orange.svg)](https://github.com/unslothai/unsloth)

This repository contains **end-to-end recipes to _finetune_ and _infer_ GPT‑OSS and similar HF models on HPC systems** using **[Unsloth](https://github.com/unslothai/unsloth) inside Singularity/Apptainer containers**. 

- Tested on H200/A100/V100 multi-GPU nodes with the **PyTorch 2.8.0 • CUDA 12.9 devel** image.
- All steps (venv setup, training, inference, and HF upload) run inside the container.
- Compatible with Singularity or Apptainer (on many clusters singularity is a symlink to Apptainer).

**Repo:** <https://github.com/hwang2006/finetuning-gpt-oss-on-hpc>

---

## Table of Contents

- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Container Build Guide](#container-build-guide)
- [1) Pull the Container](#1-pull-the-container)
- [2) Create a Python venv **inside** the container](#2-create-a-python-venv-inside-the-container)
- [3) Sanity Check](#3-sanity-check)
- [4) Inference](#4-inference)
- [5) Training](#5-training)
- [6) Upload LoRA Adapter to Hugging Face](#6-upload-lora-adapter-to-hugging-face)
- [7) Inference with LoRA Adapters (after training/upload)](#7-inference-with-lora-adapters-after-trainingupload)
- [Version Pinning & Compatibility](#version-pinning--compatibility)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Quick Start

> **Prereqs**: NVIDIA GPU node (Ampere/Hopper advised), outbound HTTPS to `huggingface.co`, and Singularity or Apptainer.

```bash
# login node
cd /scratch/$USER
git clone https://github.com/hwang2006/finetuning-gpt-oss-on-hpc.git
cd finetuning-gpt-oss-on-hpc

# (Optional) SLURM interactive GPU shell
# srun -A <account> -p <gpu-partition> --gres=gpu:1 --pty bash

# Scratch-backed cache so you don't fill $HOME
export SIFDIR=/scratch/$USER/sifs
export SINGULARITY_CACHEDIR=/scratch/$USER/.singularity
export SINGULARITY_TMPDIR=/scratch/$USER/.singularity/tmp
mkdir -p "$SIFDIR" "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

# 1) Pull container (ORAS/OCI) 
singularity pull "$SIFDIR/pt-2.8.0-cu129-devel.sif" \
  oras://ghcr.io/hwang2006/pt-2.8.0-cu129-devel:1.0


# 2) Everything below runs IN the container
export SIF="$SIFDIR/pt-2.8.0-cu129-devel.sif"
singularity exec --nv "$SIF" bash -lc '
  echo "CUDA OK"; nvidia-smi | head -10
'
```
> Need 4-bit or mixing GPUs? See the [compatibility matrix](#compatibility-matrix--gpus--transformers--4-bit-behavior).

---
## Repository Layout

```
.
├── run_infer.sh              # Singularity-first chat/infer wrapper (streaming supported)
├── run_train.sh              # Single/Multi-GPU training wrapper (torchrun inside)
├── infer_unsloth.py          # Direct Unsloth inference
├── infer_with_peft.py        # Inference with saved LoRA adapter (PEFT)
├── train_unsloth.py          # Minimal SFT example
├── train_unsloth_flex.py     # Flexible SFT (packing, JSONL/HF datasets, torchrun-ready)
├── upload_lora_to_hf.py      # Push LoRA + tokenizer to Hugging Face
├── pt-2.8.0-cu129-devel.def  # Singularity definition file (build your own container)
├── HPC_CONTAINER_BUILD.md    # Step-by-step container build guide (login/compute node workflow)
└── README.md
```

---

## Container Build Guide

If you want to build the container yourself (instead of pulling from GitHub Container Registry (GHCR)), see [HPC_CONTAINER_BUILD.md](HPC_CONTAINER_BUILD.md).

That guide explains:
- Why a **sandbox build** is first needed (to install `git`, `wget`, etc. missing from the upstream PyTorch image).  
- Why the build typically uses **`/tmp`** on login nodes (local SSD avoids metadata errors on shared filesystems).  
- How to use **fakeroot** on the login node, then move the sandbox to `/scratch` and finalize the `.sif` packaging on a compute node (without fakeroot).  

> **Important:** You need `git` inside the container because later steps (`pip install` in venv) fetch packages directly from GitHub.

---

## 1) Pull the Container

You have **two choices**:  

### Option A — Pull the **prebuilt** image from GitHub Container Registry (GHCR) (recommended)

```bash
singularity pull "$SIFDIR/pt-2.8.0-cu129-devel.sif" \
      oras://ghcr.io/hwang2006/pt-2.8.0-cu129-devel:1.0
```

> This is the **customized image** I built and uploded to GHCR. It include:
> - `git`, `curl`, `wget` installed  
> - Locale configured (`en_US.UTF-8`) → avoids `setlocale` warnings  
> - `/scratch`, `/home01`, `/apps` bind dirs created  
> - Timezone set (`Asia/Seoul`)  

>  You can also rebuild it yourself by using or customizing the `pt-2.8.0-cu129-devel.def` definition file if needed:
---

### Option B — Pull the **raw upstream image** from Docker Hub

```bash
singularity pull "$SIFDIR/pt-2.8.0-cu129-devel.sif" \
      docker://ghcr.io/pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
```

> **Caution:** This version is “bare-metal” and will require fixes:  
> - Install `git`, `curl`, etc. manually  
> - Configure `en_US.UTF-8` locales  
> - Clean `/var/lib/apt/lists/*` before `apt-get install`  

---

## 2) Create a Python venv **inside** the container

```bash
# create venv virtual environment on scratch; install deps INSIDE the container
export REPO=/scratch/$USER/finetuning-gpt-oss-on-hpc
export VENV="$REPO/venv"
export SIF=/scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif

# (Recommended) set caches to scratch and enable HF accelerated transfers
export HF_HOME=/scratch/$USER/.huggingface
export XDG_CACHE_HOME=/scratch/$USER/.cache
export PIP_CACHE_DIR=/scratch/$USER/.cache/pip
export TMPDIR=/scratch/$USER/tmp
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

singularity exec --nv "$SIF" bash -lc '
  set -e
  python3 -V
  python3 -m venv "'"$VENV"'"
  source "'"$VENV"'/bin/activate"
  pip install -q --upgrade pip
  pip install -q \
    "torch>=2.8.0" "triton>=3.4.0" numpy  \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth.git" \
    torchvision bitsandbytes \
    git+https://github.com/huggingface/transformers \
    git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels \
    kernels
  # Optional: faster Hub I/O
  pip install -q "huggingface_hub[hf_transfer]>=0.24.0"
  echo "✅ venv ready @ '"$VENV"'"
'

# check core packages versions 
singularity exec --nv "$SIF" bash -lc '
  source "'"$VENV"'/bin/activate"
  python -m pip list | grep -Ei "transformers|peft|bitsandbytes|kernels|unsloth"
'
bitsandbytes             0.47.0
kernels                  0.9.0
peft                     0.17.1
transformers             4.56.0.dev0
triton_kernels           1.0.0
unsloth                  2025.8.10
unsloth_zoo              2025.8.9

# Optional: enable accelerated Hub transfers globally
export HF_HUB_ENABLE_HF_TRANSFER=1
```

> **Note:** Import **Unsloth first**, then `torch`/`transformers`. This ensures proper Triton patching.

---

## 3) Sanity Check 

> **Import order matters:** import **Unsloth first**, then `torch`/`transformers`.

```bash
singularity exec --nv "$SIF" bash -lc '
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

## 4) Inference 

> **Tip:** `./run_infer.sh --help` lists all options. Pass your paths explicitly if needed (`--sif`, `--venv`, `--pyfile`).  
> **This section uses *base models only* (no adapters yet).** LoRA adapter inference appears later in §7.

```bash
./run_infer.sh --help
Usage: ./run_infer.sh [options]

Paths:
  --sif PATH                 SIF image (default: /scratch/qualis/sifs/pt-2.8.0-cu129-devel.sif)
  --work DIR                 Work dir (defaults venv/pyfile under it)
  --venv DIR                 Python venv path (default: /scratch/qualis/finetuning-gpt-oss-on-hpc/venv)
  --pyfile FILE              Path to infer script (default: /scratch/qualis/finetuning-gpt-oss-on-hpc/infer_unsloth.py)

Model & decoding:
  --model ID                 HF model id (default: Qwen/Qwen2.5-0.5B-Instruct)
  --max-seq-len N            (default: 4096)
  --max-new N                (default: 512)
  --sample 0|1               do_sample (default: 1)
  --temp F                   temperature (default: 0.7)
  --top-p F                  top_p (default: 0.9)

Multi-GPU:
  --gpus LIST                e.g. "0" or "0,1,2,3"; if unset, auto-detect
  --multi-gpu 0|1            enable sharding (default: 1)
  --device-map STR           auto | balanced_low_0 | cuda:0 (default: auto)
  --headroom GiB             per-GPU reserve (default: 2)

Streaming:
  --stream 0|1               print tokens as generated (default: 1)

Prompts:
  --system "TEXT"            system prompt
  --user "TEXT"              user prompt

Adapters (PEFT / LoRA):
  --adapter PATH|REPO        LoRA adapter dir or HF repo (auto-switches script)
  --base-model ID            Base model to pair with adapter (default: --model)
  --load-in-4bit 0|1         Load base in 4-bit (default: off)
Notes:
  * When --adapter is set, this wrapper auto-switches to infer_with_peft.py
    if found next to your script or under --work.
```

### Single GPU
```bash
# Minimal example (Qwen 0.5B)
./run_infer.sh \
  --multi-gpu 0 \
  --stream 0 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --user "Tell me a fun space fact in one sentence."
```

### Multi-GPU
> For >7B models or low VRAM, shard across GPUs.
```bash
./run_infer.sh \
  --multi-gpu 1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --user "Summarize the causes of the aurora borealis."
```

### Streaming & Long Outputs
```bash
# Stream tokens as they generate
./run_infer.sh \
  --model openai/gpt-oss-20b \
  --stream 1 \
  --max-new 1200 --max-seq-len 8192 \
  --system "You write long-form, comprehensive answers." \
  --user "Explain quantum computing for software engineers."
```

---

## 5) Training 

> **Tip:** `./run_train.sh --help` lists all options (LoRA, seq len, BS/GA/LR, etc.).

```bash
./run_train.sh --help
Usage: ./run_train.sh [options]

Paths:
  --sif PATH                 Singularity image (.sif) path
                             (default: /scratch/qualis/sifs/pt-2.8.0-cu129-devel.sif)
  --work DIR                 Work dir (venv + pyfile live here)
                             (default: /scratch/qualis/finetuning-gpt-oss-on-hpc)
  --venv DIR                 Python venv path (default: /scratch/qualis/finetuning-gpt-oss-on-hpc/venv)
  --pyfile FILE              Training Python file
                             (default: /scratch/qualis/finetuning-gpt-oss-on-hpc/train_unsloth_flex.py)

Model & Output:
  --model ID                 HF model id (default: Qwen/Qwen2.5-0.5B-Instruct)
  --out DIR                  Output dir for adapter/tokenizer
                             (default: /scratch/qualis/finetuning-gpt-oss-on-hpc/outputs/Qwen2.5-0.5B-Instruct-lora-YYYYMMDD-HHMMSS)

Datasets:
  --dataset NAME             HF dataset name (default: yahma/alpaca-cleaned)
  --split SPLIT              HF dataset split (default: train)
  --jsonl FILE               JSONL file (enables JSONL mode)
  --jsonl-prompt NAME        JSONL instruction field (default: instruction)
  --jsonl-input NAME         JSONL input/ctx field (default: input)
  --jsonl-response NAME      JSONL response field (default: output)

Prompting / Sequence:
  --system "TEXT"            System prompt used in chat template
                             (default: "You are a helpful, careful assistant.")
  --max-seq-len N            Max sequence length (default: 4096)
  --packing 0|1              Pack short examples (default: 1)

LoRA:
  --lora-r N                 LoRA rank r (default: 16)
  --lora-alpha N             LoRA alpha (default: 16)
  --lora-dropout F           LoRA dropout (default: 0.0)
  --lora-targets LIST        Comma-separated target modules
                             (default: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj)

Training:
  --bs N                     per-device batch size (default: 1)
  --ga N                     gradient accumulation steps (default: 8)
  --epochs F                 number of epochs (float OK) (default: 1.0)
  --lr F                     learning rate (default: 2e-4)
  --warmup F                 warmup ratio (default: 0.03)
  --wd F                     weight decay (default: 0.0)
  --log-steps N              logging steps (default: 10)
  --save-steps N             checkpoint save steps (default: 500)
  --eval-steps N             eval steps (0 disables) (default: 0)
  --seed N                   random seed (default: 42)
  --bf16 0|1                 enable bf16 if supported (default: 1)
  --gc 0|1                   gradient checkpointing (default: 1)
  --workers N                dataset num_proc (default: 4)
  --report STR               reporting: none|wandb|tensorboard (default: none)
  --save-limit N             max checkpoints to keep (default: 3)
  --4bit 0|1                 load base in 4-bit (QLoRA) (default: 1)

GPUs:
  --gpus LIST                e.g., "0" or "0,1,2,3"; if unset, auto-detect
                             (current: "<auto>")
  --multi-gpu 0|1            use torchrun across GPUs (default: 1)

Examples:
  # Single GPU, Alpaca 1%, 4-bit + packing
  ./run_train.sh --model Qwen/Qwen2.5-0.5B-Instruct --dataset yahma/alpaca-cleaned --split 'train[:1%]' \
     --gpus 0 --multi-gpu 0 --out ./out/s1-alpaca1 --bs 1 --ga 16 --epochs 1 --4bit 1 --packing 1

  # Two GPUs data parallel (torchrun), IMDB 2%, 4-bit, no packing
  ./run_train.sh --model Qwen/Qwen2.5-0.5B-Instruct --dataset imdb --split 'train[:2%]' \
     --gpus 0,1 --multi-gpu 1 --gc 0 --out ./out/dp2-imdb2 --bs 1 --ga 16 --epochs 1 --4bit 1 --packing 0
```

### Single GPU 

```bash
./run_train.sh \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset yahma/alpaca-cleaned \
  --multi-gpu 0 \
  --split "train[:2%]" \
  --out "$REPO/unsloth-out-0.5b"
```

*Or use a local JSONL instead of a Hub dataset: `--jsonl /path/data.jsonl`.*

### Multiple GPUs 

```bash
./run_train.sh \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset yahma/alpaca-cleaned \
  --multi-gpu 1 \
  --epochs 0.1 \
  --out "$REPO/unsloth-out-7b"
```

### Flexible trainer with torchrun

Use the flexible trainer (`train_unsloth_flex.py`) directly if you want for more control (packing, JSONL/HF datasets, etc.).

```bash
# Example: Qwen 7B, Alpaca 1% split, 2 GPUs data parallel
singularity exec --nv --cleanenv  "$SIF" bash -lc '
  source "'"$VENV"'/bin/activate"
  MODEL_ID="Qwen/Qwen2.5-7B-Instruct" \
  DATASET="yahma/alpaca-cleaned" DATASET_SPLIT="train[:2%]" \
  OUTPUT_DIR="'"$REPO"'/unsloth-out-7b-flex" \
  MAX_SEQ_LEN=4096 USE_4BIT=1 PACKING=1 \
  BATCH_SIZE=1 GRAD_ACCUM=16 EPOCHS=1 LR=1e-4 LOG_STEPS=10 SAVE_STEPS=500 \
  torchrun --standalone --nproc_per_node=2 train_unsloth_flex.py
'
```

> `--nproc_per_node` should match the number of GPUs you intend to use. `torchrun` spawns 1 process per GPU.

### SLURM batch template (example)

This script trains on all GPUs allocated to the job by calling the host wrapper `run_train.sh` (which itself launches the container). Save it as `train_llm.sbatch` and submit with `sbatch train_llm.sbatch`.

```bash
#!/bin/bash
#SBATCH --comment=pytorch
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
  --model openai/gpt-oss-20b \
  --dataset yahma/alpaca-cleaned --split 'train[:2%]' \
  --out "$REPO/unsloth-out-20b" \
  --multi-gpu 1 \
  --max-seq-len 4096 --packing 1 --4bit 1 \
  --bs 1 --ga 16 --epochs 1 --lr 1e-4 \
  --log-steps 10 --save-steps 500 \
  --out "$REPO/unsloth-out-20b"
```

### Quick submits
```bash
sbatch train_llm.sbatch
```

### Notes
- `run_train.sh` auto-detects visible GPUs and launches `torchrun` with one process per GPU (via `SLURM_GPUS_ON_NODE`).
- To override paths or knobs at submit time:
```bash
sbatch --gpus=2 \
  --export=ALL,SIF=/scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif,VENV=/scratch/$USER/finetuning-gpt-oss-on-hpc/venv \
  train_llm.sbatch
```
- Monitor progress:
```bash
squeue -u $USER
tail -f slurm-<jobid>.out
```
- Outputs (LoRA adapter, tokenizer, logs) land in the `--out` directory.

---

## 6) Upload LoRA Adapter to Hugging Face

```bash
# Where your adapter was saved and your target repo
export REPO=/scratch/$USER/finetuning-gpt-oss-on-hpc
export ADAPTER_DIR="$REPO/unsloth-out-20b"
export REPO_ID=hwang2006/gpt-oss-20b-alpaca-2pct-lora   # <your-username>/<repo-name>
export HF_TOKEN=hf_********************************    # a token with "Write" scope
export VENV="$REPO/venv"
export SIF=/scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif

singularity exec --nv "$SIF" bash -lc '
  source "'"$VENV"'/bin/activate"
  pip install -q "huggingface_hub[hf_transfer]>=0.24.0"
  python upload_lora_to_hf.py \
    --adapter-dir "'"$ADAPTER_DIR"'" \
    --repo-id "'"$REPO_ID"'" \
    --token "'"$HF_TOKEN"'" \
    --base-model openai/gpt-oss-20b \
    --datasets yahma/alpaca-cleaned \
    --language en \
    --tags lora,unsloth,peft,gpt-oss,fine-tuning \
    --ignore checkpoint-* \
    --license apache-2.0
'
```

**Target repo naming & permissions**

- Use your **Hugging Face username or org** and a short, descriptive repo name (for example):  
  `REPO_ID="yourname/qwen2.5-7b-alpaca-1pct-lora"`
- The script will **create the repo if it doesn’t exist** (public by default). Make it private later if desired.
- `HF_TOKEN` must have **write access** to that namespace (user/org).  
- Large files are pushed with **hf_transfer** when available for faster I/O.

This script:
- Fixes `adapter_config.json` fields if needed  
- Generates a README model card  
- Creates the repo (public by default) and uploads `adapter_model.safetensors`, tokenizer files, and metadata  
- Uses `hf_transfer` for faster uploads when available

---

## 7) Inference with LoRA Adapters (after training/upload)

> Now that you’ve trained/uploaded adapters, you can run adapter‑aware inference.  
> `run_infer.sh` will **auto‑switch** to `infer_with_peft.py` when `--adapter` is provided.

### Inference with a **local** LoRA adapter

```bash
./run_infer.sh \
  --adapter "$REPO/unsloth-out-20b" \
  --base-model openai/gpt-oss-20b \
  --load-in-4bit 1 --stream 1 \
  --user "Write a short, friendly paragraph about learning Python."
```

### Inference with an **HF‑hosted** LoRA adapter

```bash
./run_infer.sh \
  --adapter "hwang2006/gpt-oss-20b-alpaca-2pct-lora" \
  --base-model openai/gpt-oss-20b \
  --load-in-4bit 1 --stream 1 \
  --user "Suggest a heartwarming film and explain why in one sentence."
```

> On **Transformers 4.55.4**, the script will **fall back to bf16/fp16** if 4-bit quantization isn’t supported by that version. See the next section for the pinning strategy.
---
## Version Pinning & Compatibility
### Why pin at all?
Transformers **4.56–4.57** introduced a new quantization stack (AutoHfQuantizer). Mixing old/new quant paths caused errors like:
- `AttributeError: 'BitsAndBytesConfig' object has no attribute 'get_loading_attributes'`
- `AttributeError: 'Bnb4BitHfQuantizer' object has no attribute 'get_loading_attributes'`

This repo uses a pragmatic split:
- **Training**: relies on Unsloth + stable Transformers on your image; no hard pin in `run_train.sh`. The script auto-disables fragile paths (e.g., fused CE) when needed.
- **Inference**: `run_infer.sh` contains a **pin block** that prefers **Transformers 4.55.4** for Unsloth’s eager path **unless** you explicitly want 4-bit with the new API.
  - On 4.55.4, --load-in-4bit gracefully falls back to bf16/fp16.
  - If you truly need 4-bit via the new API, set up a **separate venv** pinned to `transformers>=4.56,<4.58` and use the adapter-aware `infer_with_peft.py` (which can build the new quantizer config). Some HPC mirrors don’t carry those wheels; your pin block will tell you if that’s the case.

### Check the active versions (inside the container):
```bash
singularity exec --nv "$SIF" bash -lc '
  source "'"$VENV"'/bin/activate"
  python -m pip list | grep -Ei "transformers|peft|bitsandbytes|unsloth"
'
```
### Compatibility Matrix — GPUs × Transformers × 4-bit behavior

| GPU (arch)     | SM | Preferred dtype | Transformers **4.55.4** (legacy path) | Transformers **4.56.0.dev0 (new quant) | 
|---|---:|---|---|---|
| **H200 / H100 (Hopper)** | 90 | **bf16** | ✅ Stable infer/train (bf16). `--load-in-4bit` ➜ **falls back** to bf16 with warning. | ✅ Recommended for true 4-bit via *AutoHfQuantizer* (separate venv). |
| **A100 (Ampere)**        | 80 | **bf16** | ✅ Stable infer/train (bf16). `--load-in-4bit` ➜ **falls back** to bf16 with warning. | ✅ Recommended for true 4-bit via *AutoHfQuantizer* (separate venv). |
| **V100 (Volta)**         | 70 | **fp16** | ✅ Stable infer/train (fp16). `--load-in-4bit` ➜ **falls back** to fp16 with warning. | ✅ 4-bit can work via *AutoHfQuantizer*; non-quant paths stay **fp16**. |

**Notes**
1) `run_infer.sh` pins to **Transformers 4.55.4** for reliable Unsloth eager inference. On this version, `--load-in-4bit` **gracefully falls back** to bf16/fp16.  
2) If you **need real 4-bit**, use a **separate venv** with `transformers>=4.56,<4.58` and run `infer_with_peft.py` (new *AutoHfQuantizer* path). Some HPC mirrors may not carry those wheels.  
3) Make sure `bitsandbytes` (e.g., 0.47.0) and Triton (≥3.4.0) are in the venv; otherwise you’ll see MXFP4 fallback messages.  
4) `run_train.sh` doesn’t hard-pin TF; it toggles safe DDP flags and only requires Flash-Attn 2 if `PACKING=1`.  
5) V100 lacks bf16 → use fp16 (scripts already choose this).  

---
## Troubleshooting
- **“PACKING=1 requested but flash-attn not installed; forcing PACKING=0”**
Flash-Attention 2 isn’t available on your node. The trainer will run with packing off. That’s fine; performance just won’t get the extra packing boost.
- **Quantization API errors on 4.56–4.57**
If you see `get_loading_attributes` crashes, either:
  - stay on **4.55.4** (default) and skip 4-bit, or
  - use a separate venv with `transformers>=4.56,<4.58` (if your cluster mirror offers it), and rely on `infer_with_peft.py`’s new quantizer path.
- **bitsandbytes not found**
Install `bitsandbytes` in the venv (already in the quick start). Some clusters require NCCL/CUDA matching; use the container we ship.
- **No GPU in container**
Add `--nv` to every `singularity exec` and run on a GPU node. Check `nvidia-smi` inside the container.
- **HF auth / private repos**
Use `huggingface-cli login` inside the container, or set `HF_TOKEN` for scripts that push models.
- **Long downloads/uploads**
`pip install "huggingface_hub[hf_transfer]"` and export `HF_HUB_ENABLE_HF_TRANSFER=1`.
- **Kernel warnings**
Very old kernels may cause hangs with Accelerate/torch.distributed; this is a cluster-level setting. If you see warnings, ask your admins about a newer host kernel.
---

## License

MIT — see [LICENSE](LICENSE). If you use this work in research or production, a citation or a ⭐ is appreciated!
