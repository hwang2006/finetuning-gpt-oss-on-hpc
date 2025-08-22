# Finetuning GPT-OSS on a Supercomputer (SLURM + Singularity)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.9](https://img.shields.io/badge/CUDA-12.9-brightgreen.svg)]()
[![Singularity](https://img.shields.io/badge/Container-Singularity%2FApptainer-indigo.svg)]()

This repository contains **end-to-end recipes to _finetune_ and _infer_ GPT‑OSS and similar HF models on HPC systems** using **Singularity/Apptainer only**. All steps (env setup, inference, training, and uploading to Hugging Face) run **inside the container**.

> Works with **Singularity _or_ Apptainer**. On many clusters, `singularity` is a symlink to Apptainer—commands below work unchanged.

Tested on multi‑GPU nodes (A100/H100/H200) with the **PyTorch 2.8.0 • CUDA 12.9 devel** image.

**Repo:** <https://github.com/hwang2006/finetuning-gpt-oss-on-hpc>

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
- [7) Inference with LoRA Adapters (after training/upload)](#7-inference-with-lora-adapters-after-trainingupload)
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

# (Optional) On SLURM clusters, get an interactive **GPU** shell (login nodes usually have no GPUs):
# (replace <account> and <gpu-partition> for your site)
srun -A <account> -p <gpu-partition> --gres=gpu:1 --pty bash

# 1) Pull container (once)
singularity pull /scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif \
  docker://ghcr.io/pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

# 2) Everything below runs IN the container
export SIF=/scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif
# Tip: pass LC_ALL/LANG to avoid locale warnings at shell startup.
singularity exec --nv --env LC_ALL=C.UTF-8 --env LANG=C.UTF-8 "$SIF" bash -lc 'echo "CUDA ok"; nvidia-smi | head -10'
```

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
└── README.md
```

---

## 1) Pull the Container

```bash
# One-time pull to your scratch (recommended)
export SINGULARITY_TMPDIR=/scratch/$USER/.singularity/tmp
export SINGULARITY_CACHEDIR=/scratch/$USER/.singularity
mkdir -p /scratch/$USER/sifs

singularity pull /scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif \
  docker://ghcr.io/pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
```

> **Why the “devel” image?** Unsloth compiles Triton kernels on first run; the devel image already includes GCC & build tools.
>
> **GPU architectures:** CUDA 12.9 builds target **Volta+** (V100), **Ampere** (A100) and **Hopper** (H100/H200).

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

singularity exec --nv --env LC_ALL=C.UTF-8 --env LANG=C.UTF-8 "$SIF" bash -lc '
  set -e
  python3 -V
  python3 -m venv "'"$VENV"'"
  source "'"$VENV"'/bin/activate"
  pip install -q --upgrade pip
  # Unsloth + zoo from PyPI (recommended)
  pip install -q "unsloth[base]" "unsloth_zoo[base]"
  # Optional: faster Hub I/O
  pip install -q "huggingface_hub[hf_transfer]>=0.24.0"
  echo "✅ venv ready @ '"$VENV"'"
'
# Optional: enable accelerated Hub transfers globally
export HF_HUB_ENABLE_HF_TRANSFER=1
```

**Notes**
- If your site has strict egress, pre‑stage models into `$HF_HOME` or mirror them internally.
- If PyPI versions drift, you can fall back to Git installs (inside the container):
  ```bash
  pip uninstall -y unsloth unsloth_zoo
  pip install -q --no-deps git+https://github.com/unslothai/unsloth_zoo.git
  pip install -q --no-deps git+https://github.com/unslothai/unsloth.git
  ```

---

## 3) Sanity Check 

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
  --model Qwen/Qwen2.5-7B-Instruct \
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
                             (default: /scratch/qualis/finetuning-gpt-oss-on-hpc/outputs/Qwen2.5-0.5B-Instruct-lora-20250822-000248)

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

### Single-GPU 

```bash
./run_train.sh \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset yahma/alpaca-cleaned \
  --out /scratch/$USER/unsloth-out
```

*Or use a local JSONL instead of a Hub dataset: `--jsonl /path/data.jsonl`.*

### Multi-GPU 

```bash
./run_train.sh \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset yahma/alpaca-cleaned \
  --multi-gpu 1 \
  --out "$REPO/unsloth-out-7b"
```

### Flexible trainer with torchrun

Use the flexible trainer (`train_unsloth_flex.py`) directly if you want for more control (packing, JSONL/HF datasets, etc.).

```bash
# Example: Qwen 7B, Alpaca 1% split, 2 GPUs data parallel
singularity exec --nv --env LC_ALL=C.UTF-8 --env LANG=C.UTF-8 "$SIF" bash -lc '
  export LC_ALL=C.UTF-8 LANG=C.UTF-8
  source "'"$VENV"'/bin/activate"
  MODEL_ID="Qwen/Qwen2.5-7B-Instruct" \
  DATASET="yahma/alpaca-cleaned" DATASET_SPLIT="train[:2%]" \
  OUTPUT_DIR="$REPO/unsloth-out-7b" \
  MAX_SEQ_LEN=4096 USE_4BIT=1 PACKING=1 \
  BATCH_SIZE=1 GRAD_ACCUM=16 EPOCHS=1 LR=1e-4 LOG_STEPS=10 SAVE_STEPS=500 \
  torchrun --standalone --nproc_per_node=2 train_unsloth_flex.py
'
```

> `--nproc_per_node` should match the number of GPUs you intend to use. `torchrun` spawns 1 process per GPU.

### SLURM batch template (example)

This script trains on all GPUs allocated to the job by calling the host wrapper run_train.sh (which itself launches the container). Save it as train_llm.sbatch and submit with sbatch train_llm.sbatch.

```bash
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
  --out "$REPO/unsloth-out-7b" \
  --multi-gpu 1 \
  --max-seq-len 4096 --packing 1 --4bit 1 \
  --bs 1 --ga 16 --epochs 1 --lr 1e-4 \
  --log-steps 10 --save-steps 500
```

### Quick submits
```bash
sbatch train_llm.sbatch
```

### Notes
- run_train.sh auto-detects visible GPUs and launches torchrun with one process per GPU (via SLURM_GPUS_ON_NODE).
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
- Outputs (LoRA adapter, tokenizer, logs) land in the --out directory.

---

## 6) Upload LoRA Adapter to Hugging Face (inside the container)

```bash
# Where your adapter was saved and your target repo
export REPO=
export ADAPTER_DIR=/scratch/$USER/unsloth-out-7b
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
- Creates the repo (public by default) and uploads `adapter_model.safetensors`, tokenizer files, and metadata  
- Uses `hf_transfer` for faster uploads when available

---

## 7) Inference with LoRA Adapters (after training/upload)

> Now that you’ve trained/uploaded adapters, you can run adapter‑aware inference.  
> `run_infer.sh` will **auto‑switch** to `infer_with_peft.py` when `--adapter` is provided.

### Inference with a **local** LoRA adapter

```bash
./run_infer.sh \
  --adapter "/scratch/$USER/unsloth-out-7b" \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --load-in-4bit 1 --stream 1 \
  --user "Write a short, friendly paragraph about learning Python."
```

### Inference with an **HF‑hosted** LoRA adapter

```bash
./run_infer.sh \
  --adapter "hwang2006/qwen2.5-7b-alpaca-1pct-lora" \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --load-in-4bit 1 --stream 1 \
  --user "Suggest a heartwarming film and explain why in one sentence."
```

---

## Troubleshooting

- **No GPU on login:** use `srun -A <account> -p <gpu-partition> --gres=gpu:1 --pty bash` first.
- **Locale warning at shell startup:** add `--env LC_ALL=C.UTF-8 --env LANG=C.UTF-8` to every `singularity exec` that launches `bash`.
- **First‑run compile (Triton/GCC):** the **devel** image includes compilers; no extra bind mounts required.
- **Tokenizers “fork” warning:** harmless under `torchrun`; set `TOKENIZERS_PARALLELISM=false` to silence.
- **Slow long generations:** enable `--stream 1` in `run_infer.sh` to see tokens as they’re produced.
- **Adapter ↔ base mismatch:** ensure the base model matches what the adapter was trained on (same family/shape/tokenizer).
- **Hub auth:** gated/private models or pushing to private repos require `HF_TOKEN` or a prior `huggingface-cli login`.
- **Slow downloads/uploads:** `pip install "huggingface_hub[hf_transfer]"` and `export HF_HUB_ENABLE_HF_TRANSFER=1`.
- **Multi‑GPU training:** `--nproc_per_node` = number of GPUs; `torchrun` launches 1 process/GPU.

---

## License

MIT — see [LICENSE](LICENSE). If you use this work in research or production, a citation or a ⭐ is appreciated!
