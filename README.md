# Finetuning GPT‑OSS on Supercomputer (Singularity‑only)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.9](https://img.shields.io/badge/CUDA-12.9-brightgreen.svg)]()
[![Singularity](https://img.shields.io/badge/Container-Singularity%2B-indigo.svg)]()

This repository contains **end‑to‑end recipes to _finetune_ and _infer_ GPT‑OSS and similar HF models on HPC systems** using **Singularity only**. All steps (env setup, inference, training, and upload to Hugging Face) run **inside the container**.

Tested on multi‑GPU nodes (A100/V100/H200) with the [PyTorch 2.8.0 CUDA 12.9 devel image].

> **Repo name:** `finetuning-gpt-oss-on-supercomputer`

---

## Table of Contents

- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [1) Pull the Container](#1-pull-the-container)
- [2) Create a Python venv **inside** the container](#2-create-a-python-venv-inside-the-container)
- [3) Sanity Check (inside the container)](#3-sanity-check-inside-the-container)
- [4) Inference (inside the container)](#4-inference-inside-the-container)
  - [Streaming & Long Outputs](#streaming--long-outputs)
  - [PEFT Adapter Inference](#peft-adapter-inference)
- [5) Training (inside the container)](#5-training-inside-the-container)
  - [Single‑GPU](#singlegpu)
  - [Multi‑GPU (torchrun)](#multi-gpu-torchrun)
- [6) Upload LoRA Adapter to Hugging Face (inside the container)](#6-upload-lora-adapter-to-hugging-face-inside-the-container)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Quick Start

```bash
# login node
cd /scratch/$USER
git clone https://github.com/hwang2006/finetuning-gpt-oss-on-supercomputer.git
cd finetuning-gpt-oss-on-supercomputer

# 1) Pull container (once)
singularity pull pt-2.8.0-cu129-devel.sif docker://ghcr.io/pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

# 2) Everything below runs IN the container
export SIF=$PWD/pt-2.8.0-cu129-devel.sif
singularity exec --nv "$SIF" bash -lc 'echo "CUDA ok"; nvidia-smi | head -10'
```

---

## Repository Layout

```
.
├── run_infer.sh              # Singularity-first chat/infer wrapper (streaming supported)
├── run_train.sh              # Simple single-GPU training wrapper
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
singularity pull pt-2.8.0-cu129-devel.sif docker://ghcr.io/pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
export SIF=$PWD/pt-2.8.0-cu129-devel.sif
```

> **Why “devel” image?** Unsloth compiles Triton kernels at first run; the devel image already includes GCC & build tools, avoiding host bind gymnastics.

---

## 2) Create a Python venv **inside** the container

```bash
# Create venv on scratch and install deps INSIDE the container
export VENV=/scratch/$USER/test/venv
singularity exec --nv "$SIF" bash -lc '
  set -e
  export LC_ALL=C.UTF-8 LANG=C.UTF-8
  python3 -V
  python3 -m venv "'"$VENV"'"
  source "'"$VENV"'/bin/activate"
  pip install -q --upgrade pip
  # Core stack
  pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
  # Unsloth + ecosystem
  pip install -q "unsloth[base]" "unsloth_zoo[base]" "transformers" "bitsandbytes" \
                  "accelerate" "trl" "datasets" "peft" "huggingface_hub"
  echo "✅ venv ready @ '"$VENV"'"
'
```

Optional cache location:

```bash
export HF_HOME=/scratch/$USER/.huggingface
export TMPDIR=/scratch/$USER/tmp
mkdir -p "$HF_HOME" "$TMPDIR"
```

---

## 3) Sanity Check (inside the container)

```bash
singularity exec --nv "$SIF" bash -lc '
  export LC_ALL=C.UTF-8 LANG=C.UTF-8
  source "'"$VENV"'/bin/activate"
  python - <<PY
import unsloth, unsloth_zoo, transformers, torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "GPU OK:", torch.cuda.is_available())
print("unsloth:", unsloth.__version__, "| unsloth_zoo:", unsloth_zoo.__version__)
print("transformers:", transformers.__version__)
PY
'
```

---

## 4) Inference (inside the container)

### Basic single‑GPU

```bash
singularity exec --nv "$SIF" bash -lc '
  export LC_ALL=C.UTF-8 LANG=C.UTF-8
  source "'"$VENV"'/bin/activate"
  ./run_infer.sh \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --user "Tell me a fun, one-paragraph fact about space." \
    --max-new 256 --do-sample 1 --temperature 0.7 --top-p 0.9
'
```

### Multi‑GPU inference

Select devices and enable tensor‑parallel when large models don’t fit on one GPU:

```bash
# Choose GPU IDs and pass to the script
singularity exec --nv "$SIF" bash -lc '
  export LC_ALL=C.UTF-8 LANG=C.UTF-8
  source "'"$VENV"'/bin/activate"
  ./run_infer.sh \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpus 0,1 \
    --max-new 512 --stream 1 \
    --user "Summarize the latest AI trends in ~300 words."
'
```

> The script auto‑detects available GPUs if `--gpus` is omitted. For very long generations, use `--stream 1` to print tokens as they arrive.

### Streaming & Long Outputs

```bash
# Long form + streaming
singularity exec --nv "$SIF" bash -lc '
  export LC_ALL=C.UTF-8 LANG=C.UTF-8
  source "'"$VENV"'/bin/activate"
  ./run_infer.sh \
    --model openai/gpt-oss-20b \
    --max-new 3200 --max-seq-len 8192 \
    --stream 1 \
    --system "You write long-form, comprehensive answers. Produce at least 2,000 words unless otherwise specified." \
    --user "Explain quantum computing in detail with more than 2000 words."
'
```

### PEFT Adapter Inference

**A) Use the adapter’s baked‑in base (Unsloth 4‑bit repo):**

```bash
singularity exec --nv "$SIF" bash -lc '
  export LC_ALL=C.UTF-8 LANG=C.UTF-8
  source "'"$VENV"'/bin/activate"
  python infer_with_peft_v4.py \
    --adapter /scratch/$USER/test/unsloth-out \
    --user "Quick test?"
'
```

**B) Force an official base model (FP16/BF16), optionally 4‑bit at load time:**

```bash
# FP16/BF16 base
singularity exec --nv "$SIF" bash -lc '
  export LC_ALL=C.UTF-8 LANG=C.UTF-8
  source "'"$VENV"'/bin/activate"
  python infer_with_peft_v4.py \
    --adapter /scratch/$USER/test/unsloth-out \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --user "Quick test?"
'

# Same but with on-the-fly 4-bit quant
singularity exec --nv "$SIF" bash -lc '
  export LC_ALL=C.UTF-8 LANG=C.UTF-8
  source "'"$VENV"'/bin/activate"
  python infer_with_peft_v4.py \
    --adapter /scratch/$USER/test/unsloth-out \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --load-in-4bit \
    --user "Quick test?"
'
```

> **Adapter ↔ Base compatibility:** Match architecture & shape (hidden size, layers, heads, vocab). You can check the adapter’s expected base with:
> ```bash
> singularity exec --nv "$SIF" bash -lc '
>   source "'"$VENV"'/bin/activate"
>   python - <<PY
> from peft import PeftConfig
> print(PeftConfig.from_pretrained("/scratch/$USER/test/unsloth-out").base_model_name_or_path)
> PY
> '
> ```

---

## 5) Training (inside the container)

### Single‑GPU

```bash
singularity exec --nv "$SIF" bash -lc '
  export LC_ALL=C.UTF-8 LANG=C.UTF-8
  source "'"$VENV"'/bin/activate"
  ./run_train.sh
'
```

The simple example (`train_unsloth.py`) uses IMDB (1%) for a quick smoke test and saves LoRA + tokenizer to `./unsloth-out` (or `$OUT_DIR`).

### Multi‑GPU (torchrun)

Use the flexible trainer (`train_unsloth_flex.py`) and launch with `torchrun`:

```bash
# Example: Qwen 7B, Alpaca 1% split, 2 GPUs data parallel
singularity exec --nv "$SIF" bash -lc '
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

Key envs (read by `train_unsloth_flex.py`):
- `MODEL_ID` (e.g., `Qwen/Qwen2.5-7B-Instruct`)
- `DATASET` / `DATASET_SPLIT` **or** `DATASET_JSONL` (+ `JSONL_*` field names)
- `USE_4BIT` (1 to load base weights in 4‑bit), `PACKING` (1 to pack samples)
- `BATCH_SIZE`, `GRAD_ACCUM`, `EPOCHS`, `LR`, `LOG_STEPS`, `SAVE_STEPS`
- `OUTPUT_DIR`, `MAX_SEQ_LEN`

Artifacts (LoRA + tokenizer) are saved into `OUTPUT_DIR`.

---

## 6) Upload LoRA Adapter to Hugging Face (inside the container)

```bash
# Set where your adapter lives and your target repo
export ADAPTER_DIR=/scratch/$USER/test/unsloth-out-7b
export REPO_ID=hwang2006/qwen2.5-7b-alpaca-1pct-lora
export HF_TOKEN=hf_********************************

singularity exec --nv "$SIF" bash -lc '
  export LC_ALL=C.UTF-8 LANG=C.UTF-8
  source "'"$VENV"'/bin/activate"
  pip install -q "huggingface_hub>=0.23.0"
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

---

## Troubleshooting

- **Locale warning**: add `export LC_ALL=C.UTF-8 LANG=C.UTF-8` before commands (examples above already do this).
- **First‑run compile (Triton/GCC)**: the devel image includes compilers; no extra bind mounts required.
- **Tokenizers “fork” warning**: harmless under `torchrun`; set `TOKENIZERS_PARALLELISM=false` to silence.
- **Very slow long generations**: enable `--stream 1` in `run_infer.sh` to see tokens as they’re produced.
- **Adapter mismatch**: ensure the base model you pass matches what the adapter was trained on (same family/shape). See the “Adapter ↔ Base compatibility” note.
- **Multi‑GPU 4‑bit training**: `train_unsloth_flex.py` handles device mapping per rank. If you modify it, keep `device_map={'': torch.cuda.current_device()}` per process.

---

## License

MIT — see [LICENSE](LICENSE). If you use this work in research or production, a citation or a star ⭐ is appreciated!

