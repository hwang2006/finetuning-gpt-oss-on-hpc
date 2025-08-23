#!/usr/bin/env bash
set -euo pipefail

# =========================
# Defaults (can be overridden by CLI)
# =========================
SIF="${SIF:-/scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif}"
WORK="${WORK:-/scratch/$USER/finetuning-gpt-oss-on-hpc}"
VENV="${VENV:-$WORK/venv}"
PYFILE="${PYFILE:-$WORK/train_unsloth_flex.py}"

# Model / Output
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK/outputs/${MODEL_ID##*/}-lora-$(date +%Y%m%d-%H%M%S)}"

# Data (HF or JSONL)
DATASET="${DATASET:-yahma/alpaca-cleaned}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATASET_JSONL="${DATASET_JSONL:-}"  # set path to enable JSONL mode
JSONL_PROMPT_FIELD="${JSONL_PROMPT_FIELD:-instruction}"
JSONL_INPUT_FIELD="${JSONL_INPUT_FIELD:-input}"
JSONL_RESPONSE_FIELD="${JSONL_RESPONSE_FIELD:-output}"

# Prompting / sequence
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful, careful assistant.}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
PACKING="${PACKING:-1}"  # 1=true, 0=false

# LoRA
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

# Training
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-1.0}"
LR="${LR:-2e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LOG_STEPS="${LOG_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-0}"   # 0 disables eval
SEED="${SEED:-42}"
BF16="${BF16:-1}"               # 1=true, 0=false
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
REPORT_TO="${REPORT_TO:-none}"  # none|wandb|tensorboard
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
USE_4BIT="${USE_4BIT:-1}"       # 1=true, 0=false

# GPUs
GPUS="${CUDA_VISIBLE_DEVICES:-}"  # can be set via --gpus or CUDA_VISIBLE_DEVICES
MULTI_GPU="${MULTI_GPU:-1}"       # 1=torchrun, 0=single process

print_help() {
  cat <<EOF
Usage: $0 [options]

Paths:
  --sif PATH                 Singularity image (.sif) path
                             (default: $SIF)
  --work DIR                 Work dir (venv + pyfile live here)
                             (default: $WORK)
  --venv DIR                 Python venv path (default: $VENV)
  --pyfile FILE              Training Python file
                             (default: $PYFILE)

Model & Output:
  --model ID                 HF model id (default: $MODEL_ID)
  --out DIR                  Output dir for adapter/tokenizer
                             (default: $OUTPUT_DIR)

Datasets:
  --dataset NAME             HF dataset name (default: $DATASET)
  --split SPLIT              HF dataset split (default: $DATASET_SPLIT)
  --jsonl FILE               JSONL file (enables JSONL mode)
  --jsonl-prompt NAME        JSONL instruction field (default: $JSONL_PROMPT_FIELD)
  --jsonl-input NAME         JSONL input/ctx field (default: $JSONL_INPUT_FIELD)
  --jsonl-response NAME      JSONL response field (default: $JSONL_RESPONSE_FIELD)

Prompting / Sequence:
  --system "TEXT"            System prompt used in chat template
                             (default: "$SYSTEM_PROMPT")
  --max-seq-len N            Max sequence length (default: $MAX_SEQ_LEN)
  --packing 0|1              Pack short examples (default: $PACKING)

LoRA:
  --lora-r N                 LoRA rank r (default: $LORA_R)
  --lora-alpha N             LoRA alpha (default: $LORA_ALPHA)
  --lora-dropout F           LoRA dropout (default: $LORA_DROPOUT)
  --lora-targets LIST        Comma-separated target modules
                             (default: $LORA_TARGET_MODULES)

Training:
  --bs N                     per-device batch size (default: $BATCH_SIZE)
  --ga N                     gradient accumulation steps (default: $GRAD_ACCUM)
  --epochs F                 number of epochs (float OK) (default: $EPOCHS)
  --lr F                     learning rate (default: $LR)
  --warmup F                 warmup ratio (default: $WARMUP_RATIO)
  --wd F                     weight decay (default: $WEIGHT_DECAY)
  --log-steps N              logging steps (default: $LOG_STEPS)
  --save-steps N             checkpoint save steps (default: $SAVE_STEPS)
  --eval-steps N             eval steps (0 disables) (default: $EVAL_STEPS)
  --seed N                   random seed (default: $SEED)
  --bf16 0|1                 enable bf16 if supported (default: $BF16)
  --gc 0|1                   gradient checkpointing (default: $GRADIENT_CHECKPOINTING)
  --workers N                dataset num_proc (default: $NUM_WORKERS)
  --report STR               reporting: none|wandb|tensorboard (default: $REPORT_TO)
  --save-limit N             max checkpoints to keep (default: $SAVE_TOTAL_LIMIT)
  --4bit 0|1                 load base in 4-bit (QLoRA) (default: $USE_4BIT)

GPUs:
  --gpus LIST                e.g., "0" or "0,1,2,3"; if unset, auto-detect
                             (current: "${GPUS:-<auto>}")
  --multi-gpu 0|1            use torchrun across GPUs (default: $MULTI_GPU)

Examples:
  # Single GPU, Alpaca 1%, 4-bit + packing
  $0 --model Qwen/Qwen2.5-0.5B-Instruct --dataset yahma/alpaca-cleaned --split 'train[:1%]' \\
     --gpus 0 --multi-gpu 0 --out ./out/s1-alpaca1 --bs 1 --ga 16 --epochs 1 --4bit 1 --packing 1

  # Two GPUs data parallel (torchrun), IMDB 2%, 4-bit, no packing
  $0 --model Qwen/Qwen2.5-0.5B-Instruct --dataset imdb --split 'train[:2%]' \\
     --gpus 0,1 --multi-gpu 1 --gc 0 --out ./out/dp2-imdb2 --bs 1 --ga 16 --epochs 1 --4bit 1 --packing 0

EOF
}

# =========================
# CLI parsing
# =========================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sif) SIF="$2"; shift 2;;
    --work) WORK="$2"; VENV="$WORK/venv"; [[ -z "${PYFILE:-}" ]] && PYFILE="$WORK/train_unsloth_flex.py"; shift 2;;
    --venv) VENV="$2"; shift 2;;
    --pyfile) PYFILE="$2"; shift 2;;

    --model) MODEL_ID="$2"; shift 2;;
    --out) OUTPUT_DIR="$2"; shift 2;;

    --dataset) DATASET="$2"; shift 2;;
    --split) DATASET_SPLIT="$2"; shift 2;;
    --jsonl) DATASET_JSONL="$2"; shift 2;;
    --jsonl-prompt) JSONL_PROMPT_FIELD="$2"; shift 2;;
    --jsonl-input) JSONL_INPUT_FIELD="$2"; shift 2;;
    --jsonl-response) JSONL_RESPONSE_FIELD="$2"; shift 2;;

    --system) SYSTEM_PROMPT="$2"; shift 2;;
    --max-seq-len) MAX_SEQ_LEN="$2"; shift 2;;
    --packing) PACKING="$2"; shift 2;;

    --lora-r) LORA_R="$2"; shift 2;;
    --lora-alpha) LORA_ALPHA="$2"; shift 2;;
    --lora-dropout) LORA_DROPOUT="$2"; shift 2;;
    --lora-targets) LORA_TARGET_MODULES="$2"; shift 2;;

    --bs) BATCH_SIZE="$2"; shift 2;;
    --ga) GRAD_ACCUM="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --warmup) WARMUP_RATIO="$2"; shift 2;;
    --wd) WEIGHT_DECAY="$2"; shift 2;;
    --log-steps) LOG_STEPS="$2"; shift 2;;
    --save-steps) SAVE_STEPS="$2"; shift 2;;
    --eval-steps) EVAL_STEPS="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --bf16) BF16="$2"; shift 2;;
    --gc) GRADIENT_CHECKPOINTING="$2"; shift 2;;
    --workers) NUM_WORKERS="$2"; shift 2;;
    --report) REPORT_TO="$2"; shift 2;;
    --save-limit) SAVE_TOTAL_LIMIT="$2"; shift 2;;
    --4bit) USE_4BIT="$2"; shift 2;;

    --gpus) GPUS="$2"; shift 2;;
    --multi-gpu) MULTI_GPU="$2"; shift 2;;

    -h|--help) print_help; exit 0;;
    *) echo "Unknown arg: $1"; echo; print_help; exit 1;;
  esac
done

# =========================
# GPU autodetect
# =========================
if [[ -z "$GPUS" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  else
    GPUS="0"
  fi
fi
NGPU="$(awk -F, '{print NF}' <<< "$GPUS")"

# =========================
# Guards & dirs
# =========================
[[ -s "$SIF" ]]    || { echo "[ERR] SIF not found: $SIF"; exit 1; }
[[ -f "$PYFILE" ]] || { echo "[ERR] Python file missing: $PYFILE"; exit 1; }

mkdir -p "/scratch/$USER/.cache/triton" "/scratch/$USER/.huggingface" "/scratch/$USER/tmp" "$OUTPUT_DIR"

# =========================
# Env to container
# =========================
COMMON_ENV=(
  "SINGULARITYENV_CUDA_VISIBLE_DEVICES=$GPUS"
  "SINGULARITYENV_TRITON_CACHE_DIR=/scratch/$USER/.cache/triton"
  "SINGULARITYENV_HF_HOME=/scratch/$USER/.huggingface"
  "SINGULARITYENV_TMPDIR=/scratch/$USER/tmp"
  "SINGULARITYENV_PYTHONUNBUFFERED=1"
  "SINGULARITYENV_LC_ALL=C.UTF-8"
  "SINGULARITYENV_LANG=C.UTF-8"
  "SINGULARITYENV_TOKENIZERS_PARALLELISM=false"
  "SINGULARITYENV_OMP_NUM_THREADS=1"

  # Model / Output
  "SINGULARITYENV_MODEL_ID=$MODEL_ID"
  "SINGULARITYENV_OUTPUT_DIR=$OUTPUT_DIR"

  # Data
  "SINGULARITYENV_DATASET=$DATASET"
  "SINGULARITYENV_DATASET_SPLIT=$DATASET_SPLIT"
  "SINGULARITYENV_DATASET_JSONL=$DATASET_JSONL"
  "SINGULARITYENV_JSONL_PROMPT_FIELD=$JSONL_PROMPT_FIELD"
  "SINGULARITYENV_JSONL_INPUT_FIELD=$JSONL_INPUT_FIELD"
  "SINGULARITYENV_JSONL_RESPONSE_FIELD=$JSONL_RESPONSE_FIELD"

  # Prompting / seq
  "SINGULARITYENV_SYSTEM_PROMPT=$SYSTEM_PROMPT"
  "SINGULARITYENV_MAX_SEQ_LEN=$MAX_SEQ_LEN"
  "SINGULARITYENV_PACKING=$PACKING"

  # LoRA
  "SINGULARITYENV_LORA_R=$LORA_R"
  "SINGULARITYENV_LORA_ALPHA=$LORA_ALPHA"
  "SINGULARITYENV_LORA_DROPOUT=$LORA_DROPOUT"
  "SINGULARITYENV_LORA_TARGET_MODULES=$LORA_TARGET_MODULES"

  # Training
  "SINGULARITYENV_BATCH_SIZE=$BATCH_SIZE"
  "SINGULARITYENV_GRAD_ACCUM=$GRAD_ACCUM"
  "SINGULARITYENV_EPOCHS=$EPOCHS"
  "SINGULARITYENV_LR=$LR"
  "SINGULARITYENV_WARMUP_RATIO=$WARMUP_RATIO"
  "SINGULARITYENV_WEIGHT_DECAY=$WEIGHT_DECAY"
  "SINGULARITYENV_LOG_STEPS=$LOG_STEPS"
  "SINGULARITYENV_SAVE_STEPS=$SAVE_STEPS"
  "SINGULARITYENV_EVAL_STEPS=$EVAL_STEPS"
  "SINGULARITYENV_SEED=$SEED"
  "SINGULARITYENV_BF16=$BF16"
  "SINGULARITYENV_GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING"
  "SINGULARITYENV_NUM_WORKERS=$NUM_WORKERS"
  "SINGULARITYENV_REPORT_TO=$REPORT_TO"
  "SINGULARITYENV_SAVE_TOTAL_LIMIT=$SAVE_TOTAL_LIMIT"
  "SINGULARITYENV_USE_4BIT=$USE_4BIT"
)

# =========================
# Launch inside container
# =========================
env "${COMMON_ENV[@]}" \
singularity exec --nv --bind /scratch:/scratch "$SIF" bash -lc "
  set -euo pipefail
  mkdir -p /scratch/$USER/.cache/triton /scratch/$USER/.huggingface /scratch/$USER/tmp '$OUTPUT_DIR'

  # venv bootstrap
  if [ ! -x '$VENV/bin/python' ]; then
    echo '[INFO] Creating venv at $VENV ...'
    python -m venv '$VENV'
    source '$VENV/bin/activate'
    python -m pip -q install -U pip
    python -m pip -q install 'unsloth[base]' 'unsloth_zoo[base]' datasets trl
  else
    source '$VENV/bin/activate'
    python - <<'PY'
need = []
try:
  import unsloth, unsloth_zoo  # noqa
except Exception: need += ['unsloth[base]','unsloth_zoo[base]']
try:
  import datasets  # noqa
except Exception: need += ['datasets']
try:
  import trl  # noqa
except Exception: need += ['trl']
if need:
  import sys, subprocess
  subprocess.check_call([sys.executable,'-m','pip','install','-q',*need])
PY
  fi

  # --- Require flash-attn when single-process + packing=1 ---
  HAS_FA2=\$(python -c 'import importlib.util; print(1 if importlib.util.find_spec(\"flash_attn\") else 0)' 2>/dev/null || echo 0)
  : \"\${HAS_FA2:=0}\"
  if [ '$MULTI_GPU' = '0' ] && [ \"\$PACKING\" = '1' ] && [ \"\$HAS_FA2\" = '0' ]; then
    echo '[WARN] PACKING=1 requested but flash-attn not installed; forcing PACKING=0 for single-process run.'
    export PACKING=0
  fi

  echo '[INFO] Training starting on GPUs:' \${CUDA_VISIBLE_DEVICES:-unset}

  if [ '$MULTI_GPU' = '1' ] && [ '$NGPU' -ge 2 ]; then
    torchrun --standalone --nproc_per_node=$NGPU '$PYFILE'
  else
    python '$PYFILE'
  fi
"

