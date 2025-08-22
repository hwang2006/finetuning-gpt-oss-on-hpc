#!/usr/bin/env bash
# run_infer.sh - Unsloth inference inside a devel SIF (streaming supported).
set -euo pipefail

# ---------- Defaults ----------
SIF="${SIF:-/scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif}"
WORK="${WORK:-/scratch/$USER/finetuning-gpt-oss-on-hpc}"
VENV="${VENV:-$WORK/venv}"
PYFILE="${PYFILE:-$WORK/infer_unsloth.py}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
DO_SAMPLE="${DO_SAMPLE:-1}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"

MULTI_GPU="${MULTI_GPU:-1}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
PER_GPU_HEADROOM_GB="${PER_GPU_HEADROOM_GB:-2}"

STREAM="${STREAM:-1}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a concise, helpful assistant. Avoid making up facts.}"
USER_PROMPT="${USER_PROMPT:-Tell me a fun, one-paragraph fact about space.}"

# Adapter options (PEFT)
ADAPTER="${ADAPTER:-}"           # path or HF repo id
BASE_MODEL="${BASE_MODEL:-}"     # default: MODEL_ID (when adapter is used)
LOAD_IN_4BIT="${LOAD_IN_4BIT:-}" # 0|1

GPUS="${CUDA_VISIBLE_DEVICES:-}" # optional; can be set by --gpus

# ---------- CLI parsing ----------
while [ $# -gt 0 ]; do
  case "$1" in
    --sif) SIF="$2"; shift 2;;
    --work) WORK="$2"; VENV="$WORK/venv"; [ -z "${ADAPTER:-}" ] && PYFILE="$WORK/infer_unsloth.py"; shift 2;;
    --venv) VENV="$2"; shift 2;;
    --pyfile) PYFILE="$2"; shift 2;;
    --model) MODEL_ID="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --max-seq-len) MAX_SEQ_LEN="$2"; shift 2;;
    --max-new) MAX_NEW_TOKENS="$2"; shift 2;;
    --sample) DO_SAMPLE="$2"; shift 2;;
    --temp) TEMPERATURE="$2"; shift 2;;
    --top-p) TOP_P="$2"; shift 2;;
    --multi-gpu) MULTI_GPU="$2"; shift 2;;
    --device-map) DEVICE_MAP="$2"; shift 2;;
    --headroom) PER_GPU_HEADROOM_GB="$2"; shift 2;;
    --stream) STREAM="$2"; shift 2;;
    --system) SYSTEM_PROMPT="$2"; shift 2;;
    --user) USER_PROMPT="$2"; shift 2;;
    --adapter) ADAPTER="$2"; shift 2;;
    --base-model) BASE_MODEL="$2"; shift 2;;
    --load-in-4bit) LOAD_IN_4BIT="$2"; shift 2;;
    -h|--help)
      cat <<EOF
Usage: $0 [options]

Paths:
  --sif PATH                 SIF image (default: $SIF)
  --work DIR                 Work dir (defaults venv/pyfile under it)
  --venv DIR                 Python venv path (default: $VENV)
  --pyfile FILE              Path to infer script (default: $PYFILE)

Model & decoding:
  --model ID                 HF model id (default: $MODEL_ID)
  --max-seq-len N            (default: $MAX_SEQ_LEN)
  --max-new N                (default: $MAX_NEW_TOKENS)
  --sample 0|1               do_sample (default: $DO_SAMPLE)
  --temp F                   temperature (default: $TEMPERATURE)
  --top-p F                  top_p (default: $TOP_P)

Multi-GPU:
  --gpus LIST                e.g. "0" or "0,1,2,3"; if unset, auto-detect
  --multi-gpu 0|1            enable sharding (default: $MULTI_GPU)
  --device-map STR           auto | balanced_low_0 | cuda:0 (default: $DEVICE_MAP)
  --headroom GiB             per-GPU reserve (default: $PER_GPU_HEADROOM_GB)

Streaming:
  --stream 0|1               print tokens as generated (default: $STREAM)

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
EOF
      exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

# ---------- GPU autodetect ----------
if [ -z "$GPUS" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  else
    GPUS="0"
  fi
fi

# ---------- Guards ----------
[ -s "$SIF" ] || { echo "[ERR] SIF not found: $SIF" >&2; exit 1; }

# Auto-switch to PEFT helper if adapter is set
if [ -n "${ADAPTER:-}" ]; then
  guess1="$(dirname "$PYFILE")/infer_with_peft.py"
  guess2="$WORK/infer_with_peft.py"
  if   [ -f "$guess1" ]; then PYFILE="$guess1"
  elif [ -f "$guess2" ]; then PYFILE="$guess2"
  fi
fi
[ -f "$PYFILE" ] || { echo "[ERR] Python file missing: $PYFILE" >&2; exit 1; }

mkdir -p "/scratch/$USER/.cache/triton" "/scratch/$USER/.huggingface" "/scratch/$USER/tmp"

# ---------- Export env for Singularity ----------
export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPUS"
export SINGULARITYENV_TRITON_CACHE_DIR="/scratch/$USER/.cache/triton"
export SINGULARITYENV_HF_HOME="/scratch/$USER/.huggingface"
export SINGULARITYENV_TMPDIR="/scratch/$USER/tmp"
export SINGULARITYENV_PYTHONUNBUFFERED="1"
export SINGULARITYENV_LC_ALL="C.UTF-8"
export SINGULARITYENV_LANG="C.UTF-8"

export SINGULARITYENV_MODEL_ID="$MODEL_ID"
export SINGULARITYENV_MAX_SEQ_LEN="$MAX_SEQ_LEN"
export SINGULARITYENV_MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
export SINGULARITYENV_DO_SAMPLE="$DO_SAMPLE"
export SINGULARITYENV_TEMPERATURE="$TEMPERATURE"
export SINGULARITYENV_TOP_P="$TOP_P"
export SINGULARITYENV_MULTI_GPU="$MULTI_GPU"
export SINGULARITYENV_DEVICE_MAP="$DEVICE_MAP"
export SINGULARITYENV_PER_GPU_HEADROOM_GB="$PER_GPU_HEADROOM_GB"
export SINGULARITYENV_SYSTEM_PROMPT="$SYSTEM_PROMPT"
export SINGULARITYENV_USER_PROMPT="$USER_PROMPT"
export SINGULARITYENV_STREAM="$STREAM"

# Fixed: Use safe parameter expansion to avoid unbound variable errors
[ -n "${ADAPTER:-}" ]     && export SINGULARITYENV_ADAPTER="$ADAPTER"
[ -n "${BASE_MODEL:-}" ]  && export SINGULARITYENV_BASE_MODEL="$BASE_MODEL"
[ -n "${LOAD_IN_4BIT:-}" ]&& export SINGULARITYENV_LOAD_IN_4BIT="$LOAD_IN_4BIT"

# ---------- Exec inside container ----------
singularity exec --nv --bind /scratch:/scratch "$SIF" bash -lc "
  set -euo pipefail
  mkdir -p /scratch/$USER/.cache/triton /scratch/$USER/.huggingface /scratch/$USER/tmp

  if [ ! -x '$VENV/bin/python' ]; then
    echo '[INFO] Creating venv inside container at $VENV ...'
    python -m venv '$VENV'
    . '$VENV/bin/activate'
    python -m pip -q install -U pip
    python -m pip -q install 'unsloth[base]' 'unsloth_zoo[base]'
  else
    . '$VENV/bin/activate'
    python - <<'PY'
try:
  import unsloth, unsloth_zoo  # noqa
except Exception:
  import sys, subprocess
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'unsloth[base]', 'unsloth_zoo[base]'])
PY
  fi

  echo '[INFO] Inference starting on GPUs:' \${CUDA_VISIBLE_DEVICES:-unset}

  if [ -n \"\${ADAPTER:-}\" ]; then
    BASE=\"\${BASE_MODEL:-\$MODEL_ID}\"
    # Use array to safely handle quotes and special characters
    CMD_ARGS=(
      python -u '$PYFILE'
      --adapter \"\$ADAPTER\"
      --base-model \"\$BASE\"
      --user \"\$USER_PROMPT\"
      --max-new \$MAX_NEW_TOKENS
      --max-seq-len \$MAX_SEQ_LEN
    )
    if [ \"\${LOAD_IN_4BIT:-0}\" != \"0\" ]; then
      CMD_ARGS+=(--load-in-4bit)
    fi
    echo \"[INFO] Running PEFT inference with adapter: \$ADAPTER\"
    \"\${CMD_ARGS[@]}\"
  else
    python -u '$PYFILE'
  fi
"
