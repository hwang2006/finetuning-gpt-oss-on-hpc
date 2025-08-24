#!/usr/bin/env bash
# run_infer.sh — run infer_unsloth.py inside your Singularity devel SIF (with streaming & version pin).
set -euo pipefail

# ---------- Defaults (override via flags or env) ----------
SIF="${SIF:-/scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif}"
WORK="${WORK:-/scratch/$USER/finetuning-gpt-oss-on-hpc}"
VENV="${VENV:-$WORK/venv}"
PYFILE="${PYFILE:-$WORK/infer_unsloth.py}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
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

GPUS="${CUDA_VISIBLE_DEVICES:-}"   # optional; can be set by --gpus

# ---------- CLI parsing ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sif) SIF="$2"; shift 2;;
    --work) WORK="$2"; VENV="$WORK/venv"; PYFILE="$WORK/infer_unsloth.py"; shift 2;;
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
    -h|--help)
      echo "Usage: $0 [--sif PATH] [--work DIR] [--gpus 0,1] [--model ID] [--stream 0|1] ..."
      exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# ---------- GPU autodetect if not provided ----------
if [[ -z "$GPUS" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  else
    GPUS="0"
  fi
fi

# ---------- Host-side guards ----------
[[ -s "$SIF" ]]   || { echo "[ERR] SIF not found: $SIF"; exit 1; }
[[ -f "$PYFILE" ]] || { echo "[ERR] Python file missing: $PYFILE"; exit 1; }

mkdir -p "/scratch/$USER/.cache/triton" "/scratch/$USER/.huggingface" "/scratch/$USER/tmp"

# ---------- Push env into container ----------
COMMON_ENV=(
  "SINGULARITYENV_CUDA_VISIBLE_DEVICES=$GPUS"
  "SINGULARITYENV_TRITON_CACHE_DIR=/scratch/$USER/.cache/triton"
  "SINGULARITYENV_HF_HOME=/scratch/$USER/.huggingface"
  "SINGULARITYENV_TMPDIR=/scratch/$USER/tmp"
  "SINGULARITYENV_PYTHONUNBUFFERED=1"
  "SINGULARITYENV_LC_ALL=C.UTF-8"
  "SINGULARITYENV_LANG=C.UTF-8"
  "SINGULARITYENV_MODEL_ID=$MODEL_ID"
  "SINGULARITYENV_MAX_SEQ_LEN=$MAX_SEQ_LEN"
  "SINGULARITYENV_MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
  "SINGULARITYENV_DO_SAMPLE=$DO_SAMPLE"
  "SINGULARITYENV_TEMPERATURE=$TEMPERATURE"
  "SINGULARITYENV_TOP_P=$TOP_P"
  "SINGULARITYENV_MULTI_GPU=$MULTI_GPU"
  "SINGULARITYENV_DEVICE_MAP=$DEVICE_MAP"
  "SINGULARITYENV_PER_GPU_HEADROOM_GB=$PER_GPU_HEADROOM_GB"
  "SINGULARITYENV_SYSTEM_PROMPT=$SYSTEM_PROMPT"
  "SINGULARITYENV_USER_PROMPT=$USER_PROMPT"
  "SINGULARITYENV_STREAM=$STREAM"
)

env "${COMMON_ENV[@]}" \
singularity exec --nv --bind /scratch:/scratch "$SIF" bash -lc "
  set -euo pipefail
  mkdir -p /scratch/$USER/.cache/triton /scratch/$USER/.huggingface /scratch/$USER/tmp

  if [ ! -x '$VENV/bin/python' ]; then
    echo '[INFO] Creating venv inside container at $VENV ...'
    python -m venv '$VENV'
  fi
  source '$VENV/bin/activate'
  python -m pip -q install -U pip

  # --- Pin a compatible stack for Unsloth fast generation ---
  # If Transformers is missing or >=4.56 or a dev/nightly, pin to 4.55.4.
  python - <<'PY'
import sys, subprocess
def ver(name):
    try:
        import importlib.metadata as im
        return im.version(name)
    except Exception:
        return None
def pipi(*pkgs):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])
tf = ver('transformers') or ''
bad = (tf == '' or tf.startswith('4.56') or tf.startswith('4.57') or 'dev' in tf or 'rc' in tf)
if bad:
    pipi('transformers==4.55.4')
# Ensure Unsloth & zoo present (they will pin compatible deps as needed)
try:
    import unsloth, unsloth_zoo  # noqa
except Exception:
    pipi('unsloth[base]', 'unsloth_zoo[base]')
print('[INFO] Versions pinned OK.')
PY

  echo '[INFO] Inference starting on GPUs:' \${CUDA_VISIBLE_DEVICES:-unset}
  python -u '$PYFILE'
"

