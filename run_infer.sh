#!/usr/bin/env bash
# run_infer.sh — Unsloth inference inside your Singularity devel SIF
# - Keeps streaming + version pin
# - Restores PEFT/LoRA adapter support (auto-switch to infer_with_peft.py)

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

# Adapter options (PEFT)
ADAPTER="${ADAPTER:-}"           # path or HF repo id
BASE_MODEL="${BASE_MODEL:-}"     # default: MODEL_ID (when adapter is used)
LOAD_IN_4BIT="${LOAD_IN_4BIT:-}" # 0|1

GPUS="${CUDA_VISIBLE_DEVICES:-}" # optional; can be set by --gpus

# ---------- CLI parsing ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sif) SIF="$2"; shift 2;;
    --work) WORK="$2"; VENV="$WORK/venv"; shift 2;;
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
    # Adapter flags (restored)
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
  --gpus LIST                "0" or "0,1,2,3"; if unset, auto-detect
  --multi-gpu 0|1            enable sharding (default: $MULTI_GPU)
  --device-map STR           auto | balanced_low_0 | cuda:0 (default: $DEVICE_MAP)
  --headroom GiB             per-GPU reserve (default: $PER_GPU_HEADROOM_GB)

Streaming:
  --stream 0|1               print tokens as generated (default: $STREAM)

Prompts:
  --system "TEXT"            system prompt
  --user "TEXT"              user prompt

Adapters (PEFT / LoRA):
  --adapter PATH|REPO        LoRA adapter dir or HF repo (auto-switches to infer_with_peft.py)
  --base-model ID            Base model to pair with adapter (default: --model)
  --load-in-4bit 0|1         Load base in 4-bit (default: off)
EOF
      exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

# ---------- GPU autodetect ----------
if [[ -z "$GPUS" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  else
    GPUS="0"
  fi
fi

# ---------- Guards ----------
[[ -s "$SIF" ]] || { echo "[ERR] SIF not found: $SIF" >&2; exit 1; }

# If adapter is set, prefer a dedicated PEFT helper if present
if [[ -n "${ADAPTER:-}" ]]; then
  guess1="$(dirname "$PYFILE")/infer_with_peft.py"
  guess2="$WORK/infer_with_peft.py"
  if   [[ -f "$guess1" ]]; then PYFILE="$guess1"
  elif [[ -f "$guess2" ]]; then PYFILE="$guess2"
  fi
fi
[[ -f "$PYFILE" ]] || { echo "[ERR] Python file missing: $PYFILE" >&2; exit 1; }

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

# Optional adapter-related envs (only set if non-empty)
[[ -n "${ADAPTER:-}" ]]      && COMMON_ENV+=("SINGULARITYENV_ADAPTER=$ADAPTER")
[[ -n "${BASE_MODEL:-}" ]]   && COMMON_ENV+=("SINGULARITYENV_BASE_MODEL=$BASE_MODEL")
[[ -n "${LOAD_IN_4BIT:-}" ]] && COMMON_ENV+=("SINGULARITYENV_LOAD_IN_4BIT=$LOAD_IN_4BIT")

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

  if [ -n \"\${ADAPTER:-}\" ]; then
    BASE=\"\${BASE_MODEL:-\$MODEL_ID}\"
    CMD_ARGS=( python -u '$PYFILE'
      --adapter \"\$ADAPTER\"
      --base-model \"\$BASE\"
      --user \"\$USER_PROMPT\"
      --max-new \"\$MAX_NEW_TOKENS\"
      --max-seq-len \"\$MAX_SEQ_LEN\"
    )
    # pass boolean switch only if truthy
    if [ \"\${LOAD_IN_4BIT:-0}\" != \"0\" ]; then
      CMD_ARGS+=( --load-in-4bit )
    fi
    echo \"[INFO] Running PEFT inference with adapter: \$ADAPTER (base=\$BASE)\"
    \"\${CMD_ARGS[@]}\"
  else
    python -u '$PYFILE'
  fi
"

