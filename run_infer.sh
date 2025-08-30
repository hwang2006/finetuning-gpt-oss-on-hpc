#!/usr/bin/env bash
# run_infer.sh — Unsloth inference inside your Singularity devel SIF
#
# Versioning policy (future-proof):
#   PIN_POLICY=auto (default)
#     • MXFP4 base (e.g. openai/gpt-oss-20b): ensure Transformers >= 4.56 (use dtype="auto" downstream).
#     • Non-MXFP4 base (e.g. Qwen): leave Transformers as-is (no pin).
#   PIN_POLICY=stability
#     • MXFP4 base: ensure >= 4.56.
#     • Non-MXFP4 base: pin Transformers==4.55.4 (helps Unsloth fast on some stacks).
#   PIN_POLICY=none  (or --no-pin)
#     • Never modify Transformers (even if MXFP4 would benefit).
#
# PEFT/LoRA: auto-switches to infer_with_peft.py when --adapter is given.

set -euo pipefail

# ---------- Defaults ----------
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

# Pinning policy: auto | stability | none
PIN_POLICY="${PIN_POLICY:-auto}"
NO_PIN="${NO_PIN:-0}"  # legacy switch; if set, forces PIN_POLICY=none

SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a concise, helpful assistant. Avoid making up facts.}"
USER_PROMPT="${USER_PROMPT:-Tell me a fun, one-paragraph fact about space.}"

# PEFT
ADAPTER="${ADAPTER:-}"
BASE_MODEL="${BASE_MODEL:-}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-}"

GPUS="${CUDA_VISIBLE_DEVICES:-}"

# Debug banner toggle (OFF by default)
DEBUG_BANNER="${DEBUG_BANNER:-0}"

# ---------- CLI ----------
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
    --adapter) ADAPTER="$2"; shift 2;;
    --base-model) BASE_MODEL="$2"; shift 2;;
    --load-in-4bit) LOAD_IN_4BIT="$2"; shift 2;;
    --pin-policy) PIN_POLICY="$2"; shift 2;;
    --no-pin) NO_PIN=1; PIN_POLICY="none"; shift;;
    -h|--help)
      cat <<EOF
Usage: $0 [options]

Container & workspace:
  --sif PATH                Singularity image (default: $SIF)
  --work DIR                Work dir; also sets VENV=\$WORK/venv (default: $WORK)
  --venv DIR                Python virtualenv inside container (default: $VENV)
  --pyfile FILE             Python entrypoint (default: $PYFILE)

Model & prompts:
  --model ID                Base model id (default: $MODEL_ID)
  --system STR              System prompt (default: $SYSTEM_PROMPT)
  --user STR                User prompt (default: $USER_PROMPT)

PEFT / LoRA:
  --adapter PATH|REPO       LoRA adapter path/repo (switches to infer_with_peft.py)
  --base-model ID           Base model for adapter (default: --model)
  --load-in-4bit            Use 4-bit BnB for non-MXFP4 bases (ignored if base is MXFP4)

Decoding & runtime:
  --max-seq-len N           Max sequence length (default: $MAX_SEQ_LEN)
  --max-new N               Max new tokens (default: $MAX_NEW_TOKENS)
  --sample 0|1              Enable sampling (default: $DO_SAMPLE)
  --temp FLOAT              Temperature (default: $TEMPERATURE)
  --top-p FLOAT             Top-p (default: $TOP_P)
  --stream 0|1              Stream tokens (default: $STREAM)
  --device-map STR          Device map (auto|balanced_low_0|cuda:0|...) (default: $DEVICE_MAP)
  --multi-gpu 0|1           Shard across GPUs (default: $MULTI_GPU)
  --headroom GB             Per-GPU memory headroom in GB (default: $PER_GPU_HEADROOM_GB)

GPU selection:
  --gpus CSV                Explicit GPU list, e.g. "0,1"; if unset, auto-detects

Transformers pin policy:
  --pin-policy {auto|stability|none}   Policy for transformers version (default: $PIN_POLICY)
  --no-pin                 Alias for --pin-policy none

Help:
  -h, --help               Show this message and exit

Notes:
  • QUANTIZE (env): auto|none|4bit — affects quantization strategy.
  • Path selection (env): HF_ONLY=0|1, USE_UNSLOTH=0|1 (defaults auto-choose for tiny models).
  • DISABLE_STREAM=1 forces non-streaming even when --stream 1.
  • BASE_ID is computed from --base-model (if set) or --model for MXFP4 detection.
  • DEBUG_BANNER=1 enables the debug banner at startup (default: 0).
EOF
      exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

# ---------- GPUs ----------
if [[ -z "$GPUS" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  else
    GPUS="0"
  fi
fi

# ---------- Guards ----------
[[ -s "$SIF" ]] || { echo "[ERR] SIF not found: $SIF" >&2; exit 1; }

if [[ -n "$ADAPTER" ]]; then
  guess1="$(dirname "$PYFILE")/infer_with_peft.py"
  guess2="$WORK/infer_with_peft.py"
  if   [[ -f "$guess1" ]]; then PYFILE="$guess1"
  elif [[ -f "$guess2" ]]; then PYFILE="$guess2"
  fi
fi
[[ -f "$PYFILE" ]] || { echo "[ERR] Python file missing: $PYFILE" >&2; exit 1; }

mkdir -p "/scratch/$USER/.cache/triton" "/scratch/$USER/.huggingface" "/scratch/$USER/tmp"

# Compute the true base id we’ll inspect for MXFP4
BASE_ID="${BASE_MODEL:-$MODEL_ID}"

# ---------- Container env ----------
COMMON_ENV=(
  "SINGULARITYENV_CUDA_VISIBLE_DEVICES=$GPUS"
  "SINGULARITYENV_TRITON_CACHE_DIR=/scratch/$USER/.cache/triton"
  "SINGULARITYENV_HF_HOME=/scratch/$USER/.huggingface"
  "SINGULARITYENV_TMPDIR=/scratch/$USER/tmp"
  "SINGULARITYENV_PYTHONUNBUFFERED=1"
  "SINGULARITYENV_LC_ALL=C.UTF-8"
  "SINGULARITYENV_LANG=C.UTF-8"

  "SINGULARITYENV_MODEL_ID=$MODEL_ID"
  "SINGULARITYENV_BASE_ID=$BASE_ID"
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
  "SINGULARITYENV_PIN_POLICY=$PIN_POLICY"
  "SINGULARITYENV_DEBUG_BANNER=$DEBUG_BANNER"
)
[[ -n "$ADAPTER" ]]      && COMMON_ENV+=("SINGULARITYENV_ADAPTER=$ADAPTER")
[[ -n "$BASE_MODEL" ]]   && COMMON_ENV+=("SINGULARITYENV_BASE_MODEL=$BASE_MODEL")
[[ -n "$LOAD_IN_4BIT" ]] && COMMON_ENV+=("SINGULARITYENV_LOAD_IN_4BIT=$LOAD_IN_4BIT")

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

  # --- Debug banner ---
  if [ \"\${DEBUG_BANNER:-0}\" -eq 1 ]; then
    echo \"================= DEBUG BANNER =================\"
    echo \"  Host user:          $USER\"
    echo \"  SIF:                $SIF\"
    echo \"  Workdir:            $WORK\"
    echo \"  Venv:               $VENV\"
    echo \"  Python file:        $PYFILE\"
    echo
    echo \"  MODEL_ID:           $MODEL_ID\"
    echo \"  BASE_MODEL:         $BASE_MODEL\"
    echo \"  BASE_ID:            $BASE_ID\"
    echo \"  ADAPTER:            $ADAPTER\"
    echo \"  LOAD_IN_4BIT:       $LOAD_IN_4BIT\"
    echo
    echo \"  MAX_SEQ_LEN:        $MAX_SEQ_LEN\"
    echo \"  MAX_NEW_TOKENS:     $MAX_NEW_TOKENS\"
    echo \"  DO_SAMPLE:          $DO_SAMPLE\"
    echo \"  TEMPERATURE:        $TEMPERATURE\"
    echo \"  TOP_P:              $TOP_P\"
    echo \"  STREAM:             $STREAM\"
    echo \"  MULTI_GPU:          $MULTI_GPU\"
    echo \"  DEVICE_MAP:         $DEVICE_MAP\"
    echo \"  PER_GPU_HEADROOM_GB:$PER_GPU_HEADROOM_GB\"
    echo
    echo \"  PIN_POLICY:         $PIN_POLICY\"
    echo \"  NO_PIN:             $NO_PIN\"
    echo \"  DEBUG_BANNER:       $DEBUG_BANNER\"
    echo \"  CUDA_VISIBLE_DEVICES:\${CUDA_VISIBLE_DEVICES:-unset}\"
    echo \"================================================\"
  fi

  # --- Decide & apply pin policy (future-proof) ---
  python - <<'PY'
import os, subprocess, sys
from packaging.version import Version
BASE = os.environ.get('BASE_ID','')
POL  = (os.environ.get('PIN_POLICY','auto') or 'auto').lower()
debug = int(os.environ.get('DEBUG_BANNER','0'))

def tf_ver():
    try:
        import transformers as t
        return Version(t.__version__.split('+')[0])
    except Exception:
        return Version('0')

def is_mxfp4(base_id:str)->bool:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(base_id, trust_remote_code=True)
        qc  = getattr(cfg, 'quantization_config', None)
        s   = (str(qc)+' '+str(type(qc))).lower()
        return 'mxfp4' in s
    except Exception:
        return False

def pip(*args):
    try:
        subprocess.check_call([sys.executable,'-m','pip',*args])
        return True
    except Exception:
        return False

mx = is_mxfp4(BASE)
cur = tf_ver()

# Print PIN DECISION line when debug banner is on
if debug:
    if POL == 'none':
        print('[PIN DECISION] No pin (policy=none)')
    elif mx and cur < Version('4.56.0'):
        print(f'[PIN DECISION] MXFP4 base + TF {cur} < 4.56 ⇒ upgrading to >=4.56')
    elif mx:
        print(f'[PIN DECISION] MXFP4 base + TF {cur} >= 4.56 ⇒ keep as-is')
    elif POL == 'stability' and cur != Version('4.55.4'):
        print('[PIN DECISION] Non-MXFP4 + stability ⇒ pinning to 4.55.4')
    elif POL == 'stability':
        print('[PIN DECISION] Non-MXFP4 + stability ⇒ already 4.55.4')
    else:
        print(f'[PIN DECISION] Non-MXFP4 + auto ⇒ keep TF {cur}')

print(f\"[INFO] transformers detected: {cur} | base={BASE or '?'} | MXFP4={mx} | policy={POL}\")

if POL == 'none':
    print('[INFO] PIN_POLICY=none -> leaving transformers untouched.')
elif mx:
    if cur < Version('4.56.0'):
        print('[INFO] MXFP4 base -> upgrading transformers to >=4.56 …')
        if not pip('install','-q','--upgrade','transformers>=4.56'):
            print('[WARN] Could not fetch stable >=4.56, trying pre-release …')
            pip('install','-q','--pre','--upgrade','transformers')
        import transformers; print('[INFO] transformers now:', transformers.__version__)
    else:
        print('[INFO] MXFP4: transformers already >=4.56; keeping as-is.')
else:
    if POL == 'stability':
        if cur != Version('4.55.4'):
            print('[INFO] Non-MXFP4 base -> pinning transformers==4.55.4 for Unsloth fast stability')
            pip('install','-q','transformers==4.55.4')
        import transformers; print('[INFO] transformers now:', transformers.__version__)
    else:
        print('[INFO] Non-MXFP4 base + policy=auto -> leaving transformers as-is.')
PY

  # --- Ensure Unsloth libs present (no forced re-pin) ---
  python - <<'PY'
import importlib, subprocess, sys
need=[]
for m,extra in (('unsloth','[base]'),('unsloth_zoo','[base]')):
    try: importlib.import_module(m)
    except Exception: need.append(m+extra)
if need:
    subprocess.check_call([sys.executable,'-m','pip','install','-q',*need])
print('[INFO] Unsloth deps OK.')
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
    if [ \"\${LOAD_IN_4BIT:-0}\" != \"0\" ]; then
      CMD_ARGS+=( --load-in-4bit )
    fi
    echo \"[INFO] Running PEFT inference with adapter: \$ADAPTER (base=\$BASE)\"
    \"\${CMD_ARGS[@]}\"
  else
    python -u '$PYFILE'
  fi
"

