#!/usr/bin/env python3
"""
infer_unsloth.py
- Unsloth FastLanguageModel inference (single or multi-GPU)
- Safe against MXFP4 ("kernels") import errors by retrying without that quantizer
- Optional streaming output
"""

import os
import sys
import traceback
from typing import Optional

import torch

# Import Unsloth BEFORE transformers so patches apply
from unsloth import FastLanguageModel
import transformers  # noqa: F401


# -------------------------
# Env helpers
# -------------------------
def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, str(int(default))).strip().lower()
    return v in ("1", "true", "yes", "on")

def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))

def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


# -------------------------
# Config (matches run_infer.sh)
# -------------------------
MODEL_ID        = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
MAX_SEQ_LEN     = env_int("MAX_SEQ_LEN", 4096)
MAX_NEW_TOKENS  = env_int("MAX_NEW_TOKENS", 512)
DO_SAMPLE       = env_bool("DO_SAMPLE", True)
TEMPERATURE     = env_float("TEMPERATURE", 0.7)
TOP_P           = env_float("TOP_P", 0.9)
STREAM          = env_bool("STREAM", True)

MULTI_GPU       = env_bool("MULTI_GPU", True)
DEVICE_MAP      = os.getenv("DEVICE_MAP", "auto")  # "auto" | "balanced_low_0" | "cuda:0" | ...

SYSTEM_PROMPT   = os.getenv("SYSTEM_PROMPT", "You are a concise, helpful assistant. Avoid making up facts.")
USER_PROMPT     = os.getenv("USER_PROMPT", "Tell me a fun, one-paragraph fact about space.")

# Optional override: QUANTIZE in {"auto","none","4bit"}
QUANTIZE        = os.getenv("QUANTIZE", "auto").lower()


# -------------------------
# Banners
# -------------------------
print("[0/4] env ready; GPU available:", torch.cuda.is_available())
print("[1/4] importing Unsloth (before transformers)")


# -------------------------
# Device map selection
# -------------------------
if torch.cuda.is_available():
    device_map: Optional[str] = DEVICE_MAP if MULTI_GPU else "cuda:0"
else:
    device_map = "cpu"


# -------------------------
# Robust model loader
# -------------------------
def load_model_and_tokenizer() -> tuple:
    kw = dict(
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,              # Let Unsloth decide (bf16/fp16 when available)
        device_map=device_map,
    )

    # Respect QUANTIZE override
    if QUANTIZE == "4bit":
        kw["load_in_4bit"] = True
    elif QUANTIZE == "none":
        kw["load_in_4bit"] = False
        kw["quantization_config"] = None

    # 1st attempt (what the model/repo asks for)
    try:
        return FastLanguageModel.from_pretrained(MODEL_ID, **kw)
    except Exception as e:
        msg = f"{e}"
        needs_retry = (
            "mxfp4" in msg.lower()
            or "quantizer_mxfp4" in msg.lower()
            or "No module named 'kernels'" in msg
            or "No module named \"kernels\"" in msg
        )
        if not needs_retry or QUANTIZE in ("none", "4bit"):
            # Either not an MXFP4 problem, or user forced a quant mode already
            raise

        print("[WARN] MXFP4 path requested by the model but required 'kernels' package is missing.")
        print("[WARN] Retrying WITHOUT MXFP4 quantization (float/bfloat or 4-bit if you set QUANTIZE=4bit).")
        kw.pop("load_in_4bit", None)               # leave to default (False)
        kw["quantization_config"] = None
        try:
            return FastLanguageModel.from_pretrained(MODEL_ID, **kw)
        except Exception:
            print("[ERR] Retry without MXFP4 also failed:")
            traceback.print_exc()
            raise


print(f"[2/4] loading model: {MODEL_ID} (multi_gpu={MULTI_GPU}, device_map={device_map}, quantize={QUANTIZE})")
model, tok = load_model_and_tokenizer()

# Enable Unsloth fast inference path (KV cache, fused kernels)
model = FastLanguageModel.for_inference(model)
if hasattr(model, "config"):
    model.config.use_cache = True
model.eval()
torch.set_grad_enabled(False)

# Tokenizer safety
if tok.pad_token_id is None and tok.eos_token_id is not None:
    tok.pad_token_id = tok.eos_token_id


# -------------------------
# Build chat prompt
# -------------------------
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT},
]

try:
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
except Exception:
    prompt_text = f"[SYSTEM]\n{SYSTEM_PROMPT}\n[USER]\n{USER_PROMPT}\n[ASSISTANT]\n"

inputs = tok([prompt_text], return_tensors="pt")
# works for both single-GPU and sharded multi-GPU
inputs = {k: v.to(model.device) for k, v in inputs.items()}


# -------------------------
# Generation
# -------------------------
gen_kwargs = dict(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=DO_SAMPLE,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    use_cache=True,
    pad_token_id=tok.pad_token_id,
    eos_token_id=tok.eos_token_id,
)

print("[3/4] generating…")

def non_streaming_generate():
    out = model.generate(**inputs, **gen_kwargs)
    text = tok.decode(out[0], skip_special_tokens=True)
    # Light prompt trimming for CLI readability
    tail = text.split(USER_PROMPT, 1)
    ans = tail[-1].strip() if len(tail) > 1 else text.strip()
    print(ans)

if env_bool("STREAM", True):
    try:
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

        import threading
        t = threading.Thread(target=model.generate, kwargs={**inputs, **gen_kwargs})
        t.start()
        for token in streamer:
            sys.stdout.write(token)
            sys.stdout.flush()
        t.join()
        print()
    except Exception as e:
        # Fallback if some stacks trip on streaming+sharding edge-cases
        print(f"\n[WARN] Streaming path failed ({e.__class__.__name__}). Falling back to non-streaming.\n")
        non_streaming_generate()
else:
    non_streaming_generate()

print("[4/4] done.")

