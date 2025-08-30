#!/usr/bin/env python3
"""
infer_unsloth.py

Dual-path inference:
  1) HF-only path (default for tiny models like Qwen2.5-0.5B, or when HF_ONLY=1)
  2) Unsloth fast path (when USE_UNSLOTH=1, recommended for larger models)

Why this structure?
- Importing Unsloth BEFORE transformers globally monkey-patches classes. That’s fast,
  but fallbacks can still hit patched code. By lazy importing Unsloth only on the
  "fast" path, the HF-only path stays clean and stable.

Env knobs (same names used by run_infer.sh):
  MODEL_ID (default: Qwen/Qwen2.5-7B-Instruct)
  MAX_SEQ_LEN, MAX_NEW_TOKENS
  DO_SAMPLE (0|1), TEMPERATURE, TOP_P
  STREAM (0|1)
  MULTI_GPU (0|1), DEVICE_MAP
  SYSTEM_PROMPT, USER_PROMPT
  QUANTIZE = auto|none|4bit

Selector envs:
  HF_ONLY        = 0|1  (force pure Transformers path)
  USE_UNSLOTH    = 0|1  (force Unsloth path)
  DISABLE_STREAM = 0|1  (debug: pretend stream=0 regardless of STREAM)

Notes:
- For GPT-OSS (MXFP4 vendor quant), HF path with transformers>=4.56 should be called
  with dtype="auto" so the AutoHfQuantizer activates the MXFP4 loader.
- For Unsloth path, we keep the same Unsloth API you already use.
"""

import os
import sys
from typing import Optional, Tuple

import torch


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
# Config
# -------------------------
MODEL_ID        = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
MAX_SEQ_LEN     = env_int("MAX_SEQ_LEN", 4096)
MAX_NEW_TOKENS  = env_int("MAX_NEW_TOKENS", 512)
DO_SAMPLE       = env_bool("DO_SAMPLE", True)
TEMPERATURE     = env_float("TEMPERATURE", 0.7)
TOP_P           = env_float("TOP_P", 0.9)
STREAM          = env_bool("STREAM", True)
DISABLE_STREAM  = env_bool("DISABLE_STREAM", False)

MULTI_GPU       = env_bool("MULTI_GPU", True)
DEVICE_MAP      = os.getenv("DEVICE_MAP", "auto")  # "auto" | "balanced_low_0" | "cuda:0" | ...

SYSTEM_PROMPT   = os.getenv("SYSTEM_PROMPT", "You are a concise, helpful assistant. Avoid making up facts.")
USER_PROMPT     = os.getenv("USER_PROMPT", "Tell me a fun, one-paragraph fact about space.")

QUANTIZE        = os.getenv("QUANTIZE", "auto").strip().lower()  # auto|none|4bit

# Path chooser
HF_ONLY         = env_bool("HF_ONLY", False)
USE_UNSLOTH     = env_bool("USE_UNSLOTH", False)

# Heuristic: default to HF-only for the tiny 0.5B Qwen unless user forces Unsloth
if not (HF_ONLY or USE_UNSLOTH):
    lid = MODEL_ID.lower()
    if "qwen2.5-0.5b" in lid or "qwen2.5-0.5" in lid or "qwen/qwen2.5-0.5b" in lid:
        HF_ONLY = True

# Device map
if torch.cuda.is_available():
    device_map: Optional[str] = DEVICE_MAP if MULTI_GPU else "cuda:0"
else:
    device_map = "cpu"

print(f"[0/4] env ready; GPU={torch.cuda.is_available()} | device_map={device_map} | HF_ONLY={HF_ONLY} | USE_UNSLOTH={USE_UNSLOTH}")


# -------------------------
# Common prompt utilities
# -------------------------
def build_inputs(tok, system_prompt: str, user_prompt: str, device) -> Tuple[dict, str]:
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]
    try:
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt_text = f"[SYSTEM]\n{system_prompt}\n[USER]\n{user_prompt}\n[ASSISTANT]\n"
    inputs = tok([prompt_text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs, prompt_text

def print_only_answer(tok, full_decoded: str) -> None:
    # Trim the prompt to print a neat answer in CLI
    tail = full_decoded.split(USER_PROMPT, 1)
    ans = tail[-1].strip() if len(tail) > 1 else full_decoded.strip()
    print(ans)

def base_gen_kwargs(tok):
    return dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
    )


# -------------------------
# HF-only path
# -------------------------
def run_hf_only():
    print("[1/4] HF-only path selected")
    # Lazy import transformers; Unsloth not imported in this branch at all.
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    # Model kwargs
    kwargs = dict(
        trust_remote_code=True,
        device_map=device_map,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    # QUANTIZE control
    if QUANTIZE == "4bit":
        from transformers import BitsAndBytesConfig
        comp = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=comp
        )
    elif QUANTIZE == "none":
        kwargs["quantization_config"] = None
    else:
        # auto: let TF decide (dtype="auto" activates vendor quant like MXFP4 on 4.56+)
        kwargs["dtype"] = "auto"

    print(f"[2/4] loading HF model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    # Inputs
    target_device = getattr(model, "device", None)
    if target_device is None:
        dmap = getattr(model, "hf_device_map", None)
        target_device = (next(iter(dmap.values())) if isinstance(dmap, dict) and dmap else
                         ("cuda:0" if torch.cuda.is_available() else "cpu"))
    inputs, _ = build_inputs(tok, SYSTEM_PROMPT, USER_PROMPT, target_device)

    # Gen
    gk = base_gen_kwargs(tok)
    do_stream = (STREAM and not DISABLE_STREAM)

    print("[3/4] generating… (HF)")
    if do_stream:
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(inputs, **gk, streamer=streamer)
        t = __import__("threading").Thread(target=model.generate, kwargs=kwargs, daemon=True)
        t.start()
        for piece in streamer:
            sys.stdout.write(piece)
            sys.stdout.flush()
        t.join()
        print()
    else:
        out = model.generate(**inputs, **gk)
        print_only_answer(tok, tok.decode(out[0], skip_special_tokens=True))

    print("[4/4] done.")


# -------------------------
# Unsloth fast path
# -------------------------
def run_unsloth():
    print("[1/4] Unsloth fast path selected (importing before transformers)")
    # Import Unsloth *before* transformers to get patches
    from unsloth import FastLanguageModel
    import transformers
    from transformers import AutoTokenizer

    # Load model via Unsloth
    kw = dict(
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,             # let Unsloth choose bf16/fp16
        device_map=device_map,
    )
    if QUANTIZE == "4bit":
        kw["load_in_4bit"] = True
    elif QUANTIZE == "none":
        kw["load_in_4bit"] = False
        kw["quantization_config"] = None

    print(f"[2/4] loading Unsloth model: {MODEL_ID}")
    try:
        model, tok = FastLanguageModel.from_pretrained(MODEL_ID, **kw)
    except Exception as e:
        msg = f"{e}"
        # If MXFP4 kernels missing, retry without vendor quant
        if ("mxfp4" in msg.lower()) or ("quantizer_mxfp4" in msg.lower()) or ("No module named 'kernels'" in msg) or ('No module named "kernels"' in msg):
            print("[WARN] MXFP4 requested but 'kernels' not available; retrying without MXFP4.")
            kw.pop("load_in_4bit", None)
            kw["quantization_config"] = None
            model, tok = FastLanguageModel.from_pretrained(MODEL_ID, **kw)
        else:
            raise

    # Enable Unsloth inference fuse
    model = FastLanguageModel.for_inference(model)
    model.eval()
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    # Inputs
    target_device = getattr(model, "device", None)
    if target_device is None:
        dmap = getattr(model, "hf_device_map", None)
        target_device = (next(iter(dmap.values())) if isinstance(dmap, dict) and dmap else
                         ("cuda:0" if torch.cuda.is_available() else "cpu"))
    inputs, _ = build_inputs(tok, SYSTEM_PROMPT, USER_PROMPT, target_device)

    # Gen
    gk = base_gen_kwargs(tok)
    do_stream = (STREAM and not DISABLE_STREAM)

    print("[3/4] generating… (Unsloth)")
    try:
        if do_stream:
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            kwargs = dict(inputs, **gk, streamer=streamer)
            t = __import__("threading").Thread(target=model.generate, kwargs=kwargs, daemon=True)
            t.start()
            for piece in streamer:
                sys.stdout.write(piece)
                sys.stdout.flush()
            t.join()
            print()
        else:
            out = model.generate(**inputs, **gk)
            print_only_answer(tok, tok.decode(out[0], skip_special_tokens=True))
    except Exception as e:
        # The classic small-Qwen crash: past_key_values is None in Unsloth fast forward
        if isinstance(e, AttributeError) and "NoneType" in str(e) and "shape" in str(e):
            print("[WARN] Unsloth fast generate hit past_key_values=None crash on this stack.")
            print("[INFO] Re-run with HF_ONLY=1 to avoid Unsloth patches, or try QUANTIZE=none/4bit.")
            raise
        raise

    print("[4/4] done.")


# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    try:
        if HF_ONLY and USE_UNSLOTH:
            print("[WARN] Both HF_ONLY=1 and USE_UNSLOTH=1 set; preferring HF_ONLY.")
        if HF_ONLY or not USE_UNSLOTH:
            # HF-only unless user explicitly forces Unsloth
            run_hf_only()
        else:
            run_unsloth()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[ERR] {e.__class__.__name__}: {e}", file=sys.stderr)
        sys.exit(2)

