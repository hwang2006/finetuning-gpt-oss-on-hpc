#!/usr/bin/env python3
import argparse, os, sys, threading
import torch
from packaging.version import parse as vparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

def tf_version():
    try:
        from importlib.metadata import version
    except Exception:
        from importlib_metadata import version  # type: ignore
    try:
        return vparse(version("transformers"))
    except Exception:
        return vparse("0.0.0")

def pick_compute_dtype():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8:  # Ampere/Hopper
            return torch.bfloat16
    return torch.float16

def try_build_new_quantizer(load_in_4bit: bool):
    """Return {'quantization_config': <AutoHfQuantizer>} if available & usable, else None."""
    if not load_in_4bit:
        return None
    try:
        from transformers.quantizers import AutoHfQuantizer
    except Exception:
        return None
    # Build a dict the new API understands
    cfg = {
        "quant_method": "bitsandbytes",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        # For new API this can be string dtype names
        "bnb_4bit_compute_dtype": "bfloat16" if pick_compute_dtype() == torch.bfloat16 else "float16",
    }
    # Different TF minor versions expose different constructors; try a few.
    for name in ("from_dict", "create", "from_config"):
        if hasattr(AutoHfQuantizer, name):
            ctor = getattr(AutoHfQuantizer, name)
            try:
                qobj = ctor(cfg)
                # very old minors may not have required methods; sanity check
                if hasattr(qobj, "get_loading_attributes"):
                    return {"quantization_config": qobj}
            except Exception:
                pass
    return None

def build_legacy_bnb_config(load_in_4bit: bool):
    """
    For Transformers < 4.56, prefer legacy top-level bitsandbytes kwargs.
    Using quantization_config=BitsAndBytesConfig(...) on 4.55.x can route
    into the new quantizer path and crash with get_loading_attributes.
    """
    if not load_in_4bit:
        return None
    compute_dtype = pick_compute_dtype()
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": compute_dtype,
    }

def load_tokenizer(adapter_path: str, base_id: str):
    # Prefer adapter dir so your saved chat template comes along
    for src, msg in ((adapter_path, "adapter directory"), (base_id, "base model")):
        try:
            tok = AutoTokenizer.from_pretrained(src, use_fast=True, trust_remote_code=True)
            print(f"[INFO] Loaded tokenizer from {msg}.")
            return tok
        except Exception:
            pass
    print("[ERR] Could not load tokenizer from adapter or base.", file=sys.stderr)
    sys.exit(1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="LoRA adapter dir or HF repo")
    p.add_argument("--base-model", default=None, help="Base model HF id")
    p.add_argument("--model", default=os.environ.get("MODEL_ID", ""), help="Fallback base model id")
    p.add_argument("--device-map", default=os.environ.get("DEVICE_MAP", "auto"))
    p.add_argument("--max-seq-len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", 4096)))
    p.add_argument("--max-new", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", 512)))
    p.add_argument("--do-sample", type=int, choices=[0,1], default=int(os.environ.get("DO_SAMPLE", 1)))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", 0.7)))
    p.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", 0.9)))
    p.add_argument("--stream", type=int, choices=[0,1], default=int(os.environ.get("STREAM", 1)))
    p.add_argument("--load-in-4bit", action="store_true", help="Load base in 4-bit (bitsandbytes)")
    p.add_argument("--system", default=os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant."))
    p.add_argument("--user",   default=os.environ.get("USER_PROMPT",   "Give me one surprising fact about space."))
    args = p.parse_args()

    base_id = args.base_model or (args.model if args.model else None)
    if not base_id:
        print("[ERR] --base-model not provided and no fallback --model in env.", file=sys.stderr)
        sys.exit(1)

    tver = tf_version()
    print(f"[INFO] Loading adapter: {args.adapter}")
    print(f"[INFO] Base model: {base_id}")
    print(f"[INFO] Transformers detected: {tver.public}")
    print(f"[INFO] Max sequence length: {args.max_seq_len} (actual cap depends on the model)")

    compute_dtype = pick_compute_dtype()
    tok = load_tokenizer(args.adapter, base_id)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    # -------- Build kwargs for base model load --------
    base_kwargs = dict(
        device_map=args.device_map,
        attn_implementation="eager",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    want_4bit = bool(args.load_in_4bit)
    tried_quant = False

    # Prefer new quantizer only on TF >= 4.56
    if want_4bit and tver >= vparse("4.56.0"):
        q = try_build_new_quantizer(load_in_4bit=True)
        if q is not None:
            base_kwargs.update(q)
            tried_quant = True

    # If still no quant cfg and user asked for 4bit, try legacy BnB config.
    if want_4bit and not tried_quant:
        q = build_legacy_bnb_config(load_in_4bit=True)
        if q is not None:
            base_kwargs.update(q)
            tried_quant = True

    # Final fallback: no quantization (with a loud warning for 4.55.x)
    def load_base_no_quant():
        print("[WARN] Falling back to non-quantized load (fp16/bf16). "
              "To use 4-bit, upgrade Transformers to >= 4.56.x in the inference venv.")
        base_kwargs.pop("quantization_config", None)
        # purge any legacy top-level bnb kwargs if present
        for k in ("load_in_4bit", "bnb_4bit_quant_type", "bnb_4bit_use_double_quant", "bnb_4bit_compute_dtype"):
            base_kwargs.pop(k, None)
        base_kwargs["torch_dtype"] = compute_dtype
        return AutoModelForCausalLM.from_pretrained(base_id, **base_kwargs)

    # -------- Load base (robust to 4.55.x quantization API) --------
    try:
        if tried_quant:
            base = AutoModelForCausalLM.from_pretrained(base_id, **base_kwargs)
        else:
            base_kwargs["torch_dtype"] = compute_dtype
            base = AutoModelForCausalLM.from_pretrained(base_id, **base_kwargs)
    except AttributeError as e:
        # The 4.55.x signatures crash with get_loading_attributes; recover to no-quant.
        if "get_loading_attributes" in str(e):
            base = load_base_no_quant()
        else:
            raise
    except Exception as e:
        # Any other quantization-related failure → fallback to no-quant
        if want_4bit:
            print(f"[WARN] 4-bit load failed ({type(e).__name__}: {e}). "
                  "Falling back to non-quantized load.", flush=True)
            base = load_base_no_quant()
        else:
            raise

    # -------- Apply LoRA adapter --------
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()

    # -------- Prompt + generation --------
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.user})

    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        print(f"[WARN] Chat template failed ({e}); using simple fallback.")
        prompt = f"System: {args.system}\nUser: {args.user}\nAssistant:"

    inputs = tok([prompt], return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new,
        do_sample=bool(args.do_sample),
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    print("[INFO] Starting generation...")
    if args.stream:
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer
        t = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
        t.start()
        for piece in streamer:
            print(piece, end="", flush=True)
        t.join()
        print()
    else:
        with torch.inference_mode():
            out = model.generate(**gen_kwargs)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        print(tok.decode(gen_ids, skip_special_tokens=True).strip())

    print("[INFO] Generation complete")

if __name__ == "__main__":
    main()
