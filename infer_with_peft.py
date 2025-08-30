#!/usr/bin/env python3
import argparse, os, sys, threading, traceback
import torch
from packaging.version import parse as V
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig,
)

def tf_version():
    try:
        import transformers as t
        return V(t.__version__.split("+")[0])
    except Exception:
        return V("0")

def pick_compute_dtype():
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    return torch.float16

def make_bnb_nf4_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=pick_compute_dtype(),
    )

def is_mxfp4(base_id: str) -> bool:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(base_id, trust_remote_code=True)
        qc = getattr(cfg, "quantization_config", None)
        s = (str(qc) + " " + str(type(qc))).lower()
        return "mxfp4" in s
    except Exception:
        return False

def load_tokenizer(pref, base):
    for src, msg in ((pref, "adapter directory"), (base, "base model")):
        try:
            tok = AutoTokenizer.from_pretrained(src, use_fast=True, trust_remote_code=True)
            print(f"[INFO] Loaded tokenizer from {msg}.")
            return tok
        except Exception:
            pass
    print("[ERR] Could not load tokenizer from adapter or base.", file=sys.stderr)
    sys.exit(1)

def get_base_model(m):
    if hasattr(m, "get_base_model"):
        try: return m.get_base_model()
        except Exception: pass
    return getattr(m, "base_model", m)

def first_device_of(model_like, fallback="cuda:0"):
    dmap = getattr(model_like, "hf_device_map", None)
    if isinstance(dmap, dict) and dmap:
        return next(iter(dmap.values()))
    return getattr(model_like, "device", fallback)

def format_quant_cfg(qc):
    if qc is None: return "no quantization"
    name = type(qc).__name__
    if "Mxfp4Config" in name: return "MXFP4"
    return name

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--device-map", default=os.environ.get("DEVICE_MAP", "balanced_low_0"))
    p.add_argument("--attn", default="eager")
    p.add_argument("--max-seq-len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", 4096)))
    p.add_argument("--max-new", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", 512)))
    p.add_argument("--do-sample", type=int, choices=[0,1], default=int(os.environ.get("DO_SAMPLE", 0)))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", 0.7)))
    p.add_argument("--top-p", dest="top_p", type=float, default=float(os.environ.get("TOP_P", 0.9)))
    p.add_argument("--top_p", dest="top_p", type=float, help=argparse.SUPPRESS)
    p.add_argument("--stream", type=int, choices=[0,1], default=int(os.environ.get("STREAM", 0)))
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--system", default=os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant."))
    p.add_argument("--user",   default=os.environ.get("USER_PROMPT",   "Give me one surprising fact about space."))
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)
    args = p.parse_args()

    tver = tf_version()
    print(f"[INFO] Loading adapter: {args.adapter}")
    print(f"[INFO] Base model: {args.base_model}")
    print(f"[INFO] Transformers detected: {tver}")
    print(f"[INFO] Max sequence length: {args.max_seq_len} (actual cap depends on the model)")

    tok = load_tokenizer(args.adapter, args.base_model)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    # extra eos for Qwen-like chat template
    extra_eos = []
    try:
        e = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(e, int) and e >= 0: extra_eos.append(e)
    except Exception:
        pass

    # --- Decide base load kwargs ---
    base_kwargs = dict(
        device_map=args.device_map,
        attn_implementation=args.attn,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    mxfp4 = is_mxfp4(args.base_model)

    if mxfp4:
        # vendor quant path; dtype="auto" on TF >=4.56
        base_kwargs["dtype"] = "auto"
        if args.load_in_4bit:
            print("[WARN] Base is MXFP4; ignoring --load-in-4bit to avoid conflict.")
    else:
        if args.load_in_4bit:
            base_kwargs["quantization_config"] = make_bnb_nf4_config()
        else:
            base_kwargs["dtype"] = pick_compute_dtype()

    # --- Load base with robust fallbacks across TF versions ---
    try:
        base = AutoModelForCausalLM.from_pretrained(args.base_model, **base_kwargs)
    except TypeError as e:
        # Older remote_code models (e.g. GPT-OSS on TF 4.55) may reject dtype
        if "unexpected keyword argument 'dtype'" in str(e):
            print("[WARN] Re-trying without dtype (older model init).")
            base_kwargs.pop("dtype", None)
            base = AutoModelForCausalLM.from_pretrained(args.base_model, **base_kwargs)
        else:
            raise
    except AttributeError as e:
        # Rare: older BnB loaders via get_loading_attributes; retry non-quantized
        if "get_loading_attributes" in str(e):
            print("[WARN] BnB path hit old-API; retrying without quantization.")
            base_kwargs.pop("quantization_config", None)
            base_kwargs["dtype"] = pick_compute_dtype()
            base = AutoModelForCausalLM.from_pretrained(args.base_model, **base_kwargs)
        else:
            raise

    qc = getattr(get_base_model(base).config, "quantization_config", None)
    print(f"[INFO] Quantization path: {format_quant_cfg(qc)}")
    dmap = getattr(get_base_model(base), "hf_device_map", None)
    if isinstance(dmap, dict) and dmap:
        print("[INFO] Device map (head):", dict(list(dmap.items())[:1]))

    # --- Attach LoRA ---
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    qc_after = getattr(get_base_model(model).config, "quantization_config", None)
    print(f"[INFO] Base quantization (under PEFT): {format_quant_cfg(qc_after)}")

    # --- Prompt & generation ---
    messages = []
    if args.system: messages.append({"role":"system","content":args.system})
    messages.append({"role":"user","content":args.user})

    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        print(f"[WARN] Chat template failed ({e}); using fallback.")
        prompt = f"System: {args.system}\nUser: {args.user}\nAssistant:"

    target = first_device_of(get_base_model(model), "cuda:0")
    inputs = tok([prompt], return_tensors="pt")
    inputs = {k: v.to(target) for k,v in inputs.items()}

    gen = dict(
        **inputs,
        max_new_tokens=args.max_new,
        do_sample=bool(args.do_sample),
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        use_cache=True,
        eos_token_id=([tok.eos_token_id] + extra_eos) if tok.eos_token_id is not None else extra_eos,
        pad_token_id=tok.pad_token_id,
    )
    if gen["do_sample"]:
        gen.update(temperature=args.temperature, top_p=args.top_p)

    print("[INFO] Starting generation...")
    if args.stream:
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen["streamer"] = streamer
        t = threading.Thread(target=model.generate, kwargs=gen, daemon=True)
        t.start()
        for piece in streamer:
            print(piece, end="", flush=True)
        t.join()
        print()
    else:
        with torch.inference_mode():
            out = model.generate(**gen)
        print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    print("[INFO] Generation complete")

if __name__ == "__main__":
    main()

