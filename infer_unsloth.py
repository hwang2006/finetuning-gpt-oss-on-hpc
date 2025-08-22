#!/usr/bin/env python3
import os, sys, threading, time
os.environ.setdefault("HF_HOME", f"/scratch/{os.getenv('USER','user')}/.huggingface")
os.environ.setdefault("TMPDIR",   f"/scratch/{os.getenv('USER','user')}/tmp")

# IMPORTANT: Unsloth must be imported BEFORE transformers for patching.
from unsloth import FastLanguageModel
import unsloth_zoo  # noqa: F401  (patches kernels, speeds up)
import torch
from transformers import TextIteratorStreamer

def env_str(name, default):
    return os.getenv(name, default)

def env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def main():
    t0 = time.time()
    # -------- config from env ----------
    model_id  = env_str("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    max_seq   = env_int("MAX_SEQ_LEN", 4096)
    max_new   = env_int("MAX_NEW_TOKENS", 160)
    do_sample = bool(env_int("DO_SAMPLE", 1))
    temp      = env_float("TEMPERATURE", 0.7)
    top_p     = env_float("TOP_P", 0.9)
    stream    = bool(env_int("STREAM", 1))
    multi_gpu = bool(env_int("MULTI_GPU", 1))
    device_map= env_str("DEVICE_MAP", "auto")  # auto | balanced_low_0 | cuda:0 ...
    headroom  = env_int("PER_GPU_HEADROOM_GB", 2)

    system_prompt = env_str("SYSTEM_PROMPT", "You are a concise, helpful assistant.")
    user_prompt   = env_str("USER_PROMPT",   "Tell me a fun, one-paragraph fact about space.")

    # Detect visible GPUs
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
    ngpu = 0
    if torch.cuda.is_available():
        if cuda_visible:
            ngpu = len([x for x in cuda_visible.split(",") if x.strip() != ""])
        else:
            ngpu = torch.cuda.device_count()

    print(f"[0/4] env ready; GPU available: {torch.cuda.is_available()}", flush=True)
    print(f"[1/4] importing Unsloth (before transformers)", flush=True)

    # -------- load model ----------
    print(f"[2/4] loading model: {model_id} (ngpu={ngpu}, multi_gpu={multi_gpu}, device_map={device_map})", flush=True)
    kwargs = {
        "max_seq_length": max_seq,
        "dtype": None,                 # let Unsloth pick bf16/fp16
        "load_in_4bit": True,
    }

    # Choose device map / placement
    if torch.cuda.is_available():
        if multi_gpu and ngpu >= 2 and device_map != "cuda:0":
            kwargs["device_map"] = device_map          # e.g., "auto" or "balanced_low_0"
        else:
            kwargs["device_map"] = "cuda:0"
    else:
        kwargs["device_map"] = "cpu"

    model, tok = FastLanguageModel.from_pretrained(model_id, **kwargs)
    FastLanguageModel.for_inference(model)
    print("    model loaded.", flush=True)

    # -------- prompt formatting ----------
    # Prefer chat template if provided by tokenizer
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback to a simple chat format
        prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

    inputs = tok(prompt, return_tensors="pt").to(model.device)

    # -------- generate (streaming or not) ----------
    print("[3/4] generating…", flush=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new,
        do_sample=do_sample,
        temperature=temp,
        top_p=top_p,
        eos_token_id=tok.eos_token_id,
    )

    if stream:
        # Stream incremental text chunks as they arrive
        streamer = TextIteratorStreamer(
            tok,
            skip_special_tokens=True,
            skip_prompt=True,
            decode_kwargs={"spaces_between_special_tokens": False},
        )

        gen_kwargs["streamer"] = streamer

        # Start generation in a worker thread so we can consume output live
        worker = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
        worker.start()

        # Consume tokens as soon as they are produced
        sys.stdout.flush()
        first = True
        for text in streamer:
            # First chunk often starts mid-word; just print as-is.
            if first:
                first = False
            sys.stdout.write(text)
            sys.stdout.flush()
        # Ensure a clean newline at the end of the stream
        if not first:
            sys.stdout.write("\n")
            sys.stdout.flush()
        worker.join()
    else:
        with torch.inference_mode():
            out = model.generate(**gen_kwargs)
        text = tok.decode(out[0], skip_special_tokens=True)
        print(text)

    dt = time.time() - t0
    print(f"[4/4] done in {dt:.1f}s", flush=True)

if __name__ == "__main__":
    # Make sure stdout is unbuffered even if PYTHONUNBUFFERED was not set.
    try:
        import os
        if os.isatty(sys.stdout.fileno()):
            pass
    except Exception:
        pass
    main()

