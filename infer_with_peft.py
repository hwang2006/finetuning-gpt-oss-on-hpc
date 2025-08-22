#!/usr/bin/env python
import argparse, threading, sys, torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
)
from peft import PeftModel, AutoPeftModelForCausalLM

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="Path to LoRA adapter dir (adapter_config.json)")
    p.add_argument("--base-model", default=None, help="Base model to load (use when adapter_config lacks base)")
    p.add_argument("--max-new", type=int, default=512, help="Max new tokens to generate")
    p.add_argument("--max-seq-len", type=int, default=4096, help="Max sequence length (for compatibility)")
    p.add_argument("--do-sample", type=int, default=1, choices=[0,1])
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true", help="4-bit inference via bitsandbytes")
    p.add_argument("--system", default="You are a helpful assistant.")
    p.add_argument("--user",   default="Give me one surprising fact about space.")
    args = p.parse_args()

    print(f"[INFO] Loading adapter: {args.adapter}")
    if args.base_model:
        print(f"[INFO] Base model: {args.base_model}")
    print(f"[INFO] Max sequence length: {args.max_seq_len} (note: actual limit depends on model)")

    # dtype choice when not quantized
    chosen_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    # ---- Tokenizer (use adapter dir so your saved chat template is picked up) ----
    print("[INFO] Loading tokenizer from adapter directory...")
    tok = AutoTokenizer.from_pretrained(args.adapter)

    # ---- Load model + apply LoRA ----
    print("[INFO] Loading model and applying LoRA adapter...")
    if args.base_model:
        # Explicit base model path: load then apply PEFT adapter
        if args.load_in_4bit:
            print("[INFO] Using 4-bit quantization")
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=chosen_dtype,
            )
            base = AutoModelForCausalLM.from_pretrained(
                args.base_model, device_map="auto", quantization_config=bnb_cfg,
                trust_remote_code=False,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                args.base_model, device_map="auto", torch_dtype=chosen_dtype,
                trust_remote_code=False,
            )
        model = PeftModel.from_pretrained(base, args.adapter)
    else:
        # Adapter likely encodes base_model_name_or_path in its config
        if args.load_in_4bit:
            print("[INFO] Using 4-bit quantization with AutoPeftModel")
            model = AutoPeftModelForCausalLM.from_pretrained(
                args.adapter, device_map="auto", load_in_4bit=True,
            )
        else:
            model = AutoPeftModelForCausalLM.from_pretrained(
                args.adapter, device_map="auto", torch_dtype=chosen_dtype,
            )

    print("[INFO] Model loaded successfully")

    # ---- Build prompt with chat template ----
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user",   "content": args.user},
    ]
    
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        print(f"[WARN] Chat template failed ({e}), using fallback format")
        prompt = f"System: {args.system}\nUser: {args.user}\nAssistant:"
    
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    # ---- Generation kwargs (build safely) ----
    gen_kwargs = dict(inputs)
    gen_kwargs.update(
        max_new_tokens=args.max_new,
        do_sample=bool(args.do_sample),
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=tok.eos_token_id,
    )

    # ---- Generate (streaming optional) ----
    print("[INFO] Starting generation...")
    if args.stream:
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer
        t = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
        t.start()
        
        # Print tokens as they arrive
        for piece in streamer:
            sys.stdout.write(piece)
            sys.stdout.flush()
        t.join()
        print()  # Final newline
    else:
        with torch.inference_mode():
            out = model.generate(**gen_kwargs)
        # Decode only the new tokens (skip the input prompt)
        generated_text = tok.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        print(generated_text)

    print("[INFO] Generation complete")

if __name__ == "__main__":
    main()
