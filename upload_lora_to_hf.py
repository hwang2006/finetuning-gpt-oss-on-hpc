#!/usr/bin/env python3
import os, json, argparse, sys, datetime
from pathlib import Path
from typing import Optional, List
from string import Template
from textwrap import dedent
from huggingface_hub import HfApi, login, whoami

# ---------- helpers ----------
def eprint(*a): print(*a, file=sys.stderr)

def read_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(p: Path, obj: dict):
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def fix_adapter_config(model_dir: Path) -> Optional[dict]:
    cfg_p = model_dir / "adapter_config.json"
    if not cfg_p.exists():
        eprint("WARN: adapter_config.json not found; skipping fix.")
        return None
    cfg = read_json(cfg_p)
    # normalize to strings (avoids PEFT parsing issues)
    cfg["task_type"] = "CAUSAL_LM"
    cfg["peft_type"] = "LORA"
    write_json(cfg_p, cfg)
    return cfg

def infer_base_model(cfg: Optional[dict], override: Optional[str]) -> Optional[str]:
    if override:
        return override
    if not cfg:
        return None
    return cfg.get("base_model_name_or_path")

def infer_lora_fields(cfg: Optional[dict]) -> dict:
    out = {"lora_r": None, "lora_alpha": None, "lora_dropout": None,
           "target_modules": None, "peft_type": None}
    if not cfg: return out
    out["lora_r"]         = cfg.get("r")
    out["lora_alpha"]     = cfg.get("lora_alpha")
    out["lora_dropout"]   = cfg.get("lora_dropout")
    out["target_modules"] = cfg.get("target_modules")
    out["peft_type"]      = cfg.get("peft_type")
    return out

def yaml_header(license_id, base_model, tags, language, datasets) -> str:
    # simple YAML; avoid PyYAML dependency
    def j(x): return json.dumps(x, ensure_ascii=False)
    lines = [
        "---",
        f"license: {j(license_id)}",
        f"base_model: {j(base_model or 'UNKNOWN')}",
        f"tags: {j(tags)}",
        f"language: {j(language)}",
        f"datasets: {j(datasets)}",
        "library_name: peft",
        "pipeline_tag: text-generation",
        "---",
        "",
    ]
    return "\n".join(lines)

def build_model_card(repo_id: str,
                     base_model: Optional[str],
                     datasets: List[str],
                     language: List[str],
                     tags: List[str],
                     license_id: str,
                     lora_info: dict,
                     extra_notes: Optional[str]) -> str:

    hdr = yaml_header(license_id, base_model, tags, language, datasets)

    # Use Template + dedent to avoid f-string/brace issues
    tpl_str = dedent("""
    # LoRA Adapter for ${BASE_DISPLAY}

    This repository hosts a **LoRA adapter** (and tokenizer files) trained on top of **${BASE_DISPLAY}**.

    ## ✨ What’s inside
    - **PEFT type**: ${PEFT_TYPE}
    - **LoRA r**: ${LORA_R}
    - **LoRA alpha**: ${LORA_ALPHA}
    - **LoRA dropout**: ${LORA_DROPOUT}
    - **Target modules**: ${TARGET_MODULES}

    ## 📚 Datasets
    - ${DATASETS_DISPLAY}

    ## 🌐 Languages
    - ${LANGS_DISPLAY}

    ## 📝 Usage

    ### (A) Use adapter with the **official base model**
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    base = "${BASE_FOR_CODE}"
    adapter_id = "${REPO_ID}"

    tok = AutoTokenizer.from_pretrained(base)
    base_model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, adapter_id)

    messages = [
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":"Quick test?"},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9)

    print(tok.decode(out[0], skip_special_tokens=True))
    ```

    ### (B) 4-bit on the fly (if VRAM is tight)
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import torch

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = "${BASE_FOR_CODE}"
    adapter_id = "${REPO_ID}"

    tok = AutoTokenizer.from_pretrained(base)
    base_model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(base_model, adapter_id)
    ```

    ## ⚠️ Notes
    - Use a **compatible base** (architecture & tokenizer) with this LoRA.
    - This repo contains **only** adapters/tokenizer, not full model weights.
    - License here reflects this adapter’s repository. Ensure the **base model’s license** fits your use.

    ${EXTRA_NOTES}
    """).strip("\n")

    body_t = Template(tpl_str)

    base_display = base_model or "Unknown Base"
    target_modules = ", ".join(lora_info.get("target_modules") or [])
    datasets_display = ", ".join(datasets) if datasets else "N/A"
    langs_display = ", ".join(language) if language else "N/A"

    body = body_t.substitute(
        BASE_DISPLAY=base_display,
        PEFT_TYPE=lora_info.get("peft_type") or "LORA",
        LORA_R=str(lora_info.get("lora_r")),
        LORA_ALPHA=str(lora_info.get("lora_alpha")),
        LORA_DROPOUT=str(lora_info.get("lora_dropout")),
        TARGET_MODULES=target_modules or "N/A",
        DATASETS_DISPLAY=datasets_display,
        LANGS_DISPLAY=langs_display,
        BASE_FOR_CODE=base_model or "Qwen/Qwen2.5-7B-Instruct",
        REPO_ID=repo_id,
        EXTRA_NOTES=extra_notes or "",
    )

    return hdr + body + "\n"

# ---------- main ----------
def main():
    p = argparse.ArgumentParser("Upload a LoRA adapter folder to Hugging Face")
    p.add_argument("--adapter-dir", required=True, help="Path to saved LoRA folder")
    p.add_argument("--repo-id", required=True, help="e.g. username/qwen2.5-7b-alpaca-1pct-lora")
    p.add_argument("--token", default=os.getenv("HF_TOKEN") or os.getenv("HF_API_TOKEN"), help="HF token; or set HF_TOKEN env")
    p.add_argument("--private", action="store_true", help="Create repo as private")
    p.add_argument("--license", default="apache-2.0", help="SPDX license id for the adapter repo")
    p.add_argument("--base-model", default=None, help="Force base model id (overrides adapter_config)")
    p.add_argument("--datasets", default="", help="Comma-separated dataset ids")
    p.add_argument("--language", default="en", help="Comma-separated language codes (e.g. en,ko)")
    p.add_argument("--tags", default="lora,unsloth,peft", help="Comma-separated tags")
    p.add_argument("--ignore", default="checkpoint-*", help="Comma-separated ignore patterns for upload_folder")
    p.add_argument("--readme-notes", default="", help="Extra paragraph appended to README")
    p.add_argument("--commit-message", default=None, help="Custom commit message")
    args = p.parse_args()

    adapter_dir = Path(args.adapter_dir).resolve()
    if not adapter_dir.exists():
        eprint(f"ERR: adapter dir not found: {adapter_dir}")
        sys.exit(1)

    # Basic presence checks
    must = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [m for m in must if not (adapter_dir / m).exists()]
    if missing:
        eprint(f"ERR: missing files in adapter dir: {', '.join(missing)}")
        sys.exit(1)

    cfg = fix_adapter_config(adapter_dir)

    base_model = infer_base_model(cfg, args.base_model)
    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    language = [s.strip() for s in args.language.split(",") if s.strip()]
    tags     = [s.strip() for s in args.tags.split(",") if s.strip()]
    lora_info = infer_lora_fields(cfg)

    readme = build_model_card(
        repo_id=args.repo_id,
        base_model=base_model,
        datasets=datasets,
        language=language,
        tags=tags,
        license_id=args.license,
        lora_info=lora_info,
        extra_notes=args.readme_notes,
    )
    (adapter_dir / "README.md").write_text(readme, encoding="utf-8")
    print("✅ Wrote README.md")

    if not args.token:
        eprint("ERR: no HF token provided (use --token or set HF_TOKEN)")
        sys.exit(1)

    login(token=args.token)
    api = HfApi(token=args.token)

    try:
        me = whoami(token=args.token)
        print(f"🔐 Logged in as: {me.get('name') or me.get('fullname') or me['username']}")
    except Exception as e:
        eprint(f"WARN: whoami failed: {e}")

    print(f"📦 Creating/using repo: {args.repo_id} (private={args.private})")
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    ignore_patterns = [p.strip() for p in args.ignore.split(",") if p.strip()]
    commit_msg = args.commit_message or f"Upload LoRA adapter ({datetime.datetime.utcnow().isoformat()}Z)"
    print(f"🚀 Uploading folder: {adapter_dir} (ignore={ignore_patterns})")
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=commit_msg,
        ignore_patterns=ignore_patterns,
    )
    print(f"🎉 Done! View: https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()

