# train_unsloth_flex_fix.py
# Flexible SFT with Unsloth + TRL, single/multi-GPU (torchrun).
# Reads knobs from env (aligned with run_train.sh). Supports HF datasets and JSONL.
# - Multi-GPU DDP friendly (disables GC when world_size>1 to avoid "ready twice")
# - Patches HF Trainer.compute_loss to avoid in-place ops on views under DDP
# - Adds robust dataset field detection (Alpaca/Dolly/others)
# - Sets attn_implementation="flash_attention_2" for safe packing

import os
from typing import Dict, Any, List, Optional

# --- Import Unsloth BEFORE transformers so its patches apply ---
from unsloth import FastLanguageModel
import unsloth_zoo  # keep import for extra patches

import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
import transformers  # safe now that Unsloth is imported


# -------------------------
# Env helpers
# -------------------------
def env_str(name: str, default: str) -> str:
    return os.getenv(name, default)

def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))

def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))

def env_bool01(name: str, default: int) -> bool:
    val = os.getenv(name, str(default)).strip().lower()
    if val in ("1", "true", "yes", "on"):  return True
    if val in ("0", "false", "no", "off"): return False
    return bool(val)


# -------------------------
# Read config from env
# -------------------------
MODEL_ID       = env_str("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
OUTPUT_DIR     = env_str("OUTPUT_DIR", os.path.join(os.getcwd(), "unsloth-out-flex"))

DATASET        = env_str("DATASET", "")
DATASET_SPLIT  = env_str("DATASET_SPLIT", "train")
DATASET_JSONL  = env_str("DATASET_JSONL", "")  # if set, JSONL mode

JSONL_PROMPT_FIELD   = env_str("JSONL_PROMPT_FIELD", "instruction")
JSONL_INPUT_FIELD    = env_str("JSONL_INPUT_FIELD", "input")
JSONL_RESPONSE_FIELD = env_str("JSONL_RESPONSE_FIELD", "output")

SYSTEM_PROMPT  = env_str("SYSTEM_PROMPT", "You are a helpful, careful assistant.")
MAX_SEQ_LEN    = env_int("MAX_SEQ_LEN", 4096)
PACKING        = env_bool01("PACKING", 1)

# LoRA
LORA_R         = env_int("LORA_R", 16)
LORA_ALPHA     = env_int("LORA_ALPHA", 16)
LORA_DROPOUT   = env_float("LORA_DROPOUT", 0.0)
LORA_TARGETS   = env_str(
    "LORA_TARGET_MODULES",
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
).split(",")

# Training knobs
BATCH_SIZE     = env_int("BATCH_SIZE", 1)
GRAD_ACCUM     = env_int("GRAD_ACCUM", 8)
EPOCHS         = float(os.getenv("EPOCHS", "1.0"))
LR             = env_float("LR", 2e-4)
WARMUP_RATIO   = env_float("WARMUP_RATIO", 0.03)
WEIGHT_DECAY   = env_float("WEIGHT_DECAY", 0.0)
LOG_STEPS      = env_int("LOG_STEPS", 10)
SAVE_STEPS     = env_int("SAVE_STEPS", 500)
EVAL_STEPS     = env_int("EVAL_STEPS", 0)
SEED           = env_int("SEED", 42)
BF16           = env_bool01("BF16", 1)
GRAD_CHKPT     = env_bool01("GRADIENT_CHECKPOINTING", 1)
NUM_WORKERS    = env_int("NUM_WORKERS", 4)
REPORT_TO      = env_str("REPORT_TO", "none")
SAVE_TOTAL_LIMIT = env_int("SAVE_TOTAL_LIMIT", 3)
USE_4BIT       = env_bool01("USE_4BIT", 1)

# QoL defaults
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------
# DDP / device setup
# -------------------------
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

# 🔧 Avoid DDP re-entrant backward with LoRA + GC
if world_size > 1 and GRAD_CHKPT:
    print("[WARN] DDP world_size>1 -> disabling gradient checkpointing to avoid 'ready twice' error.")
    GRAD_CHKPT = False


# -------------------------
# Compute-loss patch (avoid in-place on views)
# -------------------------
# Some stacks do: loss *= accelerator.num_processes (in-place). Replace with out-of-place.
_original_compute_loss = transformers.trainer.Trainer.compute_loss

def _safe_compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    # Largely follows HF logic but avoids in-place scaling on the loss view.
    if self.label_smoother is not None and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None

    outputs = model(**inputs)

    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    if labels is not None:
        if hasattr(model, "module"):
            unwrapped_model = model.module
        else:
            unwrapped_model = model

        if self.label_smoother is not None:
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            if "loss" not in outputs:
                raise ValueError(
                    "Model did not return a loss. Keys: "
                    f"{','.join(outputs.keys())}. Inputs: {','.join(inputs.keys())}"
                )
            loss = outputs["loss"]
    else:
        if "loss" not in outputs:
            raise ValueError(
                "Model did not return a loss. Keys: "
                f"{','.join(outputs.keys())}. Inputs: {','.join(inputs.keys())}"
            )
        loss = outputs["loss"]

    # Replace: loss *= self.accelerator.num_processes  (in-place)
    if hasattr(self, "accelerator") and self.accelerator.num_processes > 1:
        loss = loss * self.accelerator.num_processes

    return (loss, outputs) if return_outputs else loss

# Apply class-level patch
transformers.trainer.Trainer.compute_loss = _safe_compute_loss


# -------------------------
# Load base model + tokenizer (Unsloth)
# -------------------------
device_map: Optional[dict] = {"": local_rank} if torch.cuda.is_available() else "auto"

# Use flash_attention_2 for safe packing (prevents cross-contamination)
# If your stack can't load FA2 on some nodes, remove the kwarg or set PACKING=0.
model, tok = FastLanguageModel.from_pretrained(
    MODEL_ID,
    load_in_4bit=USE_4BIT,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,                     # Unsloth chooses bf16/fp16
    device_map=device_map,
    attn_implementation="flash_attention_2",
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,      # keep 0.0 for Unsloth fast path if desired
    target_modules=LORA_TARGETS,
    use_gradient_checkpointing=GRAD_CHKPT,
)

if not GRAD_CHKPT:
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass


# -------------------------
# Field alias tables (instruction datasets)
# -------------------------
INSTRUCTION_KEYS = ["instruction", "prompt", "question", "task", "query", "title"]
OUTPUT_KEYS      = ["output", "response", "answer", "completion", "output_text"]
INPUT_KEYS       = ["input", "context", "source", "additional_input", "input_text"]


# -------------------------
# Chat template helper
# -------------------------
def _apply_chat_template(user_text: str, assistant_text: Optional[str]) -> str:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text})
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        # Fallback minimal format
        if assistant_text is None:
            return f"[SYSTEM]\n{SYSTEM_PROMPT}\n[USER]\n{user_text}\n[ASSISTANT]\n"
        return f"[SYSTEM]\n{SYSTEM_PROMPT}\n[USER]\n{user_text}\n[ASSISTANT]\n{assistant_text}"


# -------------------------
# Data loading & formatting
# -------------------------
def load_any_dataset() -> Dataset:
    # ---------- JSONL mode ----------
    if DATASET_JSONL:
        ds = load_dataset("json", data_files=DATASET_JSONL, split="train")
        cols = ds.column_names

        # Prefer env-provided names; fall back to alias sets
        in_key  = JSONL_PROMPT_FIELD   if JSONL_PROMPT_FIELD   in cols else next((k for k in INSTRUCTION_KEYS if k in cols), None)
        out_key = JSONL_RESPONSE_FIELD if JSONL_RESPONSE_FIELD in cols else next((k for k in OUTPUT_KEYS      if k in cols), None)
        aux_key = JSONL_INPUT_FIELD    if JSONL_INPUT_FIELD    in cols else next((k for k in INPUT_KEYS       if k in cols), None)

        if in_key is None or out_key is None:
            raise ValueError(f"JSONL needs instruction & response fields. Present: {cols}")

        def map_jsonl(ex):
            instr = ex.get(in_key, "")
            aux   = ex.get(aux_key, "") if aux_key else ""
            resp  = ex.get(out_key, "")
            user_text = f"{instr}\n{aux}" if aux else instr
            return {"text": _apply_chat_template(user_text, resp)}

        return ds.map(map_jsonl, remove_columns=cols, num_proc=NUM_WORKERS)

    # ---------- HF datasets ----------
    if not DATASET:
        raise ValueError("No dataset specified. Set DATASET or use DATASET_JSONL.")

    ds = load_dataset(DATASET, split=DATASET_SPLIT)
    cols = ds.column_names

    # 1) Plain 'text' column
    if "text" in cols:
        return ds

    # 2) Detect general instruction schema (Alpaca, Dolly, others)
    in_key  = next((k for k in INSTRUCTION_KEYS if k in cols), None)
    out_key = next((k for k in OUTPUT_KEYS      if k in cols), None)
    aux_key = next((k for k in INPUT_KEYS       if k in cols), None)

    if in_key and out_key:
        def map_inst(ex):
            instr = ex.get(in_key, "")
            aux   = ex.get(aux_key, "") if aux_key else ""
            resp  = ex.get(out_key, "")
            user_text = f"{instr}\n{aux}" if aux else instr
            return {"text": _apply_chat_template(user_text, resp)}
        return ds.map(map_inst, remove_columns=cols, num_proc=NUM_WORKERS)

    # 3) Last-ditch: map single text-like fields
    for cand in ["content", "sentence", "prompt"]:
        if cand in cols:
            def map_plain(ex):
                return {"text": str(ex.get(cand, ""))}
            return ds.map(map_plain, remove_columns=cols, num_proc=NUM_WORKERS)

    raise ValueError(
        f"Unsupported schema. Need a 'text' column or instruction/response pair. Columns: {cols}"
    )

train_ds = load_any_dataset()


# -------------------------
# Trainer config
# -------------------------
bf16_enable = bool(BF16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
eval_strategy = "steps" if (EVAL_STEPS and EVAL_STEPS > 0) else "no"

cfg = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    logging_steps=LOG_STEPS,
    save_steps=SAVE_STEPS,
    eval_strategy=eval_strategy,
    eval_steps=EVAL_STEPS if eval_strategy == "steps" else None,
    save_total_limit=SAVE_TOTAL_LIMIT,
    report_to=REPORT_TO,                # "none", "wandb", "tensorboard"
    bf16=bf16_enable,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=PACKING,                    # pack multiple short samples
    dataset_num_proc=NUM_WORKERS,
    seed=SEED,

    # --- DDP friendliness ---
    gradient_checkpointing=GRAD_CHKPT,  # ensure Trainer doesn't re-enable it
    ddp_find_unused_parameters=False,
)

# Rank-0 banner
if int(os.environ.get("RANK", "0")) == 0:
    eff_bs = BATCH_SIZE * GRAD_ACCUM * max(world_size, 1)
    print(f"[INFO] Training {MODEL_ID} -> {OUTPUT_DIR}")
    print(f"[INFO] 4bit={USE_4BIT}  bf16={bf16_enable}  lora_r={LORA_R}  epochs={EPOCHS}  "
          f"bs={BATCH_SIZE} x grad_accum={GRAD_ACCUM}")
    print(f"[INFO] Dataset: {'JSONL ' + DATASET_JSONL if DATASET_JSONL else f'HF {DATASET} split={DATASET_SPLIT}'}")
    print(f"[INFO] World size={world_size}  Effective batch size={eff_bs}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=train_ds,
    args=cfg,
)

# Hint DDP that the graph is static (reduces autograd "ready twice" triggers on some stacks)
try:
    if world_size > 1 and hasattr(trainer.model, "_set_static_graph"):
        trainer.model._set_static_graph()
except Exception:
    pass

# Instance-level safety net: wrap trainer.compute_loss to ensure no in-place scaling sneaks in
_old_instance_compute_loss = trainer.compute_loss
def _instance_safe_compute_loss(*args, **kwargs):
    ret = _old_instance_compute_loss(*args, **kwargs)
    # If upstream ever reintroduces in-place, guard again
    if isinstance(ret, tuple):
        loss, outputs = ret
        return (loss.clone() if loss.grad_fn is not None else loss, outputs)
    else:
        loss = ret
        return loss.clone() if loss.grad_fn is not None else loss
trainer.compute_loss = _instance_safe_compute_loss


# -------------------------
# Train
# -------------------------
trainer.train()


# -------------------------
# Save (rank 0) + clean shutdown
# -------------------------
trainer.accelerator.wait_for_everyone()
is_main = (int(os.environ.get("RANK", "0")) == 0)
if is_main:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"[INFO] Saved LoRA adapters to: {OUTPUT_DIR}")

# Graceful DDP teardown (silences NCCL destroy warning)
try:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
except Exception:
    pass

