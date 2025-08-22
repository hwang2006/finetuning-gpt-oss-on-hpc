# train_unsloth.py
import os

# Import Unsloth before torch/transformers so its patches apply cleanly.
from unsloth import FastLanguageModel
import unsloth_zoo  # keep import present to enable extra patches

import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Basics
MODEL_ID    = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "4096"))

# Use OUTPUT_DIR only (no OUT_DIR). Fall back to a repo-local default.
OUTPUT_DIR = os.getenv("OUTPUT_DIR") or os.path.join(os.getcwd(), "unsloth-out")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) Load base model (QLoRA by default)
model, tok = FastLanguageModel.from_pretrained(
    MODEL_ID,
    load_in_4bit=True,          # QLoRA
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,                 # Unsloth will pick bf16/fp16 appropriately
    device_map="auto",          # shard if torchrun; else cuda:0
)

# 2) Attach LoRA (fast path: dropout=0.0)
TARGETS = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,           # keep 0.0 for Unsloth fast kernels
    target_modules=TARGETS,
    use_gradient_checkpointing=True,
)

# 3) Tiny toy dataset (IMDB) — quick smoke test
#    For real SFT with instruction data, use train_unsloth_flex.py.
#ds = load_dataset("imdb", split="train[:1%]")  # uses 'text' column
DATASET = os.getenv("DATASET")
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "train[:1%]")
if DATASET:
    ds = load_dataset(DATASET, split=DATASET_SPLIT)  # must have a 'text' column
else:
    ds = load_dataset("imdb", split="train[:1%]")

# 4) Trainer config (TRL 0.21+)
cfg = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=500,
    bf16=(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    lr_scheduler_type="cosine",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=ds,
    dataset_text_field="text",  # IMDB text column
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
    args=cfg,
)

trainer.train()

# ---- Save adapter + tokenizer (main process only) ----
trainer.accelerator.wait_for_everyone()
if trainer.accelerator.is_main_process:
    # Saves PEFT adapter files (adapter_config.json, adapter_model.safetensors, etc.)
    trainer.model.save_pretrained(OUTPUT_DIR)
    # Saves tokenizer (special tokens, chat template, etc.)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"[INFO] Saved LoRA adapter + tokenizer to: {OUTPUT_DIR}")

