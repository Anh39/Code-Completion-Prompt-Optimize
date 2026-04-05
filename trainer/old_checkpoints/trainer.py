import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
device = "cuda"
model_name = "Qwen/Qwen2.5-Coder-1.5B" 


from typing import Any


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="right")
# tokenizer.pad_token = tokenizer.eos_token = "<|endoftext|>"
# --- Dataset ---
dataset = load_dataset("json", data_files="data/train_data_repo_gaussian_file_l.jsonl")
def tokenize(example):
    return tokenizer(example["text"], truncation=False, padding=False, max_length=2048)
dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
# Used to pad tokens, since each sample have different length
class CustomCausalCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.mid_token = 151660
    def __call__(self, features) -> Any:
        batch = self.tokenizer.pad(
            features,
            return_tensors="pt"
        )
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        for i in range(input_ids.size(0)):
            row = input_ids[i]
            pos = (row == self.mid_token).nonzero(as_tuple=True)
            if pos[0].numel() > 0:
                mid_idx = pos[0].item()
                labels[i, :mid_idx+1] = -100 # Mask <|fim_middle|> too                
        batch["labels"] = labels
        return batch
data_collator = CustomCausalCollator(tokenizer)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(device, dtype=torch.bfloat16)

model.gradient_checkpointing_enable()
model.config.use_cache = False

# --- LoRA config ---
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg).to(device, dtype=torch.bfloat16)

# --- Training ---
args = TrainingArguments(
    output_dir="qwen25-15-c-f-ft-lx",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=10,
    bf16=True,               # enables CUDA bfloat16
    logging_steps=128,
    save_strategy="steps",
    save_steps=128, # 4096 samples
    save_total_limit=100,      
    save_safetensors=True,   # ✅ use safetensors format (recommended)
    optim="adamw_torch",
    report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=dataset["train"], data_collator=data_collator)
trainer.train()