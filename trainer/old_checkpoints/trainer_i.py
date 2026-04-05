import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import numpy as np
import random
import math
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer
)
from transformers import Qwen2Tokenizer
from peft import LoraConfig, get_peft_model
from typing import Any

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B" 
DATA_PATH = "data/train_data_repo_v6.jsonl"
OUTPUT_DIR = "qwen25-05-m-v6"
MAX_LENGTH = 2048

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
PRE =  "<|fim_prefix|>"
MID = "<|fim_middle|>"
SUF = "<|fim_suffix|>"
REPO = "<|repo_name|>"
FILE = "<|file_sep|>"
EOT = "<|endoftext|>"
class RepoLevelDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: Qwen2Tokenizer, max_length: int, fim_rate: float, context_ratio=0.25) -> None:
        super().__init__()
        self.data = []
        print(f">>> Đang load dataset từ {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip():
                        self.data.append(json.loads(line))
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy file tại {file_path}")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fim_rate = fim_rate
        self.context_budget = int(max_length * context_ratio)
        self.main_budget = max_length - self.context_budget - 10
         
        self.fim_prefix = tokenizer.convert_tokens_to_ids(PRE)
        self.fim_middle = tokenizer.convert_tokens_to_ids(MID)
        self.fim_suffix = tokenizer.convert_tokens_to_ids(SUF)
        self.eos_token_id = tokenizer.eos_token_id
        self.context_header_ids = tokenizer.encode("# <repo_context>\n", add_special_tokens=False)
        self.context_footer_ids = tokenizer.encode("\n# </repo_context>\n\n", add_special_tokens=False)
        self.context_meta_len = len(self.context_header_ids) + len(self.context_footer_ids)
    def __len__(self):
        return len(self.data)
    def _apply_fim(self, tokens, generator):
        if len(tokens) < 10: return tokens, tokens
        idx1 = torch.randint(1, len(tokens) - 5, (1,), generator=generator).item()
        idx2 = torch.randint(idx1 + 1, len(tokens) - 1, (1,), generator=generator).item()
        prefix, middle, suffix = tokens[:idx1], tokens[idx1:idx2], tokens[idx2:]
        input_ids = [self.fim_prefix] + prefix + [self.fim_suffix] + suffix + [self.fim_middle] + middle
        context_len = 1 + len(prefix) + 1 + len(suffix) + 1
        labels = [-100] * context_len + middle
        return input_ids, labels

    def _comment_out_context(self, context_str):
        lines = context_str.split('\n')
        commented_lines = ["# " + line if line.strip() and not line.strip().startswith("#") else line for line in lines]
        return "\n".join(commented_lines)
    def _data_adapter(self, sample: dict[str, Any]) -> dict[str, str]:
        _, content = sample["content"].split("\n", 1)
        cfc_contexts = []
        for item in sample["cfc_contexts"]:
            text = item["text"].replace("<|file_sep|>", "# path: ")
            cfc_contexts.append(text)
        return {
            "context": "\n".join(cfc_contexts),
            "content": content
        }
    def __getitem__(self, idx):
        row = self._data_adapter(self.data[idx])
        safe_context_str = self._comment_out_context(row['context'])
        raw_context_ids = self.tokenizer.encode(safe_context_str, add_special_tokens=False)
        avail = self.context_budget - self.context_meta_len
        if len(raw_context_ids) > avail: raw_context_ids = raw_context_ids[-avail:]
        context_ids = self.context_header_ids + raw_context_ids + self.context_footer_ids

        main_ids = self.tokenizer.encode(row['content'], add_special_tokens=False)
        if len(main_ids) > self.main_budget:
            start = torch.randint(0, len(main_ids) - self.main_budget, (1,)).item()
            main_ids = main_ids[start : start + self.main_budget]

        if torch.rand(1).item() < self.fim_rate:
            main_input, main_labels = self._apply_fim(main_ids, generator=None)
        else:
            main_input = main_ids
            main_labels = list(main_ids)

        main_input = main_input + [self.eos_token_id]
        main_labels = main_labels + [self.eos_token_id]

        input_ids = context_ids + main_input
        labels = ([-100] * len(context_ids)) + main_labels
        
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {"input_ids": input_ids, "labels": labels}
def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention_mask = [], [], []
    pad_token_id = tokenizer.pad_token_id 
    for item in batch:
        l = len(item["input_ids"])
        pad_len = max_len - l
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
        attention_mask.append([1] * l + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask)
    }
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
# tokenizer.pad_token = tokenizer.eos_token = "<|endoftext|>"
# --- Dataset ---
dataset = RepoLevelDataset(DATA_PATH, tokenizer, 2048, 0.5, 0.25)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()    
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
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# --- Training ---
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=False,
    bf16=True,               # enables CUDA bfloat16
    logging_steps=10,
    save_strategy="steps",
    save_steps=100, # 4096 samples
    save_total_limit=100,      
    gradient_checkpointing=True, 
    save_safetensors=True,   # ✅ use safetensors format (recommended)
    optim="adafactor",
    report_to="none",
    remove_unused_columns=False,
    ddp_find_unused_parameters=False
)

trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collate_fn)
trainer.train()