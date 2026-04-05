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
DATA_PATH = "data/train_data_repo_v7.jsonl"
OUTPUT_DIR = "qwen25-05-m-v7_v6"
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
    def __init__(self, file_path: str, tokenizer: Qwen2Tokenizer, max_length: int, fim_rate: float) -> None:
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
        
        self.fim_prefix: int = tokenizer.convert_tokens_to_ids(PRE)
        self.fim_middle: int = tokenizer.convert_tokens_to_ids(MID)
        self.fim_suffix: int = tokenizer.convert_tokens_to_ids(SUF)
        self.eos_token_id = tokenizer.eos_token_id
        self.ratios = [0.6, 0.1, 0.3]
        self.first = True
    def __len__(self):
        return len(self.data)
    def _comment_out_context(self, context_str):
        lines = context_str.split('\n')
        commented_lines = ["# " + line if line.strip() and not line.strip().startswith("#") else line for line in lines]
        return "\n".join(commented_lines)
    # def _apply_fim(self, text: str) -> str:
    #     return self.get_intra_file_random_span_token_level(text)
    def _apply_fim(self, text: str):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 10: return text
        index_1 = random.randint(1, len(tokens)-5)
        index_2 = random.randint(index_1 + 1, len(tokens)-1)
        prefix, middle, suffix = tokens[:index_1], tokens[index_1:index_2], tokens[index_2:]
        input_ids = [self.fim_prefix] + prefix + [self.fim_suffix] + suffix + [self.fim_middle] + middle
        return self.tokenizer.decode(input_ids)
    def __getitem__(self, index):
        sample = self.data[index]
        repo_header = REPO + sample["repo_name"] + "\n"
        cfc = "".join(FILE + item["path"] + "\n" + item["text"] for item in sample["cfc_contexts"])
        header: str = FILE + sample["path"] + "\n"
        content: str = sample["content"]
        use_fim = False
        if random.random() < self.fim_rate:
            use_fim = True
            content = self._apply_fim(content)
        total = (
            repo_header 
            + cfc
            + header
            + content
            + EOT
        )
        ids = self.tokenizer.encode(total, truncation=False, padding=False, max_length=self.max_length, add_special_tokens=False)
        use_fim = self.fim_middle in ids
        if use_fim:
            if self.first:
                self.first = False
                with open("/root/workspace/sample.py", 'w', encoding='utf-8') as file:
                    file.write(total) 
                    # print(total)
            mid_idx = ids.index(self.fim_middle)
            labels = [-100 if idx <= mid_idx else id for idx, id in enumerate(ids)]
        else:
            labels = ids.copy()
        return {
            "input_ids": ids,
            "labels": labels
        }
    def uniform(self, n: int) -> tuple[int, int, int]:        
        r2 = random.random() * 2 * 0.1 # E(r1) = 0.1, 0 <= r1 <= 0.1
        r3 = random.random() * 2 * 0.3 # E(r3) = 0.3, 0 <= r3 <= 0.6
        mid_length = math.floor(r2 * n)
        suf_length = math.floor(r3 * n)
        pre_length = n - mid_length - suf_length
        return (pre_length, mid_length, suf_length)
    def out_of_range(self, value: float, min_value: float, max_value: float) -> float:
        return value > max_value or value < min_value
    def gaussian(self, n: int) -> tuple[int, int, int]:
        r2_ratio = 0.1
        r2 = -1
        while self.out_of_range(r2, 0, r2_ratio * 2):
            r2 = random.gauss(r2_ratio, r2_ratio / 2)

        r3_ratio = 0.3
        r3 = -1
        while self.out_of_range(r3, 0, r3_ratio * 2):
            r3 = random.gauss(r3_ratio, r3_ratio / 2)

        mid_length = math.floor(r2 * n)
        suf_length = math.floor(r3 * n)
        pre_length = n - mid_length - suf_length
        return (pre_length, mid_length, suf_length)
    def extract_sample(self, ids: list[int], start: int, mid_length: int) -> str:
        middle_ids = ids[start: start+mid_length]
        prefix_ids = ids[:start]
        suffix_ids = ids[start+mid_length:]
        prefix = self.tokenizer.decode(prefix_ids)
        middle = self.tokenizer.decode(middle_ids)
        suffix = self.tokenizer.decode(suffix_ids)
        first_sample = f"{PRE}{prefix}\n{SUF}{suffix}\n{MID}{middle}\n{EOT}"
        return first_sample
    def get_intra_file_random_span_token_level(self, content: str) -> str:
        ids = self.tokenizer.encode(content)
        n = len(ids)
        pre_length, mid_length, suf_length = self.gaussian(n)
        start = pre_length
        return self.extract_sample(ids, start, mid_length)
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
dataset = RepoLevelDataset(DATA_PATH, tokenizer, 2048, 0.5)
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
    save_steps=1000, # 4096 samples
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