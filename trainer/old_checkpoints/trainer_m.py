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
DATA_PATH = "data/train_data_repo_v4fw.jsonl"
OUTPUT_DIR = "qwen25-05-m-v4fwr_v2"
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
        
        self.fim_prefix = tokenizer.convert_tokens_to_ids(PRE)
        self.fim_middle = tokenizer.convert_tokens_to_ids(MID)
        self.fim_suffix = tokenizer.convert_tokens_to_ids(SUF)
        self.eos_token_id = tokenizer.eos_token_id
        self.ratios = [0.6, 0.1, 0.3]
    def __len__(self):
        return len(self.data)
    def _comment_out_context(self, context_str):
        lines = context_str.split('\n')
        commented_lines = ["# " + line if line.strip() and not line.strip().startswith("#") else line for line in lines]
        return "\n".join(commented_lines)
    def _apply_fim_original(self, text: str):
        first_line, text = text.split("\n", 1)
        tokens = self.tokenizer.encode(text)
        if len(tokens) < 10: 
            return first_line + "\n" + text
        idx1 = torch.randint(1, len(tokens) - 5, (1,)).item()
        idx2 = torch.randint(idx1 + 1, len(tokens) - 1, (1,)).item()
        prefix, middle, suffix = tokens[:idx1], tokens[idx1:idx2], tokens[idx2:]
        input_ids = [self.fim_prefix] + prefix + [self.fim_suffix] + suffix + [self.fim_middle] + middle
        return first_line + "\n" + self.tokenizer.decode(input_ids)
    def _apply_fim(self, text: str) -> str:
        first_line, rest = text.split("\n", 1)
        rest = self.get_intra_file_random_span_token_level(text)
        return first_line + "\n" + rest
    def __getitem__(self, index):
        sample = self.data[index]
        header = sample["header"]
        cfc = "".join(item["text"] for item in sample["cfc_contexts"])
        content: str = sample["content"]
        if random.random() < self.fim_rate:
            content = self._apply_fim_original(content)
        total = self._comment_out_context(header + cfc)+content+EOT
        return self.tokenizer(total, truncation=False, padding=False, max_length=self.max_length)
            
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
class CustomCausalCollator:
    def __init__(self, tokenizer: Qwen2Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.fim_middle = tokenizer.convert_tokens_to_ids(MID)
        self.file_sep = tokenizer.convert_tokens_to_ids(FILE)
        # print(self.fim_middle, self.file_sep)
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
            pos = (row == self.fim_middle).nonzero(as_tuple=True)
            if pos[0].numel() > 0:
                # Has <|fim_middle|>
                mid_idx = pos[0].item()
                labels[i, :mid_idx+1] = -100 # Mask <|fim_middle|> too   
            else:
                pos = (row == self.file_sep).nonzero(as_tuple=True)
                file_idx = pos[0][-1].item()
                labels[i, :file_idx+1] = -100 # Mask <|file_sep|> too
        batch["labels"] = labels
        return batch
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True, padding_side="right")
# tokenizer.pad_token = tokenizer.eos_token = "<|endoftext|>"
# --- Dataset ---
dataset = RepoLevelDataset(DATA_PATH, tokenizer, 2048+128, 0.5)
data_collator = CustomCausalCollator(tokenizer)
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

trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=data_collator)
trainer.train()