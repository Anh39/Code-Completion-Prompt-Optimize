# ==========================================
# 0. KHẮC PHỤC CÁC CẢNH BÁO (WARNINGS)
# ==========================================
import os
# Tắt cảnh báo Tokenizer parallelism gây khó chịu
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# ==========================================
# 2. CẤU HÌNH (T4 FP16 - MEMORY SAVER)
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B" 
# Sửa lại đường dẫn dataset của bạn cho đúng
DATA_PATH = "data/train_data_repo_level_chunked_v5.jsonl"
OUTPUT_DIR = "qwen25-15-m-ft-t"
MAX_LENGTH = 2048

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==========================================
# 3. DATASET CLASS
# ==========================================
class RepoLevelDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=2048, context_ratio=0.25):
        self.data = []
        print(f">>> Đang load dataset từ {jsonl_path}...")
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line))
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy file tại {jsonl_path}!")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_budget = int(max_length * context_ratio)
        self.main_budget = max_length - self.context_budget - 4 
        
        self.fim_prefix = tokenizer.convert_tokens_to_ids("<|fim_prefix|>")
        self.fim_middle = tokenizer.convert_tokens_to_ids("<|fim_middle|>")
        self.fim_suffix = tokenizer.convert_tokens_to_ids("<|fim_suffix|>")
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

    def __getitem__(self, idx):
        row = self.data[idx]
        
        safe_context_str = self._comment_out_context(row['context'])
        raw_context_ids = self.tokenizer.encode(safe_context_str, add_special_tokens=False)
        avail = self.context_budget - self.context_meta_len
        if len(raw_context_ids) > avail: raw_context_ids = raw_context_ids[-avail:]
        context_ids = self.context_header_ids + raw_context_ids + self.context_footer_ids

        main_ids = self.tokenizer.encode(row['content'], add_special_tokens=False)
        if len(main_ids) > self.main_budget:
            start = torch.randint(0, len(main_ids) - self.main_budget, (1,)).item()
            main_ids = main_ids[start : start + self.main_budget]

        if torch.rand(1).item() < 0.5:
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

# ==========================================
# 4. LOAD MODEL (FP16 + FIX GRADIENT)
# ==========================================
print(">>> Đang load Tokenizer & Model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

# Load model FP16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)

# --- [FIX QUAN TRỌNG] Bật Gradient Checkpointing thủ công ---
# Dòng này giúp tránh lỗi "element 0... does not require grad"
model.gradient_checkpointing_enable() 
model.enable_input_require_grads()    
model.config.use_cache = False

# LoRA Config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)
print("\n>>> Số lượng tham số train:")
model.print_trainable_parameters()

# ==========================================
# 5. TRAINING (CẤU HÌNH MEMORY SAVER)
# ==========================================
dataset = RepoLevelDataset(DATA_PATH, tokenizer, max_length=MAX_LENGTH)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,  # <--- Batch 1 để tiết kiệm VRAM tối đa
    gradient_accumulation_steps=8, # <--- Bù lại bằng Accumulation
    learning_rate=2e-4, 
    logging_steps=10,
    num_train_epochs=2,
    save_steps=500,
    fp16=False,                  
    bf16=True,                 
    optim="adafactor",           # <--- Adafactor nhẹ hơn AdamW rất nhiều
    gradient_checkpointing=True, 
    report_to="none",
    # dataloader_num_workers=1,
    remove_unused_columns=False,
    ddp_find_unused_parameters=False
)

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=collate_fn,
)

print("\n>>> BẮT ĐẦU TRAINING (ỔN ĐỊNH)...")
trainer.train()

# Save
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f">>> DONE. Saved to {OUTPUT_DIR}")