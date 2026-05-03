import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.makedirs("log", exist_ok=True)

import random
import torch
import yaml
import argparse
import numpy as np
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from code_dataset import CodeDataSet, collate_fn

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="config.yaml")
with open(parser.parse_args().config_file, "r") as file:
    config = yaml.safe_load(file)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Set random seed: {seed}")
    
set_seed(config["config"]["seed"])

MODEL_ID = config["config"]["model"]["id"]
MAX_LENGTH = config["trainer"]["max_length"]

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True, 
    use_fast=config["tokenizer"]["use_fast"]
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = CodeDataSet(
    file_path=config["config"]["input"],
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
    fim_rate=config["config"]["ratio"]["fim"],
    sfim_rate=config["config"]["ratio"]["sfim"],
    prefix_mark=config["tokens"]["prefix"],
    middle_mark=config["tokens"]["middle"],
    suffix_mark=config["tokens"]["suffix"],
    file_mark=config["tokens"]["file"],
    eot_mark=config["tokens"]["eot"],
    template=config["template"]["target_file"],
    split_mode=config["config"]["split"]
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation=config["config"]["model"]["attn_implementation"]
)
def collate_fn_wrapper(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    return collate_fn(tokenizer, batch)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
args = TrainingArguments(
    output_dir=config["config"]["output"],
    per_device_train_batch_size=config["trainer"]["batch_size"],
    gradient_accumulation_steps=config["trainer"]["gradient_step"],
    learning_rate=float(config["trainer"]["lr"]),
    num_train_epochs=config["trainer"]["num_epochs"],
    fp16=False,
    bf16=True,
    logging_steps=config["trainer"]["logging_step"],
    save_strategy="no",
    gradient_checkpointing=True,
    save_safetensors=True,
    optim="adamw_torch",
    report_to="none",
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    disable_tqdm=True,
)

trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collate_fn_wrapper)
trainer.train()
trainer.save_model()
tokenizer.save_pretrained(config["config"]["output"])
