import json
import random
import torch
from typing import Literal
from torch.utils.data import Dataset
from transformers import Qwen2TokenizerFast
from sfim_cache import function_cache

class CodeDataSet(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: Qwen2TokenizerFast,
        max_length: int,
        fim_rate: float,
        sfim_rate: float,
        prefix_mark: str,
        middle_mark: str,
        suffix_mark: str,
        file_mark: str,
        eot_mark: str,
        template: str,
        split_mode: Literal["token", "character"]
    ) -> None:
        super().__init__()
        self.data = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file.read().splitlines():
                if line.strip():
                    self.data.append(json.loads(line))
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fim_rate = fim_rate
        self.sfim_rate = sfim_rate
        self.prefix_mark = prefix_mark
        self.prefix_token = self._get_tokens(prefix_mark, "Prefix")
        self.middle_mark = middle_mark
        self.middle_token = self._get_tokens(middle_mark, "Middle")
        self.suffix_mark = suffix_mark
        self.suffix_token = self._get_tokens(suffix_mark, "Suffix")
        self.file_mark = file_mark
        self.file_token = self._get_tokens(file_mark, "File")
        self.eot_mark = eot_mark
        self.eot_token = self._get_tokens(eot_mark, "EOT")
        self.template = template
        self.split_mode: Literal["token", "character"] = split_mode
        
        self._skip_length = 10
        self._min_length = 5
    def _get_tokens(self, text: str, name: str) -> list[int]:
        token = self.tokenizer.convert_tokens_to_ids(text)
        if isinstance(token, list):
            print(f"{name} mark is not single token")
            return token
        else:
            return [token]
    def __len__(self) -> int:
        return len(self.data)
    def _apply_fim_token(self, text: str) -> str:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < self._skip_length:
            return text
        index_1 = random.randint(1, len(tokens) - self._min_length)
        index_2 = random.randint(index_1 + 1, len(tokens) - 1)
        prefix = tokens[:index_1]
        middle = tokens[index_1:index_2]
        suffix = tokens[index_2:]
        input_ids = (
            self.prefix_token
            + prefix
            + self.suffix_token
            + suffix
            + self.middle_token
            + middle
        )
        return self.tokenizer.decode(input_ids)
    def _apply_fim_character(self, text: str) -> str:
        if len(text) < self._skip_length:
            return text
        index_1 = random.randint(1, len(text) - self._min_length)
        index_2 = random.randint(index_1 + 1, len(text) - 1)
        prefix = text[:index_1]
        middle = text[index_1:index_2]
        suffix = text[index_2:]
        result = (
            self.prefix_mark
            + prefix
            + self.suffix_mark
            + suffix
            + self.middle_mark
            + middle
        )
        return result
    def _apply_fim(self, text: str) -> str:
        if self.split_mode == "token":
            return self._apply_fim_token(text)
        else:
            return self._apply_fim_character(text)
    def _apply_sfim(self, text: str) -> str:
        fim_text = self._apply_fim(text)
        if fim_text == text:
           return text 
        return function_cache(fim_text, self.prefix_mark, self.middle_mark, self.suffix_mark)
    def __getitem__(self, index: int) -> dict[str, list[int]]:
        sample = self.data[index]
        path = sample["path"]
        text = sample["content"]
        prompt = self.template.format(
            file_mark=self.file_mark, 
            path=path, 
            text=text
        )
        num = random.random()
        if num < self.sfim_rate:
            prompt = self._apply_sfim(prompt)
        elif num < self.sfim_rate + self.fim_rate:
            prompt = self._apply_fim(prompt)
        prompt_with_eot = prompt + self.eot_mark
        ids = self.tokenizer.encode(
            prompt_with_eot,
            truncation=False,
            padding=False,
            max_length=self.max_length,
            add_special_tokens=False
        )
        middle_token = self.middle_token[0]
        use_fim = middle_token in ids
        if use_fim:
            mid_idx = ids.index(middle_token)
            labels = [-100 if idx <= mid_idx else token_id for idx, token_id in enumerate(ids)]
        else:
            labels = ids.copy()
        return {
            "input_ids": ids,
            "labels": labels
        }

def collate_fn(tokenizer: Qwen2TokenizerFast, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention_mask = [], [], []
    pad_token_id = tokenizer.pad_token_id 
    for item in batch:
        length = len(item["input_ids"])
        pad_len = max_len - length
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
        attention_mask.append([1] * length + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask)
    }