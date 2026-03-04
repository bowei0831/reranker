#!/usr/bin/env python3
"""
BGE-Reranker 訓練腳本
使用 Transformers Trainer 從 xlm-roberta-large 訓練 reranker
"""

import os
import json
import logging
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    set_seed,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="FacebookAI/xlm-roberta-large")
    cache_dir: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    train_data: str = field(default="./data/train.jsonl")
    query_max_len: int = field(default=256)
    passage_max_len: int = field(default=512)
    train_group_size: int = field(default=8)


class RerankerDataset(Dataset):
    """Reranker 訓練資料集"""
    
    def __init__(self, data_path, tokenizer, query_max_len, passage_max_len, train_group_size):
        self.tokenizer = tokenizer
        self.max_len = query_max_len + passage_max_len
        self.train_group_size = train_group_size
        
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        if "query" in item and "pos" in item:
                            self.data.append(item)
                    except:
                        continue
        
        logger.info(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["query"]
        
        # 選擇正樣本
        pos_list = item["pos"]
        pos = pos_list[idx % len(pos_list)]
        
        # 選擇負樣本
        neg_list = item.get("neg", [])
        num_neg = self.train_group_size - 1
        
        if len(neg_list) >= num_neg:
            negs = random.sample(neg_list, num_neg)
        else:
            negs = (neg_list * (num_neg // max(len(neg_list), 1) + 1))[:num_neg]
            if not negs:
                negs = [""] * num_neg
        
        passages = [pos] + negs
        
        # Tokenize
        batch_data = []
        for passage in passages:
            encoded = self.tokenizer(
                query, passage,
                max_length=self.max_len,
                truncation="only_second",
                padding="max_length",
                return_tensors="pt"
            )
            batch_data.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            })
        
        return {
            "input_ids": torch.stack([d["input_ids"] for d in batch_data]),
            "attention_mask": torch.stack([d["attention_mask"] for d in batch_data]),
            "labels": torch.tensor([1] + [0] * (self.train_group_size - 1)),
        }


class RerankerCollator:
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.cat([f["input_ids"] for f in features], dim=0),
            "attention_mask": torch.cat([f["attention_mask"] for f in features], dim=0),
            "labels": torch.cat([f["labels"] for f in features], dim=0),
        }


class RerankerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        scores = logits[:, 1] if logits.shape[-1] == 2 else logits.squeeze(-1)
        
        loss = nn.BCEWithLogitsLoss()(scores, labels.float())
        
        return (loss, outputs) if return_outputs else loss


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    set_seed(training_args.seed)
    
    # Load tokenizer & model
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,
        cache_dir=model_args.cache_dir,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Dataset
    train_dataset = RerankerDataset(
        data_path=data_args.train_data,
        tokenizer=tokenizer,
        query_max_len=data_args.query_max_len,
        passage_max_len=data_args.passage_max_len,
        train_group_size=data_args.train_group_size,
    )
    
    # Trainer
    trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=RerankerCollator(),
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    trainer.save_model()
    
    logger.info("✓ Training completed!")


if __name__ == "__main__":
    main()