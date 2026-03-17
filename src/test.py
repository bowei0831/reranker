#!/usr/bin/env python3
"""
BGE-Reranker 測試腳本
"""

import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def test_model(model_path: str):
    """測試 reranker 模型"""
    
    print(f"Loading: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    if torch.cuda.is_available():
        model = model.half().cuda()
    
    model.eval()
    
    # 測試資料
    test_pairs = [
        ("What is Python?", "Python is a programming language.", "相關"),
        ("What is Python?", "The weather is nice today.", "不相關"),
        ("What is machine learning?", "ML is a subset of AI.", "相關"),
        ("What is machine learning?", "I had pizza for lunch.", "不相關"),
    ]
    
    print("\n" + "="*60)
    print("測試結果")
    print("="*60)
    
    with torch.no_grad():
        for query, passage, expected in test_pairs:
            inputs = tokenizer(
                [[query, passage]],
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = model(**inputs)
            score = outputs.logits.view(-1).float().item()
            prob = torch.sigmoid(torch.tensor(score)).item()
            
            print(f"\nQuery: {query}")
            print(f"Passage: {passage[:50]}...")
            print(f"預期: {expected}")
            print(f"Score: {score:.4f} | Prob: {prob:.4f}")
    
    print("\n" + "="*60)
    print("✓ 測試完成")


def test_with_flagembedding(model_path: str):
    """使用 FlagEmbedding 測試"""
    try:
        from FlagEmbedding import FlagReranker
        
        print(f"Loading with FlagEmbedding: {model_path}")
        reranker = FlagReranker(model_path, use_fp16=True)
        
        pairs = [
            ["What is Python?", "Python is a programming language."],
            ["What is Python?", "The weather is nice."],
        ]
        
        scores = reranker.compute_score(pairs)
        
        print("\nFlagEmbedding 結果:")
        for pair, score in zip(pairs, scores):
            print(f"  {pair[0]} | {pair[1][:30]}... → {score:.4f}")
            
    except ImportError:
        print("FlagEmbedding 未安裝")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./outputs/bge-reranker-large")
    parser.add_argument("--flag", action="store_true", help="使用 FlagEmbedding")
    
    args = parser.parse_args()
    
    if args.flag:
        test_with_flagembedding(args.model_path)
    else:
        test_model(args.model_path)