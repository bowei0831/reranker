#!/usr/bin/env python3
"""
快速評估 reranker 模型
比較你的模型和 bge-reranker-large
"""

from FlagEmbedding import FlagReranker
import time

# 測試數據
test_pairs = [
    # 中文
    ("什麼是機器學習？", "機器學習是人工智慧的一個分支，讓電腦能從數據中學習。"),
    ("什麼是機器學習？", "今天天氣很好，適合出門散步。"),
    ("如何學習Python？", "Python是一種簡單易學的程式語言，可以從官方教程開始學習。"),
    ("如何學習Python？", "蘋果是一種水果，富含維生素。"),
    # 英文
    ("What is deep learning?", "Deep learning is a subset of machine learning using neural networks."),
    ("What is deep learning?", "The movie was really boring and I fell asleep."),
]

def evaluate_model(model_path, model_name):
    print(f"\n{'='*50}")
    print(f"評估模型: {model_name}")
    print(f"路徑: {model_path}")
    print('='*50)
    
    start = time.time()
    reranker = FlagReranker(model_path, use_fp16=True)
    load_time = time.time() - start
    print(f"載入時間: {load_time:.2f}s")
    
    for query, passage in test_pairs:
        score = reranker.compute_score([query, passage])
        relevance = "✓ 相關" if score > 0 else "✗ 不相關"
        print(f"\nQuery: {query[:30]}...")
        print(f"Passage: {passage[:30]}...")
        print(f"Score: {score:.4f} ({relevance})")

if __name__ == "__main__":
    # 評估你的模型
    evaluate_model(
        "/home/peter831/test/outputs/bge-reranker-reproduce",
        "Your Model"
    )
    
    # 評估 BGE-Reranker-Large
    evaluate_model(
        "BAAI/bge-reranker-large",
        "BGE-Reranker-Large"
    )