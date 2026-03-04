# eval_thesis_reranker.py
import json
import os
from collections import defaultdict
from FlagEmbedding import FlagReranker
import torch
from tqdm import tqdm

EVAL_DATA_PATH = "/home/peter831/test/eval_data/thesis_eval_dataset_v3.jsonl"
OUTPUT_DIR = "/home/peter831/test/eval_results"

def load_eval_data(filepath):
    """載入處理好的評估資料"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def evaluate_reranker(model_path, model_name, eval_data):
    """評估 reranker"""
    print(f"\n{'='*60}")
    print(f"評估模型: {model_name}")
    print(f"樣本數: {len(eval_data)}")
    print('='*60)
    
    reranker = FlagReranker(model_path, use_fp16=True)
    
    correct = 0
    total = 0
    category_results = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for item in tqdm(eval_data, desc=model_name):
        query = item['query']
        positive = item['positive']
        negative = item['negative']
        category = item['category']
        
        # 計算分數
        pos_score = reranker.compute_score([query, positive])
        neg_score = reranker.compute_score([query, negative])
        
        # 確保是純量
        if hasattr(pos_score, '__len__'):
            pos_score = pos_score[0]
        if hasattr(neg_score, '__len__'):
            neg_score = neg_score[0]
        
        if float(pos_score) > float(neg_score):
            correct += 1
            category_results[category]['correct'] += 1
        
        total += 1
        category_results[category]['total'] += 1
    
    accuracy = correct / total if total > 0 else 0
    
    del reranker
    torch.cuda.empty_cache()
    
    return accuracy, dict(category_results)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 載入評估資料
    print("載入評估資料...")
    eval_data = load_eval_data(EVAL_DATA_PATH)
    print(f"評估樣本數: {len(eval_data)}")
    
    # 評估你的模型
    your_acc, your_cat = evaluate_reranker(
        "/home/peter831/test/outputs2/base_bge_add_library",
        "Your Model",
        eval_data
    )
    
    # 評估 BGE-Reranker-Large
    bge_acc, bge_cat = evaluate_reranker(
        "BAAI/bge-reranker-large",
        "BGE-Reranker-Large",
        eval_data
    )
    
    # 輸出結果
    print("\n" + "=" * 60)
    print("總體結果")
    print("=" * 60)
    print(f"Your Model:         {your_acc:.4f}")
    print(f"BGE-Reranker-Large: {bge_acc:.4f}")
    print(f"差距:               {your_acc - bge_acc:+.4f}")
    
    # 按學門分析
    print("\n" + "=" * 60)
    print("各學門結果")
    print("=" * 60)
    print(f"{'學門':<20} {'Your Model':<12} {'BGE-Large':<12} {'樣本數':<8}")
    print("-" * 60)
    
    all_categories = set(your_cat.keys()) | set(bge_cat.keys())
    for cat in sorted(all_categories):
        your_c = your_cat.get(cat, {'correct': 0, 'total': 0})
        bge_c = bge_cat.get(cat, {'correct': 0, 'total': 0})
        your_cat_acc = your_c['correct'] / your_c['total'] if your_c['total'] > 0 else 0
        bge_cat_acc = bge_c['correct'] / bge_c['total'] if bge_c['total'] > 0 else 0
        sample_count = your_c['total']
        print(f"{cat:<20} {your_cat_acc:<12.4f} {bge_cat_acc:<12.4f} {sample_count:<8}")
    
    # 保存結果
    results = {
        "your_model": {"accuracy": your_acc, "by_category": your_cat},
        "bge_large": {"accuracy": bge_acc, "by_category": bge_cat},
        "eval_data_path": EVAL_DATA_PATH,
        "total_samples": len(eval_data),
        "negative_strategy": "same_category"
    }
    
    with open(f"{OUTPUT_DIR}/thesis_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果已保存到: {OUTPUT_DIR}/large_thesis_eval_results.json")