#eval_reranker.py
import json
import os
from FlagEmbedding import FlagReranker
from datasets import load_dataset
import torch
from tqdm import tqdm

# 每個任務最多評估多少筆（加速測試）
MAX_SAMPLES = 500

def evaluate_reranker(model_path, model_name):
    print(f"\n{'='*60}")
    print(f"評估模型: {model_name}")
    print(f"路徑: {model_path}")
    print(f"每任務最多 {MAX_SAMPLES} 筆樣本")
    print('='*60)
    
    reranker = FlagReranker(model_path, use_fp16=True)
    results = {}
    
    tasks = [
        ("T2Reranking", "C-MTEB/T2Reranking"),
        ("Mmarco-reranking", "C-MTEB/Mmarco-reranking"),
        ("CMedQAv1-reranking", "C-MTEB/CMedQAv1-reranking"),
        ("CMedQAv2-reranking", "C-MTEB/CMedQAv2-reranking"),
    ]
    
    for i, (task_name, dataset_name) in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task_name}...")
        try:
            # 自動偵測 split
            try:
                ds = load_dataset(dataset_name, split="dev")
            except ValueError:
                ds = load_dataset(dataset_name, split="test")
            
            # 限制樣本數
            if len(ds) > MAX_SAMPLES:
                ds = ds.select(range(MAX_SAMPLES))
            
            correct = 0
            total = 0
            
            for item in tqdm(ds, desc=task_name):
                query = item["query"]
                positive = item["positive"]
                negative = item["negative"]
                
                # 處理 positive/negative 可能是 list 的情況
                if isinstance(positive, list):
                    positive = positive[0] if positive else ""
                if isinstance(negative, list):
                    negative = negative[0] if negative else ""
                
                # 跳過空的情況
                if not positive or not negative:
                    continue
                
                # 計算分數
                pos_score = reranker.compute_score([query, positive])
                neg_score = reranker.compute_score([query, negative])
                
                # 確保是純量
                if hasattr(pos_score, '__len__') and len(pos_score) == 1:
                    pos_score = pos_score[0]
                if hasattr(neg_score, '__len__') and len(neg_score) == 1:
                    neg_score = neg_score[0]
                
                if float(pos_score) > float(neg_score):
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            results[task_name] = accuracy
            print(f"{task_name} Accuracy: {accuracy:.4f} ({correct}/{total})")
            
        except Exception as e:
            print(f"{task_name} Error: {e}")
            import traceback
            traceback.print_exc()
            results[task_name] = str(e)
    
    del reranker
    torch.cuda.empty_cache()
    
    return results

if __name__ == "__main__":
    os.makedirs("/home/peter831/test/eval_results", exist_ok=True)
    
    # 評估你的模型
    your_results = evaluate_reranker(
        "/home/peter831/test/outputs_large/large_bge_add_library",
        "Your Model (xlm-roberta-large)"
    )
    
    # 評估 BGE-Reranker-Large
    bge_results = evaluate_reranker(
        "BAAI/bge-reranker-large",
        "BGE-Reranker-Large"
    )
    
    # 輸出比較
    print("\n" + "=" * 70)
    print(f"結果比較 (每任務 {MAX_SAMPLES} 筆樣本)")
    print("=" * 70)
    print(f"{'Task':<25} {'Your Model':<15} {'BGE-Large':<15} {'Diff':<10}")
    print("-" * 70)
    
    tasks = ["T2Reranking", "Mmarco-reranking", "CMedQAv1-reranking", "CMedQAv2-reranking"]
    your_total = 0
    bge_total = 0
    count = 0
    
    for task in tasks:
        your_score = your_results.get(task, "N/A")
        bge_score = bge_results.get(task, "N/A")
        
        if isinstance(your_score, float) and isinstance(bge_score, float):
            diff = your_score - bge_score
            diff_str = f"{diff:+.4f}"
            your_total += your_score
            bge_total += bge_score
            count += 1
            your_str = f"{your_score:.4f}"
            bge_str = f"{bge_score:.4f}"
        else:
            diff_str = "N/A"
            your_str = str(your_score)[:12]
            bge_str = str(bge_score)[:12]
        
        print(f"{task:<25} {your_str:<15} {bge_str:<15} {diff_str:<10}")
    
    print("-" * 70)
    if count > 0:
        your_avg = your_total / count
        bge_avg = bge_total / count
        diff_avg = your_avg - bge_avg
        print(f"{'Average':<25} {your_avg:<15.4f} {bge_avg:<15.4f} {diff_avg:+.4f}")
    
    # 保存結果
    all_results = {"your_model": your_results, "bge_large": bge_results, "max_samples": MAX_SAMPLES}
    with open("/home/peter831/test/eval_results/comparison_quick.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果已保存到: /home/peter831/test/eval_results/comparison_quick.json")