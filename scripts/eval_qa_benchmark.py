# eval_qa_benchmark.py
import json
import os
import random
from collections import defaultdict
from rank_bm25 import BM25Okapi
import jieba
from FlagEmbedding import FlagReranker
import torch
from tqdm import tqdm

# 路徑設定
QA_BENCHMARK_PATH = "/home/peter831/test/eval_library/silver_test_benchmark_180qapairs_with_uid.jsonl"
THESIS_DIR = "/home/peter831/test/library"
OUTPUT_DIR = "/home/peter831/test/eval_results"

# BM25 候選數量（用於 Recall@K）
BM25_TOP_K = 100

def tokenize(text):
    """中文分詞"""
    return list(jieba.cut(text))

def is_valid_string(value):
    if value is None:
        return False
    if isinstance(value, float):
        return False
    if not isinstance(value, str):
        return False
    if not value.strip():
        return False
    return True

def load_qa_benchmark(filepath):
    """載入 QA 評估資料"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data

def load_thesis_data(data_dir):
    """載入所有論文資料，建立 uid → 論文 的對應"""
    uid_to_thesis = {}
    all_thesis = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(data_dir, filename)
            print(f"載入 {filename}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        uid = item.get('uid', '')
                        if uid and is_valid_string(item.get('摘要')):
                            uid_to_thesis[uid] = item
                            all_thesis.append(item)
                    except:
                        continue
    
    return uid_to_thesis, all_thesis

def build_bm25_index(all_thesis):
    """建立 BM25 索引"""
    print("建立 BM25 索引...")
    corpus = [str(item['摘要']).strip() for item in all_thesis]
    tokenized_corpus = [tokenize(doc) for doc in tqdm(corpus, desc="分詞")]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, corpus

# ============================================================
# 方式一：Accuracy（二選一）
# ============================================================
def evaluate_accuracy(model_path, model_name, qa_data, uid_to_thesis, bm25, all_thesis, corpus):
    """方式一：Accuracy 評估"""
    print(f"\n{'='*60}")
    print(f"[Accuracy 評估] {model_name}")
    print('='*60)
    
    reranker = FlagReranker(model_path, use_fp16=True)
    
    correct = 0
    total = 0
    
    for item in tqdm(qa_data, desc="Accuracy"):
        uid = item['uid']
        questions = [item.get('問題一'), item.get('問題二')]
        
        # 取得正確論文的摘要
        if uid not in uid_to_thesis:
            continue
        positive = str(uid_to_thesis[uid]['摘要']).strip()
        
        for query in questions:
            if not query:
                continue
            
            # 用 BM25 找 Hard Negative（排除正確答案）
            tokenized_query = tokenize(query)
            scores = bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            
            # 找一個不是正確答案的 negative
            negative = None
            for idx in top_indices[:20]:
                if all_thesis[idx].get('uid') != uid:
                    negative = corpus[idx]
                    break
            
            if not negative:
                continue
            
            # 計算分數
            pos_score = reranker.compute_score([query, positive])
            neg_score = reranker.compute_score([query, negative])
            
            if hasattr(pos_score, '__len__'):
                pos_score = pos_score[0]
            if hasattr(neg_score, '__len__'):
                neg_score = neg_score[0]
            
            if float(pos_score) > float(neg_score):
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    del reranker
    torch.cuda.empty_cache()
    
    return accuracy

# ============================================================
# 方式二：Recall@K（真實檢索場景）
# ============================================================
def evaluate_recall_at_k(model_path, model_name, qa_data, uid_to_thesis, bm25, all_thesis, corpus, top_k=100):
    """方式二：Recall@K 評估"""
    print(f"\n{'='*60}")
    print(f"[Recall@K 評估] {model_name}")
    print('='*60)
    
    reranker = FlagReranker(model_path, use_fp16=True)
    
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    total = 0
    
    for item in tqdm(qa_data, desc="Recall@K"):
        uid = item['uid']
        questions = [item.get('問題一'), item.get('問題二')]
        
        if uid not in uid_to_thesis:
            continue
        
        for query in questions:
            if not query:
                continue
            
            # Step 1: BM25 先撈出 top-K 候選
            tokenized_query = tokenize(query)
            scores = bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            candidates = []
            correct_idx = -1
            for i, idx in enumerate(top_indices):
                thesis = all_thesis[idx]
                candidates.append({
                    'idx': i,
                    'uid': thesis.get('uid'),
                    'abstract': corpus[idx]
                })
                if thesis.get('uid') == uid:
                    correct_idx = i
            
            # 如果正確答案不在 top-K 候選中，跳過
            if correct_idx == -1:
                continue
            
            # Step 2: Reranker 重新排序
            pairs = [[query, c['abstract']] for c in candidates]
            rerank_scores = reranker.compute_score(pairs)
            
            if not hasattr(rerank_scores, '__len__'):
                rerank_scores = [rerank_scores]
            
            # 根據 reranker 分數排序
            ranked_indices = sorted(range(len(rerank_scores)), key=lambda i: rerank_scores[i], reverse=True)
            
            # 找正確答案的新排名
            new_rank = -1
            for rank, idx in enumerate(ranked_indices):
                if candidates[idx]['uid'] == uid:
                    new_rank = rank + 1  # 排名從 1 開始
                    break
            
            if new_rank == 1:
                recall_at_1 += 1
            if new_rank <= 5:
                recall_at_5 += 1
            if new_rank <= 10:
                recall_at_10 += 1
            total += 1
    
    r1 = recall_at_1 / total if total > 0 else 0
    r5 = recall_at_5 / total if total > 0 else 0
    r10 = recall_at_10 / total if total > 0 else 0
    
    print(f"Recall@1:  {r1:.4f} ({recall_at_1}/{total})")
    print(f"Recall@5:  {r5:.4f} ({recall_at_5}/{total})")
    print(f"Recall@10: {r10:.4f} ({recall_at_10}/{total})")
    
    del reranker
    torch.cuda.empty_cache()
    
    return {'recall@1': r1, 'recall@5': r5, 'recall@10': r10, 'total': total}

# ============================================================
# 主程式
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 載入資料
    print("載入 QA 評估資料...")
    qa_data = load_qa_benchmark(QA_BENCHMARK_PATH)
    print(f"QA 評估資料: {len(qa_data)} 筆 ({len(qa_data) * 2} 個問題)")
    
    print("\n載入論文資料...")
    uid_to_thesis, all_thesis = load_thesis_data(THESIS_DIR)
    print(f"論文資料: {len(all_thesis)} 筆")
    
    # 建立 BM25 索引
    bm25, corpus = build_bm25_index(all_thesis)
    
    # 要評估的模型
    models = [
        ("/home/peter831/test/outputs_large/large_bge_add_library", "Your Large Model"),
        ("BAAI/bge-reranker-large", "BGE-Reranker-Large"),("jinaai/jina-reranker-v3", "jina-v3")
    ]
    
    results = {}
    
    for model_path, model_name in models:
        print(f"\n{'#'*60}")
        print(f"評估模型: {model_name}")
        print('#'*60)
        
        # 方式一：Accuracy
        acc = evaluate_accuracy(model_path, model_name, qa_data, uid_to_thesis, bm25, all_thesis, corpus)
        
        # 方式二：Recall@K
        recall = evaluate_recall_at_k(model_path, model_name, qa_data, uid_to_thesis, bm25, all_thesis, corpus)
        
        results[model_name] = {
            'accuracy': acc,
            'recall@1': recall['recall@1'],
            'recall@5': recall['recall@5'],
            'recall@10': recall['recall@10'],
        }
    
    # 輸出比較
    print("\n" + "=" * 70)
    print("結果比較")
    print("=" * 70)
    print(f"{'指標':<15} {'Your Model':<20} {'BGE-Large':<20} {'Diff':<10}")
    print("-" * 70)
    
    your_results = results.get("Your Large Model", {})
    bge_results = results.get("BGE-Reranker-Large", {})
    
    for metric in ['accuracy', 'recall@1', 'recall@5', 'recall@10']:
        your_score = your_results.get(metric, 0)
        bge_score = bge_results.get(metric, 0)
        diff = your_score - bge_score
        print(f"{metric:<15} {your_score:<20.4f} {bge_score:<20.4f} {diff:+.4f}")
    
    # 保存結果
    with open(f"{OUTPUT_DIR}/qa_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果已保存到: {OUTPUT_DIR}/qa_benchmark_results.json")

if __name__ == '__main__':
    main()