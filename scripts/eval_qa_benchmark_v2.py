# eval_qa_benchmark.py
import json
import os
import random
from collections import defaultdict
from rank_bm25 import BM25Okapi
import jieba
import torch
from tqdm import tqdm

# 路徑設定
QA_BENCHMARK_PATH = "/home/peter831/test/eval_library/silver_test_benchmark_180qapairs_with_uid.jsonl"
THESIS_DIR = "/home/peter831/test/library"
OUTPUT_DIR = "/home/peter831/test/eval_results"

BM25_TOP_K = 100

def tokenize(text):
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
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data

def load_thesis_data(data_dir):
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
    print("建立 BM25 索引...")
    corpus = [str(item['摘要']).strip() for item in all_thesis]
    tokenized_corpus = [tokenize(doc) for doc in tqdm(corpus, desc="分詞")]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, corpus

# ============================================================
# 載入不同類型的 Reranker
# ============================================================
def load_reranker(model_path, model_type="flag"):
    """根據模型類型載入 reranker"""
    if model_type == "jina":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).cuda().eval()
        return {"type": "jina", "model": model, "tokenizer": tokenizer}
    else:
        from FlagEmbedding import FlagReranker
        reranker = FlagReranker(model_path, use_fp16=True)
        return {"type": "flag", "reranker": reranker}

def compute_score(reranker_obj, query, passage):
    """計算單筆分數"""
    if reranker_obj["type"] == "jina":
        model = reranker_obj["model"]
        tokenizer = reranker_obj["tokenizer"]
        inputs = tokenizer(
            [[query, passage]], 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
            score = outputs.logits.squeeze().item()
        return score
    else:
        score = reranker_obj["reranker"].compute_score([query, passage])
        if hasattr(score, '__len__'):
            score = score[0]
        return float(score)

def compute_scores_batch(reranker_obj, pairs):
    """批次計算分數"""
    if reranker_obj["type"] == "jina":
        model = reranker_obj["model"]
        tokenizer = reranker_obj["tokenizer"]
        inputs = tokenizer(
            pairs, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
            scores = outputs.logits.squeeze().tolist()
        if not isinstance(scores, list):
            scores = [scores]
        return scores
    else:
        scores = reranker_obj["reranker"].compute_score(pairs)
        if not hasattr(scores, '__len__'):
            scores = [scores]
        return [float(s) for s in scores]

# ============================================================
# 方式一：Accuracy
# ============================================================
def evaluate_accuracy(reranker_obj, model_name, qa_data, uid_to_thesis, bm25, all_thesis, corpus):
    print(f"\n{'='*60}")
    print(f"[Accuracy 評估] {model_name}")
    print('='*60)
    
    correct = 0
    total = 0
    
    for item in tqdm(qa_data, desc="Accuracy"):
        uid = item['uid']
        questions = [item.get('問題一'), item.get('問題二')]
        
        if uid not in uid_to_thesis:
            continue
        positive = str(uid_to_thesis[uid]['摘要']).strip()
        
        for query in questions:
            if not query:
                continue
            
            tokenized_query = tokenize(query)
            scores = bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            
            negative = None
            for idx in top_indices[:20]:
                if all_thesis[idx].get('uid') != uid:
                    negative = corpus[idx]
                    break
            
            if not negative:
                continue
            
            pos_score = compute_score(reranker_obj, query, positive)
            neg_score = compute_score(reranker_obj, query, negative)
            
            if pos_score > neg_score:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return accuracy

# ============================================================
# 方式二：Recall@K
# ============================================================
def evaluate_recall_at_k(reranker_obj, model_name, qa_data, uid_to_thesis, bm25, all_thesis, corpus, top_k=100):
    print(f"\n{'='*60}")
    print(f"[Recall@K 評估] {model_name}")
    print('='*60)
    
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
            
            if correct_idx == -1:
                continue
            
            pairs = [[query, c['abstract']] for c in candidates]
            rerank_scores = compute_scores_batch(reranker_obj, pairs)
            
            ranked_indices = sorted(range(len(rerank_scores)), key=lambda i: rerank_scores[i], reverse=True)
            
            new_rank = -1
            for rank, idx in enumerate(ranked_indices):
                if candidates[idx]['uid'] == uid:
                    new_rank = rank + 1
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
    
    return {'recall@1': r1, 'recall@5': r5, 'recall@10': r10, 'total': total}

# ============================================================
# 主程式
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("載入 QA 評估資料...")
    qa_data = load_qa_benchmark(QA_BENCHMARK_PATH)
    print(f"QA 評估資料: {len(qa_data)} 筆 ({len(qa_data) * 2} 個問題)")
    
    print("\n載入論文資料...")
    uid_to_thesis, all_thesis = load_thesis_data(THESIS_DIR)
    print(f"論文資料: {len(all_thesis)} 筆")
    
    bm25, corpus = build_bm25_index(all_thesis)
    
    # ============================================================
    # 三個模型：你的模型、BGE、Jina
    # ============================================================
    models = [
        ("/home/peter831/test/outputs_large/large_bge_add_library", "Your Large Model", "flag"),
        ("BAAI/bge-reranker-large", "BGE-Reranker-Large", "flag"),
        ("jinaai/jina-reranker-v2-base-multilingual", "Jina-Reranker-v2", "jina"),
    ]
    
    results = {}
    
    for model_path, model_name, model_type in models:
        print(f"\n{'#'*60}")
        print(f"評估模型: {model_name}")
        print(f"類型: {model_type}")
        print('#'*60)
        
        # 載入模型
        reranker_obj = load_reranker(model_path, model_type)
        
        # 方式一：Accuracy
        acc = evaluate_accuracy(reranker_obj, model_name, qa_data, uid_to_thesis, bm25, all_thesis, corpus)
        
        # 方式二：Recall@K
        recall = evaluate_recall_at_k(reranker_obj, model_name, qa_data, uid_to_thesis, bm25, all_thesis, corpus)
        
        results[model_name] = {
            'accuracy': acc,
            'recall@1': recall['recall@1'],
            'recall@5': recall['recall@5'],
            'recall@10': recall['recall@10'],
        }
        
        # 清理記憶體
        if model_type == "jina":
            del reranker_obj["model"]
            del reranker_obj["tokenizer"]
        else:
            del reranker_obj["reranker"]
        del reranker_obj
        torch.cuda.empty_cache()
    
    # ============================================================
    # 輸出三個模型比較
    # ============================================================
    print("\n" + "=" * 85)
    print("結果比較")
    print("=" * 85)
    print(f"{'指標':<12} {'Your Model':<18} {'BGE-Large':<18} {'Jina-v2':<18}")
    print("-" * 85)
    
    your_results = results.get("Your Large Model", {})
    bge_results = results.get("BGE-Reranker-Large", {})
    jina_results = results.get("Jina-Reranker-v2", {})
    
    for metric in ['accuracy', 'recall@1', 'recall@5', 'recall@10']:
        your_score = your_results.get(metric, 0)
        bge_score = bge_results.get(metric, 0)
        jina_score = jina_results.get(metric, 0)
        print(f"{metric:<12} {your_score:<18.4f} {bge_score:<18.4f} {jina_score:<18.4f}")
    
    # 輸出差距
    print("\n" + "-" * 85)
    print("與 Your Model 的差距")
    print("-" * 85)
    print(f"{'指標':<12} {'vs BGE-Large':<18} {'vs Jina-v2':<18}")
    print("-" * 85)
    
    for metric in ['accuracy', 'recall@1', 'recall@5', 'recall@10']:
        your_score = your_results.get(metric, 0)
        bge_score = bge_results.get(metric, 0)
        jina_score = jina_results.get(metric, 0)
        diff_bge = your_score - bge_score
        diff_jina = your_score - jina_score
        print(f"{metric:<12} {diff_bge:+.4f}            {diff_jina:+.4f}")
    
    # 保存結果
    with open(f"{OUTPUT_DIR}/qa_benchmark_3models.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果已保存到: {OUTPUT_DIR}/qa_benchmark_3models.json")

if __name__ == '__main__':
    main()