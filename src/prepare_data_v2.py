# preprocess_data.py
"""
通用資料預處理腳本
換資料時只需修改 CONFIG 區塊
"""

import json
import os
import random
from collections import defaultdict
from rank_bm25 import BM25Okapi
import jieba
from tqdm import tqdm

# ============================================================
# CONFIG：換資料時修改這裡
# ============================================================
CONFIG = {
    # 輸入輸出路徑
    "raw_data_dir": "/home/peter831/test/library",           # 原始資料目錄
    "output_train_path": "/home/peter831/test/data/domain_train.jsonl",  # 訓練資料輸出
    "output_eval_path": "/home/peter831/test/eval_data/domain_eval.jsonl",  # 評估資料輸出
    
    # 欄位對應（根據你的資料格式修改）
    "query_field": "中文關鍵詞",      # 用哪個欄位當 query
    "positive_field": "摘要",         # 用哪個欄位當 positive
    "id_field": "uid",                # 唯一識別碼欄位
    "category_field": "學門",         # 分類欄位（可選，用於分析）
    
    # 處理參數
    "num_negatives": 7,               # 每筆資料的負樣本數量
    "bm25_top_k": 10,                 # 從 BM25 top-k 中選負樣本
    "eval_samples": 1000,             # 評估資料筆數
    "seed": 42,
}
# ============================================================

random.seed(CONFIG["seed"])

def is_valid_string(value):
    """檢查是否為有效字串"""
    if value is None:
        return False
    if isinstance(value, float):
        return False
    if not isinstance(value, str):
        return False
    if not value.strip():
        return False
    return True

def tokenize(text):
    """中文分詞"""
    return list(jieba.cut(text))

def load_raw_data(data_dir):
    """載入原始資料"""
    all_data = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(data_dir, filename)
            print(f"載入 {filename}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        # 檢查必要欄位
                        if (is_valid_string(item.get(CONFIG["query_field"])) and 
                            is_valid_string(item.get(CONFIG["positive_field"]))):
                            all_data.append(item)
                    except:
                        continue
    
    return all_data

def build_bm25_index(all_data):
    """建立 BM25 索引"""
    print("建立 BM25 索引...")
    corpus = [str(item[CONFIG["positive_field"]]).strip() for item in all_data]
    tokenized_corpus = [tokenize(doc) for doc in tqdm(corpus, desc="分詞")]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, corpus

def create_training_data(all_data, bm25, corpus):
    """建立訓練資料（含 BM25 Hard Negative）"""
    print(f"建立訓練資料 (num_negatives={CONFIG['num_negatives']}, top_k={CONFIG['bm25_top_k']})...")
    
    training_data = []
    
    for idx, item in enumerate(tqdm(all_data, desc="處理訓練資料")):
        query = str(item[CONFIG["query_field"]]).replace('\n', ' ').strip()
        positive = str(item[CONFIG["positive_field"]]).strip()
        item_id = item.get(CONFIG["id_field"], str(idx))
        
        # BM25 搜尋
        tokenized_query = tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        # 找 Hard Negative（排除自己）
        neg_candidates = []
        for i in top_indices:
            if i != idx and len(neg_candidates) < CONFIG["bm25_top_k"]:
                neg_candidates.append(i)
        
        # 選取負樣本
        if len(neg_candidates) >= CONFIG["num_negatives"]:
            selected_indices = random.sample(neg_candidates, CONFIG["num_negatives"])
        else:
            selected_indices = neg_candidates
        
        negatives = [corpus[i] for i in selected_indices]
        
        if negatives:
            training_data.append({
                "query": query,
                "pos": [positive],
                "neg": negatives
            })
    
    return training_data

def create_eval_data(all_data, bm25, corpus, num_samples):
    """建立評估資料"""
    print(f"建立評估資料 (num_samples={num_samples})...")
    
    # 隨機抽樣
    sampled_indices = random.sample(range(len(all_data)), min(num_samples, len(all_data)))
    
    eval_data = []
    
    for idx in tqdm(sampled_indices, desc="處理評估資料"):
        item = all_data[idx]
        query = str(item[CONFIG["query_field"]]).replace('\n', ' ').strip()
        positive = str(item[CONFIG["positive_field"]]).strip()
        item_id = item.get(CONFIG["id_field"], str(idx))
        category = item.get(CONFIG["category_field"], "未知")
        if not is_valid_string(category):
            category = "未知"
        
        # BM25 找 Hard Negative
        tokenized_query = tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        # 找一個負樣本
        negative = None
        neg_id = None
        for i in top_indices:
            if i != idx:
                negative = corpus[i]
                neg_id = all_data[i].get(CONFIG["id_field"], str(i))
                break
        
        if negative:
            eval_data.append({
                "query": query,
                "positive": positive,
                "negative": negative,
                "category": category,
                "positive_id": item_id,
                "negative_id": neg_id
            })
    
    return eval_data

def main():
    # 建立輸出目錄
    os.makedirs(os.path.dirname(CONFIG["output_train_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["output_eval_path"]), exist_ok=True)
    
    # 載入資料
    print("=" * 60)
    print("Step 1: 載入原始資料")
    print("=" * 60)
    all_data = load_raw_data(CONFIG["raw_data_dir"])
    print(f"有效資料: {len(all_data)} 筆")
    
    # 建立 BM25 索引
    print("\n" + "=" * 60)
    print("Step 2: 建立 BM25 索引")
    print("=" * 60)
    bm25, corpus = build_bm25_index(all_data)
    
    # 建立訓練資料
    print("\n" + "=" * 60)
    print("Step 3: 建立訓練資料")
    print("=" * 60)
    training_data = create_training_data(all_data, bm25, corpus)
    print(f"訓練資料: {len(training_data)} 筆")
    
    # 建立評估資料
    print("\n" + "=" * 60)
    print("Step 4: 建立評估資料")
    print("=" * 60)
    eval_data = create_eval_data(all_data, bm25, corpus, CONFIG["eval_samples"])
    print(f"評估資料: {len(eval_data)} 筆")
    
    # 儲存訓練資料
    print("\n" + "=" * 60)
    print("Step 5: 儲存資料")
    print("=" * 60)
    
    with open(CONFIG["output_train_path"], 'w', encoding='utf-8') as f:
        for item in tqdm(training_data, desc="儲存訓練資料"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"訓練資料已儲存: {CONFIG['output_train_path']}")
    
    with open(CONFIG["output_eval_path"], 'w', encoding='utf-8') as f:
        for item in tqdm(eval_data, desc="儲存評估資料"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"評估資料已儲存: {CONFIG['output_eval_path']}")
    
    # 印出範例
    print("\n" + "=" * 60)
    print("範例資料")
    print("=" * 60)
    print("\n訓練資料範例:")
    print(json.dumps(training_data[0], ensure_ascii=False, indent=2)[:500] + "...")
    print("\n評估資料範例:")
    print(json.dumps(eval_data[0], ensure_ascii=False, indent=2)[:500] + "...")

if __name__ == '__main__':
    main()