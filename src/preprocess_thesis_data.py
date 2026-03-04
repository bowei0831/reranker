# preprocess_thesis_data_hard.py
import json
import os
import random
from collections import defaultdict
from rank_bm25 import BM25Okapi
import jieba
from tqdm import tqdm

SEED = 42
MAX_SAMPLES = 5000
TOP_K_NEGATIVE = 3  # 從 BM25 top-k 中隨機選負樣本
DATA_DIR = "/home/peter831/test/library"
OUTPUT_PATH = "/home/peter831/test/eval_data/thesis_eval_dataset_v3.jsonl"

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

def load_thesis_data(data_dir):
    """載入所有論文資料"""
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(data_dir, filename)
            print(f"載入 {filename}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if is_valid_string(item.get('中文關鍵詞')) and is_valid_string(item.get('摘要')):
                            all_data.append(item)
                    except:
                        continue
    return all_data

def tokenize(text):
    """中文分詞"""
    return list(jieba.cut(text))

def build_eval_dataset_hard(all_data, max_samples=1000, seed=42, top_k=10):
    """建立評估資料集：BM25 Hard Negative"""
    random.seed(seed)
    
    print("建立 BM25 索引...")
    # 用摘要建立 BM25 索引
    corpus = [str(item['摘要']).strip() for item in all_data]
    tokenized_corpus = [tokenize(doc) for doc in tqdm(corpus, desc="分詞")]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 隨機抽樣
    sampled_indices = random.sample(range(len(all_data)), min(max_samples, len(all_data)))
    
    eval_data = []
    for idx in tqdm(sampled_indices, desc="建立 Hard Negative"):
        item = all_data[idx]
        query = str(item['中文關鍵詞']).replace('\n', ' ').strip()
        positive = str(item['摘要']).strip()
        uid = item.get('uid', '')
        
        category = item.get('學門', '未知')
        if not is_valid_string(category):
            category = '未知'
        
        # 用 query 搜尋 BM25，找相似的摘要
        tokenized_query = tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        
        # 排序，取 top-k（排除自己）
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        # 找 hard negative（排除自己，從 top 2~top_k+1 中選）
        neg_candidates = []
        for i in top_indices:
            if i != idx and len(neg_candidates) < top_k:
                neg_candidates.append(i)
        
        if neg_candidates:
            neg_idx = random.choice(neg_candidates)
            neg_item = all_data[neg_idx]
            negative = str(neg_item['摘要']).strip()
            neg_uid = neg_item.get('uid', '')
            neg_rank = neg_candidates.index(neg_idx) + 2  # 記錄是 BM25 第幾名
        else:
            # fallback
            neg_idx = random.choice([i for i in range(len(all_data)) if i != idx])
            negative = str(all_data[neg_idx]['摘要']).strip()
            neg_uid = all_data[neg_idx].get('uid', '')
            neg_rank = -1
        
        eval_data.append({
            'query': query,
            'positive': positive,
            'negative': negative,
            'category': category,
            'positive_uid': uid,
            'negative_uid': neg_uid,
            'neg_bm25_rank': neg_rank
        })
    
    return eval_data

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    print("載入論文資料...")
    all_data = load_thesis_data(DATA_DIR)
    print(f"共載入 {len(all_data)} 筆有效資料")
    
    print(f"\n建立 Hard Negative 評估資料集 (max_samples={MAX_SAMPLES})...")
    eval_data = build_eval_dataset_hard(all_data, max_samples=MAX_SAMPLES, seed=SEED, top_k=TOP_K_NEGATIVE)
    print(f"評估樣本數: {len(eval_data)}")
    
    # 儲存
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n已儲存到: {OUTPUT_PATH}")
    
    # 印出範例
    print("\n" + "=" * 60)
    print("範例資料 (前 2 筆)")
    print("=" * 60)
    for i, item in enumerate(eval_data[:2]):
        print(f"\n--- 第 {i+1} 筆 ---")
        print(f"Query: {item['query'][:60]}...")
        print(f"Positive: {item['positive'][:60]}...")
        print(f"Negative: {item['negative'][:60]}...")
        print(f"Neg BM25 Rank: {item['neg_bm25_rank']}")