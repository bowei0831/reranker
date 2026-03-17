# merge_training_data.py
"""
通用資料合併腳本（多進程版本）
換資料時只需修改 CONFIG 區塊
"""

import json
import os
import random
from collections import defaultdict
from rank_bm25 import BM25Okapi
import jieba
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np

# ============================================================
# CONFIG：換資料時修改這裡
# ============================================================
CONFIG = {
    # 輸入路徑
    "bge_data_path": "/home/peter831/test/data/train_only_zh_en.jsonl",  # BGE 官方資料
    "domain_data_dir": "/home/peter831/test/library",                    # 領域資料目錄
    
    # 輸出路徑
    "output_path": "/home/peter831/test/data_merged/train_merged.jsonl",
    
    # 領域資料欄位對應（根據你的資料格式修改）
    "query_field": "中文關鍵詞",      # 用哪個欄位當 query
    "positive_field": "摘要",         # 用哪個欄位當 positive
    
    # 處理參數
    "domain_repeat": 5,               # 領域資料重複次數
    "num_negatives": 7,               # 每筆資料的負樣本數量
    "bm25_top_k": 10,                 # 從 BM25 top-k 中選負樣本
    
    # 多進程參數
    "num_workers": 32,                # 國網 CPU 多，可以開大一點
    
    "seed": 42,
}
# ============================================================

random.seed(CONFIG["seed"])

# 全域變數（給多進程用）
global_bm25 = None
global_corpus = None

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

def load_domain_data(data_dir):
    """載入領域資料"""
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

def init_worker(bm25, corpus):
    """初始化 worker 的全域變數"""
    global global_bm25, global_corpus
    global_bm25 = bm25
    global_corpus = corpus

def process_single_item(args):
    """處理單筆資料"""
    idx, item, num_negatives, top_k = args
    global global_bm25, global_corpus
    
    query = str(item[CONFIG["query_field"]]).replace('\n', ' ').strip()
    positive = str(item[CONFIG["positive_field"]]).strip()
    
    # BM25 搜尋
    tokenized_query = tokenize(query)
    scores = global_bm25.get_scores(tokenized_query)
    
    # 取 top-k（排除自己）
    top_indices = np.argsort(scores)[::-1]
    
    neg_candidates = []
    for i in top_indices:
        if i != idx and len(neg_candidates) < top_k:
            neg_candidates.append(i)
    
    # 選負樣本
    if len(neg_candidates) >= num_negatives:
        selected_indices = random.sample(neg_candidates, num_negatives)
    else:
        selected_indices = neg_candidates
    
    negatives = [global_corpus[i] for i in selected_indices]
    
    if negatives:
        return {
            "query": query,
            "pos": [positive],
            "neg": negatives
        }
    return None

def convert_domain_to_training_format(all_data):
    """多進程版本：將領域資料轉換成訓練格式"""
    
    num_negatives = CONFIG["num_negatives"]
    top_k = CONFIG["bm25_top_k"]
    num_workers = CONFIG["num_workers"]
    
    print("建立 BM25 索引...")
    corpus = [str(item[CONFIG["positive_field"]]).strip() for item in all_data]
    
    # 分詞
    print("分詞中...")
    jieba.enable_parallel(num_workers)
    tokenized_corpus = [tokenize(doc) for doc in tqdm(corpus, desc="分詞")]
    jieba.disable_parallel()
    
    print("建立 BM25 索引...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    print(f"多進程處理 ({num_workers} workers)...")
    
    # 準備參數
    args_list = [(idx, item, num_negatives, top_k) for idx, item in enumerate(all_data)]
    
    # 多進程處理
    with Pool(num_workers, initializer=init_worker, initargs=(bm25, corpus)) as pool:
        results = list(tqdm(
            pool.imap(process_single_item, args_list, chunksize=100),
            total=len(args_list),
            desc="建立 Hard Negative"
        ))
    
    converted = [r for r in results if r is not None]
    return converted

def main():
    # 建立輸出目錄
    os.makedirs(os.path.dirname(CONFIG["output_path"]), exist_ok=True)
    
    # 1. 載入 BGE 官方訓練資料
    print("=" * 60)
    print("Step 1: 載入 BGE 官方訓練資料")
    print("=" * 60)
    bge_data = []
    with open(CONFIG["bge_data_path"], 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="載入 BGE 資料"):
            bge_data.append(json.loads(line.strip()))
    print(f"BGE 訓練資料: {len(bge_data)} 筆")
    
    # 2. 載入並轉換領域資料
    print("\n" + "=" * 60)
    print("Step 2: 載入並轉換領域資料（BM25 Hard Negative - 多進程）")
    print("=" * 60)
    domain_data = load_domain_data(CONFIG["domain_data_dir"])
    print(f"有效領域資料: {len(domain_data)} 筆")
    
    domain_converted = convert_domain_to_training_format(domain_data)
    print(f"轉換後領域資料: {len(domain_converted)} 筆")
    
    # 3. 合併資料
    print("\n" + "=" * 60)
    print(f"Step 3: 合併資料 (領域資料重複 {CONFIG['domain_repeat']} 次)")
    print("=" * 60)
    
    merged_data = bge_data.copy()
    for _ in range(CONFIG["domain_repeat"]):
        merged_data.extend(domain_converted)
    
    random.shuffle(merged_data)
    
    print(f"BGE 訓練資料: {len(bge_data)} 筆")
    print(f"領域資料 (重複 {CONFIG['domain_repeat']} 次): {len(domain_converted) * CONFIG['domain_repeat']} 筆")
    print(f"合併後總計: {len(merged_data)} 筆")
    print(f"領域佔比: {len(domain_converted) * CONFIG['domain_repeat'] / len(merged_data) * 100:.1f}%")
    
    # 4. 儲存
    print("\n" + "=" * 60)
    print("Step 4: 儲存")
    print("=" * 60)
    
    with open(CONFIG["output_path"], 'w', encoding='utf-8') as f:
        for item in tqdm(merged_data, desc="儲存"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n已儲存到: {CONFIG['output_path']}")

if __name__ == "__main__":
    main()