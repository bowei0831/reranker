# merge_training_data.py
import json
import os
import random
from collections import defaultdict
from rank_bm25 import BM25Okapi
import jieba
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np

SEED = 42
random.seed(SEED)

# 路徑設定
TRAIN_DATA_PATH = "/home/peter831/test/data/train_only_zh_en.jsonl"
THESIS_DIR = "/home/peter831/test/library"
OUTPUT_PATH = "/home/peter831/test/data_merged/train_merged.jsonl"

# 論文資料重複次數
THESIS_REPEAT = 5
NUM_NEGATIVES = 7
TOP_K_NEGATIVE = 10

# 多進程數量
NUM_WORKERS = 32  # 國網應該有很多 CPU

# 全域變數（給多進程用）
global_bm25 = None
global_corpus = None
global_tokenized_corpus = None

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

def tokenize(text):
    return list(jieba.cut(text))

def load_thesis_data(data_dir):
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

def init_worker(bm25, corpus):
    """初始化 worker 的全域變數"""
    global global_bm25, global_corpus
    global_bm25 = bm25
    global_corpus = corpus

def process_single_item(args):
    """處理單筆資料"""
    idx, item, num_negatives, top_k = args
    global global_bm25, global_corpus
    
    query = str(item['中文關鍵詞']).replace('\n', ' ').strip()
    positive = str(item['摘要']).strip()
    
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

def convert_thesis_to_training_format_parallel(all_data, num_negatives=7, top_k=10, num_workers=32):
    """多進程版本"""
    
    print("建立 BM25 索引...")
    corpus = [str(item['摘要']).strip() for item in all_data]
    
    # 分詞（這部分也可以多進程，但 jieba 已經有內建優化）
    print("分詞中...")
    jieba.enable_parallel(num_workers)  # jieba 多進程
    tokenized_corpus = [tokenize(doc) for doc in tqdm(corpus, desc="分詞")]
    jieba.disable_parallel()
    
    print("建立 BM25 索引...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    print(f"多進程處理 ({num_workers} workers)...")
    
    # 準備參數
    args_list = [(idx, item, num_negatives, top_k) for idx, item in enumerate(all_data)]
    
    # 多進程處理
    converted = []
    with Pool(num_workers, initializer=init_worker, initargs=(bm25, corpus)) as pool:
        results = list(tqdm(
            pool.imap(process_single_item, args_list, chunksize=100),
            total=len(args_list),
            desc="建立 Hard Negative"
        ))
    
    converted = [r for r in results if r is not None]
    return converted

def main():
    # 1. 載入原始訓練資料
    print("=" * 60)
    print("Step 1: 載入原始訓練資料")
    print("=" * 60)
    train_data = []
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="載入訓練資料"):
            train_data.append(json.loads(line.strip()))
    print(f"原始訓練資料: {len(train_data)} 筆")
    
    # 2. 載入並轉換論文資料
    print("\n" + "=" * 60)
    print("Step 2: 載入並轉換論文資料（BM25 Hard Negative - 多進程）")
    print("=" * 60)
    thesis_data = load_thesis_data(THESIS_DIR)
    print(f"有效論文資料: {len(thesis_data)} 筆")
    
    thesis_converted = convert_thesis_to_training_format_parallel(
        thesis_data, 
        num_negatives=NUM_NEGATIVES, 
        top_k=TOP_K_NEGATIVE,
        num_workers=NUM_WORKERS
    )
    print(f"轉換後論文資料: {len(thesis_converted)} 筆")
    
    # 3. 合併資料
    print("\n" + "=" * 60)
    print(f"Step 3: 合併資料 (論文重複 {THESIS_REPEAT} 次)")
    print("=" * 60)
    
    merged_data = train_data.copy()
    for _ in range(THESIS_REPEAT):
        merged_data.extend(thesis_converted)
    
    random.shuffle(merged_data)
    
    print(f"原始訓練資料: {len(train_data)} 筆")
    print(f"論文資料 (重複 {THESIS_REPEAT} 次): {len(thesis_converted) * THESIS_REPEAT} 筆")
    print(f"合併後總計: {len(merged_data)} 筆")
    print(f"論文佔比: {len(thesis_converted) * THESIS_REPEAT / len(merged_data) * 100:.1f}%")
    
    # 4. 儲存
    print("\n" + "=" * 60)
    print("Step 4: 儲存")
    print("=" * 60)
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in tqdm(merged_data, desc="儲存"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n已儲存到: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()