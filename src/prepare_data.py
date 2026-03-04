#!/usr/bin/env python3
"""
資料準備腳本
下載並轉換 Shitao/bge-reranker-data
"""

import os
import json
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm


def download_data(output_dir: str = "./data"):
    """下載 bge-reranker-data"""
    os.makedirs(output_dir, exist_ok=True)
    repo_id = "Shitao/bge-reranker-data"
    
    print(f"下載 {repo_id}...")
    
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
        tar_files = [f for f in files if f.endswith('.tar')]
        
        for tar_file in tqdm(tar_files, desc="下載中"):
            hf_hub_download(
                repo_id=repo_id,
                filename=tar_file,
                repo_type="dataset",
                local_dir=output_dir
            )
        print(f"✓ 下載完成: {output_dir}")
        
    except Exception as e:
        print(f"✗ 下載失敗: {e}")
        print("請手動從 https://huggingface.co/datasets/Shitao/bge-reranker-data 下載")


def convert_format(sample: dict) -> dict:
    """轉換為 FlagEmbedding 格式"""
    if all(k in sample for k in ["query", "pos", "neg"]):
        return sample
    
    converted = {}
    
    # Query
    for key in ["query", "question", "q"]:
        if key in sample:
            converted["query"] = sample[key]
            break
    else:
        return None
    
    # Positive
    for key in ["pos", "positive", "positives", "answer", "passage"]:
        if key in sample:
            val = sample[key]
            converted["pos"] = val if isinstance(val, list) else [val]
            break
    else:
        return None
    
    # Negative
    for key in ["neg", "negative", "negatives", "hard_negatives"]:
        if key in sample:
            val = sample[key]
            converted["neg"] = val if isinstance(val, list) else [val]
            break
    else:
        converted["neg"] = []
    
    return converted


def process_data(data_dir: str, output_file: str):
    """處理並輸出訓練資料"""
    import tarfile
    
    data_dir = Path(data_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    samples = []
    
    # 處理 tar 檔案
    for tar_path in tqdm(list(data_dir.glob("**/*.tar")), desc="處理 tar"):
        try:
            with tarfile.open(tar_path, 'r') as tar:
                for member in tar.getmembers():
                    if member.name.endswith(('.json', '.jsonl')):
                        f = tar.extractfile(member)
                        if f:
                            for line in f.read().decode('utf-8').strip().split('\n'):
                                if line:
                                    try:
                                        item = json.loads(line)
                                        converted = convert_format(item)
                                        if converted:
                                            samples.append(converted)
                                    except:
                                        continue
        except Exception as e:
            print(f"處理 {tar_path} 錯誤: {e}")
    
    # 處理 jsonl 檔案
    for jsonl_path in tqdm(list(data_dir.glob("**/*.jsonl")), desc="處理 jsonl"):
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            converted = convert_format(item)
                            if converted:
                                samples.append(converted)
                        except:
                            continue
        except Exception as e:
            print(f"處理 {jsonl_path} 錯誤: {e}")
    
    # 輸出
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✓ 輸出 {len(samples)} 樣本到: {output_path}")


def create_sample_data(output_file: str = "./data/sample.jsonl"):
    """建立測試用範例資料"""
    samples = [
        {
            "query": "What is Python?",
            "pos": ["Python is a high-level programming language."],
            "neg": ["Java is a programming language.", "The weather is nice."]
        },
        {
            "query": "What is machine learning?",
            "pos": ["Machine learning is a subset of AI."],
            "neg": ["I like pizza.", "The sky is blue."]
        },
    ]
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 擴展到 100 個樣本
    extended = samples * 50
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for s in extended:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    
    print(f"✓ 建立 {len(extended)} 個範例到: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="下載資料")
    parser.add_argument("--convert", action="store_true", help="轉換格式")
    parser.add_argument("--sample", action="store_true", help="建立測試資料")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output", default="./data/train.jsonl")
    
    args = parser.parse_args()
    
    if args.download:
        download_data(args.data_dir)
    if args.convert:
        process_data(args.data_dir, args.output)
    if args.sample:
        create_sample_data()
    
    if not any([args.download, args.convert, args.sample]):
        print("用法:")
        print("  python prepare_data.py --download    # 下載資料")
        print("  python prepare_data.py --convert     # 轉換格式")
        print("  python prepare_data.py --sample      # 建立測試資料")