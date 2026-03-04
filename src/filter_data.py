#!/usr/bin/env python3
"""
資料過濾腳本
只保留中英文資料，刪除其他多語言資料（Mr.TyDi 非英文部分）
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# 要刪除的檔案（Mr.TyDi 多語言，除了英文）
FILES_TO_REMOVE = [
    "mr-tydi_japanese",
    "mr-tydi_indonesian", 
    "mr-tydi_thai",
    "mr-tydi_bengali",
    "mr-tydi_arabic",
    "mr-tydi_telugu",
    "mr-tydi_finnish",
    "mr-tydi_korean",
    "mr-tydi_swahili",
    "mr-tydi_russian",
    "mr-tydi_combined",  # 這個是多語言混合，也刪掉
]

# 要保留的檔案類型
FILES_TO_KEEP = [
    # Chinese
    "cMedQAv2",
    "LCQMC",
    "BQ_neg",
    "ATEC",
    "afqmc",
    "STS-B",
    "PAWSX",
    "dureader",
    "t2ranking",
    "nli_simcse",
    "marco_chinese",
    # English
    "msmarco_hn_train",
    "nq",
    "hotpotqa",
    "fever",
    "mr-tydi_english",  # 英文的 Mr.TyDi 保留
    # Cross-lingual (中英跨語言)
    "msmarco-zh2en",
    "msmarco-en2zh",
]


def should_keep_file(filename: str) -> bool:
    """判斷檔案是否應該保留"""
    # 檢查是否在刪除列表
    for remove_pattern in FILES_TO_REMOVE:
        if remove_pattern in filename:
            return False
    return True


def filter_data(
    input_dir: str,
    output_dir: str,
    dry_run: bool = False
):
    """
    過濾資料，只保留中英文
    
    Args:
        input_dir: 原始資料目錄 (train_15neg)
        output_dir: 輸出目錄
        dry_run: 如果 True，只顯示會做什麼，不實際執行
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ 輸入目錄不存在: {input_dir}")
        return
    
    # 統計
    kept_files = []
    removed_files = []
    kept_samples = 0
    removed_samples = 0
    
    # 找所有 jsonl/json 檔案
    all_files = list(input_path.rglob("*.jsonl")) + list(input_path.rglob("*.json"))
    
    print(f"\n📁 掃描目錄: {input_dir}")
    print(f"📄 找到 {len(all_files)} 個檔案\n")
    
    for file_path in sorted(all_files):
        filename = file_path.name
        
        # 計算樣本數
        with open(file_path, 'r', encoding='utf-8') as f:
            num_samples = sum(1 for line in f if line.strip())
        
        if should_keep_file(filename):
            kept_files.append((filename, num_samples))
            kept_samples += num_samples
            status = "✅ 保留"
        else:
            removed_files.append((filename, num_samples))
            removed_samples += num_samples
            status = "❌ 刪除"
        
        print(f"{status}: {filename} ({num_samples:,} samples)")
    
    print("\n" + "="*60)
    print("📊 統計結果")
    print("="*60)
    print(f"保留: {len(kept_files)} 個檔案, {kept_samples:,} samples")
    print(f"刪除: {len(removed_files)} 個檔案, {removed_samples:,} samples")
    print(f"總計: {len(all_files)} 個檔案, {kept_samples + removed_samples:,} samples")
    
    if dry_run:
        print("\n⚠️  Dry run 模式，未實際執行")
        return
    
    # 實際複製保留的檔案
    print(f"\n📦 複製檔案到: {output_dir}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    for file_path in tqdm(all_files, desc="複製中"):
        if should_keep_file(file_path.name):
            # 保持目錄結構
            rel_path = file_path.relative_to(input_path)
            dest_path = output_path / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_path)
    
    print(f"\n✅ 完成！過濾後的資料在: {output_dir}")
    print(f"   共 {kept_samples:,} samples")


def merge_to_single_file(
    input_dir: str,
    output_file: str
):
    """
    將過濾後的資料合併成單一 jsonl 檔案
    （FlagEmbedding 可以接受目錄或單一檔案）
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_files = list(input_path.rglob("*.jsonl")) + list(input_path.rglob("*.json"))
    
    total_samples = 0
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for file_path in tqdm(all_files, desc="合併中"):
            with open(file_path, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line.strip() + '\n')
                        total_samples += 1
    
    print(f"\n✅ 合併完成: {output_file}")
    print(f"   共 {total_samples:,} samples")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="過濾 BGE reranker 訓練資料，只保留中英文")
    parser.add_argument(
        "--input_dir",
        default="/home/peter831/test/data/train_15neg",
        help="原始資料目錄"
    )
    parser.add_argument(
        "--output_dir",
        default="/home/peter831/test/data/train_only_zh_en",
        help="過濾後的輸出目錄"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只顯示會做什麼，不實際執行"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="合併成單一檔案"
    )
    parser.add_argument(
        "--merge_output",
        default="/home/peter831/test/data/train_only_zh_en.jsonl",
        help="合併後的輸出檔案"
    )
    
    args = parser.parse_args()
    
    # 先過濾
    filter_data(args.input_dir, args.output_dir, args.dry_run)
    
    # 如果需要合併
    if args.merge and not args.dry_run:
        merge_to_single_file(args.output_dir, args.merge_output)