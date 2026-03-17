# pipeline.py
"""
Reranker 訓練 Pipeline
一個指令完成：資料處理 → 合併 → 訓練 → 評估
"""

import argparse
import subprocess
import os
import json

def run_command(cmd, description):
    """執行指令並顯示狀態"""
    print("\n" + "=" * 60)
    print(f"🚀 {description}")
    print("=" * 60)
    print(f"執行: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ {description} 失敗")
        exit(1)
    print(f"v {description} 完成")

def main():
    parser = argparse.ArgumentParser(description="Reranker 訓練 Pipeline")
    
    # 必要參數
    parser.add_argument("--domain_data_dir", type=str, required=True,
                        help="領域資料目錄路徑")
    parser.add_argument("--query_field", type=str, required=True,
                        help="Query 欄位名稱")
    parser.add_argument("--positive_field", type=str, required=True,
                        help="Positive 欄位名稱")
    
    # 可選參數
    parser.add_argument("--bge_data_path", type=str, 
                        default="/home/peter831/test/data/train_only_zh_en.jsonl",
                        help="BGE 官方訓練資料路徑")
    parser.add_argument("--output_dir", type=str, 
                        default="/home/peter831/test/outputs/new_model",
                        help="模型輸出目錄")
    parser.add_argument("--base_model", type=str,
                        default="FacebookAI/xlm-roberta-large",
                        help="Base model 名稱")
    parser.add_argument("--domain_repeat", type=int, default=5,
                        help="領域資料重複次數")
    parser.add_argument("--num_negatives", type=int, default=7,
                        help="負樣本數量")
    parser.add_argument("--num_workers", type=int, default=32,
                        help="多進程數量")
    parser.add_argument("--skip_training", action="store_true",
                        help="跳過訓練（只做資料處理）")
    parser.add_argument("--skip_eval", action="store_true",
                        help="跳過評估")
    
    args = parser.parse_args()
    
    # 建立工作目錄
    work_dir = os.path.dirname(args.output_dir)
    data_dir = os.path.join(work_dir, "data")
    eval_data_dir = os.path.join(work_dir, "eval_data")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(eval_data_dir, exist_ok=True)
    
    # 設定檔案路徑
    domain_train_path = os.path.join(data_dir, "domain_train.jsonl")
    merged_train_path = os.path.join(data_dir, "train_merged.jsonl")
    eval_data_path = os.path.join(eval_data_dir, "domain_eval.jsonl")
    
    # 產生設定檔
    config = {
        "bge_data_path": args.bge_data_path,
        "domain_data_dir": args.domain_data_dir,
        "output_path": merged_train_path,
        "query_field": args.query_field,
        "positive_field": args.positive_field,
        "domain_repeat": args.domain_repeat,
        "num_negatives": args.num_negatives,
        "bm25_top_k": 10,
        "num_workers": args.num_workers,
        "seed": 42,
        "eval_output_path": eval_data_path,
        "eval_samples": 1000,
    }
    
    config_path = os.path.join(work_dir, "pipeline_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"設定檔已儲存: {config_path}")
    
    # ============================================================
    # Step 1: 資料處理與合併
    # ============================================================
    run_command(
        f"python scripts/merge_data_v2.py --config {config_path}",
        "Step 1: 資料處理與合併"
    )
    
    if args.skip_training:
        print("\n⏭️ 跳過訓練")
        return
    
    # ============================================================
    # Step 2: 訓練
    # ============================================================
    train_cmd = f"""
    sbatch scripts/train.sh \\
        --train_data {merged_train_path} \\
        --output_dir {args.output_dir} \\
        --model_name_or_path {args.base_model}
    """
    run_command(train_cmd.strip(), "Step 2: 提交訓練任務")
    
    if args.skip_eval:
        print("\n⏭️ 跳過評估")
        return
    
    print("\n" + "=" * 60)
    print("📋 後續步驟")
    print("=" * 60)
    print(f"1. 等待訓練完成，模型會存在: {args.output_dir}")
    print(f"2. 執行評估:")
    print(f"   python scripts/eval_domain.py --model_path {args.output_dir} --eval_data {eval_data_path}")

if __name__ == "__main__":
    main()