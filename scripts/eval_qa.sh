#!/bin/bash
#SBATCH --job-name=eval-qa
#SBATCH --partition=gpNCHC_LLM
#SBATCH --account=GOV112003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --exclude=gn1013
#SBATCH --output=/home/peter831/test/logs/eval_qa_%j.out
#SBATCH --error=/home/peter831/test/logs/eval_qa_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /home/peter831/test/scripts/eval_qa_benchmark.py
EOF

mkdir -p /home/peter831/test/logs
sbatch /home/peter831/test/scripts/eval_qa.sh
```

## 兩種評估方式說明

| 方式 | 說明 | 意義 |
|------|------|------|
| **Accuracy** | 正確摘要 vs BM25 最相似的錯誤摘要 | 模型能否區分「對」和「很像但錯」 |
| **Recall@K** | 從 100 個候選中重新排序，看正確答案排第幾 | 更接近真實檢索場景 |

## 預期輸出
```
============================================================
結果比較
============================================================
指標             Your Model           BGE-Large            Diff
----------------------------------------------------------------------
accuracy        0.8500               0.8200               +0.0300
recall@1        0.6000               0.5500               +0.0500
recall@5        0.8500               0.8200               +0.0300
recall@10       0.9200               0.9000               +0.0200