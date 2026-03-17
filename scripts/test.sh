#!/bin/bash
# 步驟3: 測試模型

cd "$(dirname "$0")/.."

echo "=========================================="
echo "BGE-Reranker 測試"
echo "=========================================="

python src/test.py --model_path ./outputs/bge-reranker-large

# 若要使用 FlagEmbedding 測試:
# python src/test.py --model_path ./outputs/bge-reranker-large --flag