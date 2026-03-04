#!/bin/bash
#eval.sh
JOB_NAME=eval-reranker
PARTITION=gpNCHC_LLM
ACCOUNT=GOV112003
GPUS_PER_NODE=8
CPUS_PER_TASK=4

# ========== 設定 ==========
YOUR_MODEL="/home/peter831/test/outputs1/bge-reranker-reproduce"
BGE_RERANKER="BAAI/bge-reranker-large"
OUTPUT_DIR="/home/peter831/test/eval_results/1"
EMBEDDER="BAAI/bge-base-zh-v1.5"

COMMAND=$(cat << 'INNER'
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 安裝相容版本的 mteb
pip install mteb==1.12.0 --break-system-packages -q

mkdir -p /home/peter831/test/eval_results

echo "========== 評估你的模型 =========="
python -m FlagEmbedding.evaluation.mteb \
    --tasks T2Reranking MMarcoReranking CMedQAv1-reranking CMedQAv2-reranking \
    --output_dir /home/peter831/test/eval_results/your_model \
    --reranker_name_or_path /home/peter831/test/outputs_large/large_bge_add_library \
    --reranker_model_class encoder-only-base \
    --embedder_name_or_path BAAI/bge-base-zh-v1.5 \
    --devices cuda:0 \
    --reranker_query_max_length 64 \
    --reranker_max_length 440 \
    --reranker_batch_size 16 \
    --embedder_batch_size 64 \
    --eval_output_path /home/peter831/test/eval_results/my_model_results.md

echo "========== 評估 BGE-Reranker-Large =========="
python -m FlagEmbedding.evaluation.mteb \
    --tasks T2Reranking MMarcoReranking CMedQAv1-reranking CMedQAv2-reranking \
    --output_dir /home/peter831/test/eval_results/bge_large \
    --reranker_name_or_path BAAI/bge-reranker-large \
    --reranker_model_class encoder-only-base \
    --embedder_name_or_path BAAI/bge-base-zh-v1.5 \
    --devices cuda:0 \
    --reranker_query_max_length 256 \
    --reranker_max_length 512 \
    --reranker_batch_size 8 \
    --embedder_batch_size 64 \
    --eval_output_path /home/peter831/test/eval_results/bge_large_results.md

echo "========== 評估完成 =========="
INNER
)

SBATCH_ARGS=(
    --partition $PARTITION
    --gpus-per-node $GPUS_PER_NODE
    --cpus-per-task $CPUS_PER_TASK
    --ntasks-per-node 1
    --account $ACCOUNT
    --nodes 1
    --job-name $JOB_NAME
    --exclude gn1013
    --output /home/peter831/test/logs/eval_%j.out
    --error /home/peter831/test/logs/eval_%j.err
)

mkdir -p /home/peter831/test/logs

SBATCH_ARGS=${SBATCH_ARGS[@]}
sbatch $SBATCH_ARGS --wrap "$COMMAND"