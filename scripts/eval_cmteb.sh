#!/bin/bash

JOB_NAME=eval-cmteb
PARTITION=gpNCHC_LLM
ACCOUNT=GOV112003
GPUS_PER_NODE=4
CPUS_PER_TASK=8

COMMAND=$(cat << 'INNER'
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

pip install mteb==1.15.0 -q

mkdir -p /home/peter831/test/eval_results

python << PYTHON
from mteb import MTEB
from FlagEmbedding import FlagReranker

# 載入你的模型
model = FlagReranker('/home/peter831/test/outputs/bge-reranker-reproduce', use_fp16=True)

# C-MTEB Reranking 任務
tasks = [
    "T2Reranking",
    "MMarcoReranking", 
    "CMedQAv1-reranking",
    "CMedQAv2-reranking",
]

evaluation = MTEB(tasks=tasks)
results = evaluation.run(
    model,
    output_folder="/home/peter831/test/eval_results/your_model_cmteb",
    eval_splits=["test"],
)

print("評估完成！")
PYTHON
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
    --time 4:00:00
    --output /home/peter831/test/logs/eval_cmteb_%j.out
    --error /home/peter831/test/logs/eval_cmteb_%j.err
)

mkdir -p /home/peter831/test/logs

SBATCH_ARGS=${SBATCH_ARGS[@]}
sbatch $SBATCH_ARGS --wrap "$COMMAND"