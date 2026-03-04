#eval2.sh
#!/bin/bash

JOB_NAME=eval-reranker
PARTITION=gpNCHC_LLM
ACCOUNT=GOV112003
GPUS_PER_NODE=8
CPUS_PER_TASK=4

COMMAND="source ~/miniconda3/etc/profile.d/conda.sh && conda activate test && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python /home/peter831/test/scripts/eval_qa_benchmark_v2.py"



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