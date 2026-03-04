#!/bin/bash

JOB_NAME=list-tasks
PARTITION=gpNCHC_LLM
ACCOUNT=GOV112003
GPUS_PER_NODE=1
CPUS_PER_TASK=4

COMMAND=$(cat << 'INNER'
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

python << 'PYTHON'
import mteb

# 列出所有任務名稱
all_tasks = mteb.get_tasks()
print("所有任務：")
for task in all_tasks:
    name = task.metadata.name
    task_type = task.metadata.type
    print(f"  {name} ({task_type})")
PYTHON
INNER
)

sbatch --partition $PARTITION --gpus-per-node $GPUS_PER_NODE --cpus-per-task $CPUS_PER_TASK \
    --ntasks-per-node 1 --account $ACCOUNT --nodes 1 --job-name $JOB_NAME --time 0:10:00 \
    --output /home/peter831/test/logs/list2_%j.out --wrap "$COMMAND"