#!/bin/bash

export TRITON_PTXAS_PATH=$CONDA_PREFIX/bin/ptxas
export TRITON_CUOBJDUMP_PATH=$CONDA_PREFIX/bin/cuobjdump
export TRITON_NVDISASM_PATH=$CONDA_PREFIX/bin/nvdisasm

export WANDB_PROJECT="bge-reranker-reproduce"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="/home/peter831/.cache/huggingface/datasets"
export DATASETS_VERBOSITY=warning
###
export WANDB_RUN_NAME="large-reranker-bge_library"
###
JOB_NAME=large-reranker
PARTITION=gpNCHC_LLM
ACCOUNT=GOV112003
NODES=4
GPUS_PER_NODE=8
CPUS_PER_TASK=4

# ========== 訓練參數（模仿 BGE 官方設定）==========
MODEL_NAME="FacebookAI/xlm-roberta-large"
TRAIN_DATA="/home/peter831/test/data_merged/train_merged.jsonl"

# 核心超參數
TRAIN_GROUP_SIZE=8          # 1 pos + 15 neg（與資料格式匹配）
QUERY_MAX_LEN=64            # Query 長度（BGE 官方用 256）
OUTPUT_DIR="/home/peter831/test/outputs_large/large_bge_add_library"
PASSAGE_MAX_LEN=256          # Passage 長度
LEARNING_RATE=3e-6           # 學習率（reranker 通常較小）
NUM_EPOCHS=1                 # 先跑 1 epoch 測試
BATCH_SIZE=1                 # 每 GPU batch size
GRAD_ACCUM=16                 # 梯度累積
WARMUP_RATIO=0.1             # Warmup 比例
WEIGHT_DECAY=0.01            # 權重衰減
#TEMPERATURE=1.0              # Softmax 溫度（reranker 通常用 1.0）

# 有效 batch size = BATCH_SIZE × GPUS × GRAD_ACCUM = 2 × 8 × 8 = 128
# 每步處理 queries = 128，每 query 有 16 個 passages

COMMAND=(
    srun torchrun --nproc_per_node=$GPUS_PER_NODE \
    -m FlagEmbedding.finetune.reranker.encoder_only.base \
    --model_name_or_path $MODEL_NAME \
    --train_data $TRAIN_DATA \
    --output_dir $OUTPUT_DIR \
    --train_group_size $TRAIN_GROUP_SIZE \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --learning_rate $LEARNING_RATE \
    #--temperature $TEMPERATURE \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --fp16 \
    --dataloader_num_workers 0 \
    --dataloader_drop_last True \
    --logging_steps 50 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --report_to wandb \
    --run_name $WANDB_RUN_NAME \
    --deepspeed /home/peter831/test/configs/ds_config.json \
    --seed 42
)

COMMAND=${COMMAND[@]}
echo "========== 訓練指令 =========="
echo $COMMAND
echo "=============================="

SBATCH_ARGS=(
    --partition $PARTITION
    --gpus-per-node $GPUS_PER_NODE
    --cpus-per-task $CPUS_PER_TASK
    --ntasks-per-node 1
    --account $ACCOUNT
    --nodes $NODES
    --job-name $JOB_NAME
    --exclude gn1013
)

SBATCH_ARGS=${SBATCH_ARGS[@]}

SBATCH_OUTPUT=$(sbatch $SBATCH_ARGS --wrap "$COMMAND")
echo $SBATCH_OUTPUT

if [[ $SBATCH_OUTPUT != "Submitted batch job"* ]]; then
    exit
fi

JOB_ID=$(echo $SBATCH_OUTPUT | sed "s/Submitted batch job //")

echo "Waiting for the job to start"
while [[ $JOB_STATE != "RUNNING" ]]; do
    JOB_STATE=$(squeue -j $JOB_ID -h -o %T)
    sleep 1
done

echo "The job is running, trying to attach to the output stream ..."
sleep 3

while [[ $JOB_STATE == "RUNNING" ]]; do
    SATTACH_OUTPUT=$(sattach $JOB_ID.0 2>&1 | tee /dev/tty)
    if [[ $SATTACH_OUTPUT == *"Job/step already completing or completed"* ]] \
    || [[ $SATTACH_OUTPUT == *"Socket timed out on send/recv operation"* ]] \
    || [[ $SATTACH_OUTPUT == *"does not look like a jobid"* ]]; then
        break
    fi
    sleep 1
done