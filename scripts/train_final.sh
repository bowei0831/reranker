#!/bin/bash
# train_final.sh
# 完整 Pipeline：資料處理 → 訓練 → 評估

set -e  # 遇到錯誤就停止

# ============================================================
# CONFIG：換資料時修改這裡
# ============================================================
export WANDB_PROJECT="bge-reranker-reproduce"
export WANDB_RUN_NAME="large-reranker-final"

# 路徑設定
SCRIPTS_DIR="/home/peter831/test/scripts"
OUTPUT_DIR="/home/peter831/test/outputs_large/large_bge_final"
TRAIN_DATA="/home/peter831/test/data_merged/train_merged.jsonl"

# 模型設定
MODEL_NAME="FacebookAI/xlm-roberta-large"

# 訓練參數
TRAIN_GROUP_SIZE=8
QUERY_MAX_LEN=64
PASSAGE_MAX_LEN=256
LEARNING_RATE=3e-6
NUM_EPOCHS=1
BATCH_SIZE=1
GRAD_ACCUM=16
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01

# SLURM 設定
PARTITION=gpNCHC_LLM
ACCOUNT=GOV112003
NODES=4
GPUS_PER_NODE=8
CPUS_PER_TASK=4

# ============================================================
# Step 1: 資料處理與合併
# ============================================================
echo ""
echo "============================================================"
echo "Step 1: 資料處理與合併"
echo "============================================================"

# 提交資料處理任務並等待完成
MERGE_JOB=$(sbatch --parsable \
    --partition $PARTITION \
    --account $ACCOUNT \
    --nodes 1 \
    --cpus-per-task 32 \
    --job-name merge-data \
    --output /home/peter831/test/logs/merge_%j.out \
    --error /home/peter831/test/logs/merge_%j.err \
    --wrap "source ~/miniconda3/etc/profile.d/conda.sh && conda activate test && python ${SRC_DIR}/merge_data_v2.py")

echo "資料處理任務已提交: Job ID = $MERGE_JOB"
echo "等待資料處理完成..."

# 等待資料處理完成
while squeue -j $MERGE_JOB -h &> /dev/null; do
    sleep 30
    echo "  資料處理中... (Job $MERGE_JOB)"
done
echo "Step 1 完成：資料處理"

# ============================================================
# Step 2: 訓練
# ============================================================
echo ""
echo "============================================================"
echo "Step 2: 訓練模型"
echo "============================================================"

export TRITON_PTXAS_PATH=$CONDA_PREFIX/bin/ptxas
export TRITON_CUOBJDUMP_PATH=$CONDA_PREFIX/bin/cuobjdump
export TRITON_NVDISASM_PATH=$CONDA_PREFIX/bin/nvdisasm
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore"
export HF_DATASETS_CACHE="/home/peter831/.cache/huggingface/datasets"
export DATASETS_VERBOSITY=warning

TRAIN_COMMAND="source ~/miniconda3/etc/profile.d/conda.sh && conda activate test && \
srun torchrun --nproc_per_node=$GPUS_PER_NODE \
    -m FlagEmbedding.finetune.reranker.encoder_only.base \
    --model_name_or_path $MODEL_NAME \
    --train_data $TRAIN_DATA \
    --output_dir $OUTPUT_DIR \
    --train_group_size $TRAIN_GROUP_SIZE \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --learning_rate $LEARNING_RATE \
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
    --seed 42"

TRAIN_JOB=$(sbatch --parsable \
    --partition $PARTITION \
    --account $ACCOUNT \
    --nodes $NODES \
    --gpus-per-node $GPUS_PER_NODE \
    --cpus-per-task $CPUS_PER_TASK \
    --ntasks-per-node 1 \
    --job-name train-reranker \
    --exclude gn1013 \
    --output /home/peter831/test/logs/train_%j.out \
    --error /home/peter831/test/logs/train_%j.err \
    --wrap "$TRAIN_COMMAND")

echo "訓練任務已提交: Job ID = $TRAIN_JOB"
echo "等待訓練完成...（這會需要較長時間）"

# 等待訓練完成
while squeue -j $TRAIN_JOB -h &> /dev/null; do
    sleep 60
    echo "  訓練中... (Job $TRAIN_JOB)"
done
echo "Step 2 完成：訓練"

# ============================================================
# Step 3: 評估
# ============================================================
echo ""
echo "============================================================"
echo "Step 3: 評估模型"
echo "============================================================"

EVAL_COMMAND="source ~/miniconda3/etc/profile.d/conda.sh && conda activate test && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
python ${SCRIPTS_DIR}/eval_reranker.py"

EVAL_JOB=$(sbatch --parsable \
    --partition $PARTITION \
    --account $ACCOUNT \
    --nodes 1 \
    --gpus-per-node 1 \
    --cpus-per-task 4 \
    --job-name eval-reranker \
    --exclude gn1013 \
    --output /home/peter831/test/logs/eval_%j.out \
    --error /home/peter831/test/logs/eval_%j.err \
    --wrap "$EVAL_COMMAND")

echo "評估任務已提交: Job ID = $EVAL_JOB"
echo "等待評估完成..."

# 等待評估完成
while squeue -j $EVAL_JOB -h &> /dev/null; do
    sleep 30
    echo "  評估中... (Job $EVAL_JOB)"
done
echo "Step 3 完成：評估"

# ============================================================
# 完成
# ============================================================
echo ""
echo "============================================================"
echo "Pipeline 完成！"
echo "============================================================"
echo "模型位置: $OUTPUT_DIR"
echo "評估結果: /home/peter831/test/eval_results/"
echo ""
echo "查看訓練 log: cat /home/peter831/test/logs/train_${TRAIN_JOB}.out"
echo "查看評估結果: cat /home/peter831/test/eval_results/thesis_eval_results.json"