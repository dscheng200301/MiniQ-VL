#!/bin/bash
# MiniQ-VL DPO 训练脚本

# 默认参数
MODEL_PATH="${MODEL_PATH:-./model/Qwen3-VL-2B-Instruct}"
SAVE_DIR="${SAVE_DIR:-./out}"
DATA_PATH="${DATA_PATH:-./dataset/minimind-v_dataset/dpo_i2t.json}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
NUM_GPUS="${NUM_GPUS:-1}"

# 解析参数
NUM_WORKERS="${NUM_WORKERS:-0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"

echo "=========================================="
echo "MiniQ-VL DPO Training"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "GPUs: $NUM_GPUS"
echo "=========================================="

# 构建命令
CMD="torchrun --standalone --nproc_per_node=$NUM_GPUS trainer/train_dpo.py \
    --model_path $MODEL_PATH \
    --save_dir $SAVE_DIR \
    --data_path $DATA_PATH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_samples $MAX_SAMPLES \
    --num_workers $NUM_WORKERS \
    --grad_clip $GRAD_CLIP \
    --log_interval $LOG_INTERVAL \
    --save_interval $SAVE_INTERVAL"

echo "Running: $CMD"
eval $CMD