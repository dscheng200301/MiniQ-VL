#!/usr/bin/env bash
# ============================================================
# MiniQ-VL SFT 训练启动脚本
# 用法：bash scripts/train_sft.sh
# 修改下方"可调参数"区域即可，运行时无需改动其它部分
# ============================================================

set -euo pipefail

# ============== 项目路径（一般无需修改） ==============
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || { echo "项目目录不存在: $PROJECT_DIR"; exit 1; }

# ============== 可调参数（按需修改） ==============

# ---- 硬件 ----
NPROC_PER_NODE=4                  # 使用的 GPU 数量（1/2/4/8...）
CUDA_VISIBLE_DEVICES="0,1,2,3"    # 指定可见卡；留空则用全部
MASTER_PORT=29500                 # 通信端口（多任务并行时改一下）

# ---- 模型 / 数据 ----
MODEL_PATH="./model/Qwen3-VL-2B-Instruct"
DATA_PATH="./dataset/minimind-v_dataset/sft_i2t.parquet"
SAVE_DIR="./out"
SAVE_WEIGHT="sft_vlm"
MAX_SAMPLES=200000                # 最多使用的样本数；0=全部

# ---- 训练超参 ----
EPOCHS=2
BATCH_SIZE=1                      # 每卡 batch size
ACCUMULATION_STEPS=8              # 梯度累积步数（等效 batch = NPROC * BATCH * ACCUM）
LEARNING_RATE=2e-5                # 4卡 DDP 推荐 2e-5；单卡 1e-5
GRAD_CLIP=1.0
MAX_SEQ_LEN=1024
DTYPE="bfloat16"                  # bfloat16 / float16
SEED=42

# ---- 优化开关 ----
USE_GRAD_CHECKPOINT=1             # 1=开启（强烈建议，省显存）
USE_COMPILE=0                     # 1=torch.compile（加速但费显存，DDP 慎用）

# ---- LoRA ----
USE_LORA=1                        # 1=LoRA 训练；0=全量微调
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
LORA_MERGE_AND_SAVE=1             # 训练结束是否合并 LoRA 并保存完整模型

# ---- 日志 ----
USE_WANDB=1
WANDB_PROJECT="MiniQ-VL-SFT"
WANDB_ENTITY=""                   # 留空用默认账号
LOG_INTERVAL=10
SAVE_INTERVAL=500
KEEP_LAST_N=2

# ---- 断点续训 ----
FROM_RESUME=0                     # 1=从 SAVE_DIR 中最新 checkpoint 恢复
PRETRAIN_CHECKPOINT=""            # 手动指定 pretrain 权重路径；留空=自动
USE_PRETRAIN_MODEL=1              # 1=自动加载最新 pretrain

# ============== 环境变量（一般无需修改） ==============
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 通信优化（按需开启）
# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

# 限制可见卡
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

# ============== 启动信息 ==============
echo "============================================================"
echo "[MiniQ-VL SFT] 启动训练"
echo "  Project        : $PROJECT_DIR"
echo "  GPUs           : $NPROC_PER_NODE (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  Model          : $MODEL_PATH"
echo "  Data           : $DATA_PATH"
echo "  Save dir       : $SAVE_DIR"
echo "  Batch (eff.)   : $((NPROC_PER_NODE * BATCH_SIZE * ACCUMULATION_STEPS))"
echo "  Learning rate  : $LEARNING_RATE"
echo "  dtype          : $DTYPE"
echo "  grad_ckpt      : $USE_GRAD_CHECKPOINT"
echo "  use_compile    : $USE_COMPILE"
echo "  LoRA           : $USE_LORA (r=$LORA_R, alpha=$LORA_ALPHA)"
echo "============================================================"

# ============== 组装命令 ==============
CMD=(
  torchrun
  --standalone
  --nproc_per_node="$NPROC_PER_NODE"
  --master_port="$MASTER_PORT"
  trainer/train_sft.py
  --model_path "$MODEL_PATH"
  --data_path "$DATA_PATH"
  --save_dir "$SAVE_DIR"
  --save_weight "$SAVE_WEIGHT"
  --max_samples "$MAX_SAMPLES"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --accumulation_steps "$ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE"
  --grad_clip "$GRAD_CLIP"
  --max_seq_len "$MAX_SEQ_LEN"
  --dtype "$DTYPE"
  --seed "$SEED"
  --use_grad_checkpoint "$USE_GRAD_CHECKPOINT"
  --use_compile "$USE_COMPILE"
  --use_lora "$USE_LORA"
  --lora_r "$LORA_R"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --lora_target_modules "$LORA_TARGET_MODULES"
  --lora_merge_and_save "$LORA_MERGE_AND_SAVE"
  --use_wandb "$USE_WANDB"
  --wandb_project "$WANDB_PROJECT"
  --log_interval "$LOG_INTERVAL"
  --save_interval "$SAVE_INTERVAL"
  --keep_last_n "$KEEP_LAST_N"
  --from_resume "$FROM_RESUME"
  --use_pretrain_model "$USE_PRETRAIN_MODEL"
)

# 可选：wandb entity / 手动 pretrain 路径
[[ -n "$WANDB_ENTITY" ]]   && CMD+=(--wandb_entity "$WANDB_ENTITY")
[[ -n "$PRETRAIN_CHECKPOINT" ]] && CMD+=(--pretrain_checkpoint "$PRETRAIN_CHECKPOINT")

# ============== 执行 ==============
"${CMD[@]}"
