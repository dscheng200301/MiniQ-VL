#!/usr/bin/env bash
# ============================================================
# MiniQ-VL GRPO 训练启动脚本 (重写版)
# ============================================================
# 用法:
#   bash scripts/train_grpo.sh                     # 使用下方默认参数启动
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_grpo.sh
#
# 调参:
#   - 修改下方"参数区"即可, 一般无需改动脚本其它部分
#   - 也可通过环境变量覆盖 (例如 GROUP_SIZE=8 bash scripts/train_grpo.sh)
# ============================================================

set -euo pipefail

# ============== 项目路径 ==============
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || { echo "项目目录不存在: $PROJECT_DIR"; exit 1; }

# ============== 默认参数 ==============
# 使用 ${VAR:-default} 形式: 既能用环境变量覆盖, 又有默认值

# --- 硬件 ---
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
MASTER_PORT=${MASTER_PORT:-29501}              # 跑多任务时记得改, 不要和 SFT 冲突

# --- 模型 / 数据 ---
MODEL_PATH=${MODEL_PATH:-"./model/Qwen3-VL-2B-Instruct"}
SFT_CHECKPOINT=${SFT_CHECKPOINT:-""}           # 留空=自动从 out/ 找最新 sft_*
USE_SFT_MODEL=${USE_SFT_MODEL:-1}              # 1=自动加载 SFT, 0=不用 SFT
USE_BASE_MODEL=${USE_BASE_MODEL:-0}            # 1=用基模直接训练 (跳过 SFT)
DATA_PATH=${DATA_PATH:-"./dataset/minimind-v_dataset/sft_i2t.parquet"}
PREFILTERED_PATH=${PREFILTERED_PATH:-"./dataset/minimind-v_dataset/grpo_i2t.parquet"}  # 预筛选数据
PREPARE_DATASET=${PREPARE_DATASET:-1}          # 1=若 PREFILTERED_PATH 不存在则先跑 prepare_grpo_dataset.py
SAVE_DIR=${SAVE_DIR:-"./out"}
SAVE_WEIGHT=${SAVE_WEIGHT:-"grpo_vlm"}
MAX_SAMPLES=${MAX_SAMPLES:-1000}               # 训练只取前 N 条; 0=全部, dry-run 时设 50

# --- 训练超参 ---
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-4}                    # 每卡 prompt 数 (GRPO 较 SFT 吃显存)
GROUP_SIZE=${GROUP_SIZE:-4}                    # 每 prompt 采样 K 条 (4~8)
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-2}
LEARNING_RATE=${LEARNING_RATE:-5e-6}           # GRPO 推荐 5e-6, 比 SFT 小一个量级
GRAD_CLIP=${GRAD_CLIP:-1.0}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-512}
DTYPE=${DTYPE:-"bfloat16"}                     # bfloat16 / float16
SEED=${SEED:-42}

# --- 优化开关 ---
USE_GRAD_CHECKPOINT=${USE_GRAD_CHECKPOINT:-1}  # 强烈建议开 (ref_model 模式几乎必开)
USE_INPUT_REQUIRE_GRADS=${USE_INPUT_REQUIRE_GRADS:-0}  # 配合 grad ckpt
USE_COMPILE=${USE_COMPILE:-0}                  # torch.compile (DDP 慎用, 可能 hang)
ENABLE_THINKING=${ENABLE_THINKING:-0}          # 1=开启 deepseek thinking 模式 (默认 0=关闭, 走非思考)

# --- 生成参数 ---
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_P=${TOP_P:-0.9}

# --- Reward 权重 ---
W1=${W1:-0.3}    # CLIPScore
W2=${W2:-0.5}    # LLM-Judge
W3=${W3:-0.2}    # Length Penalty
W4=${W4:-0.0}    # Attribute Coverage
W5=${W5:-0.0}    # Diversity Reward
W6=${W6:-0.0}    # Hallucination Penalty
LENGTH_MIN=${LENGTH_MIN:-50}
LENGTH_MAX=${LENGTH_MAX:-300}

# --- KL 约束 ---
USE_REF_MODEL=${USE_REF_MODEL:-0}              # 1=用 ref_model 算 KL (额外显存, 建议先 0)
KL_COEF=${KL_COEF:-0.05}
CLIP_EPS=${CLIP_EPS:-0.2}

# --- 冻结 ---
FREEZE_VISION=${FREEZE_VISION:-1}              # 冻结视觉编码器
FREEZE_LANGUAGE=${FREEZE_LANGUAGE:-0}          # 冻结语言模型 (一般不冻)

# --- 数据筛选 ---
FILTER_MODE=${FILTER_MODE:-"auto"}             # auto / llm / keyword
FILTER_LLM_MODEL=${FILTER_LLM_MODEL:-"deepseek-v4-flash"}  # 同 judge, 默认 deepseek-v4-flash
FILTER_CACHE=${FILTER_CACHE:-""}               # 留空=与 parquet 同目录

# --- 评分器 ---
# 默认走本地路径 (避免训练时联网); 也可用 CLIP_MODEL=openai/clip-vit-base-patch32 强制走 HF Hub
CLIP_MODEL=${CLIP_MODEL:-"./model/clip-vit-base-patch32"}
# LLM 模型 (judge + filter) 默认都用 deepseek-v4-flash, 见 utils/api_client.py
JUDGE_MODEL=${JUDGE_MODEL:-"deepseek-v4-flash"}
JUDGE_API_KEY=${JUDGE_API_KEY:-""}             # 留空=用环境变量 / api_client 内置默认值
JUDGE_BASE_URL=${JUDGE_BASE_URL:-""}

# --- 日志 / 保存 ---
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-"MiniQ-VL-GRPO"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
LOG_INTERVAL=${LOG_INTERVAL:-5}
SAVE_INTERVAL=${SAVE_INTERVAL:-200}

# --- 断点续训 ---
FROM_RESUME=${FROM_RESUME:-0}                  # 1=从 SAVE_DIR 找最新 checkpoint 恢复

# ============== 环境变量 ==============
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 训练时禁止访问 HF Hub (本地已有所有模型; 想联网下资源时设 0)
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
# HF 镜像 (国内机器常用, OFFLINE=0 时才生效; 留注释即可)
# export HF_ENDPOINT=https://hf-mirror.com

# ============== 启动前环境检查 ==============
echo "[Pre-check] 残留进程与显存状态:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader || true
# ============== 准备 GRPO 筛选数据集 (关键词匹配) ==============
# 若 PREFILTERED_PATH 不存在 且 PREPARE_DATASET=1, 自动跑一次 prepare
if [[ "$PREPARE_DATASET" == "1" ]]; then
  if [[ ! -f "$PREFILTERED_PATH" ]]; then
    echo "[Dataset] $PREFILTERED_PATH 不存在, 自动跑 prepare_grpo_dataset.py (关键词匹配)"
    python3 dataset/prepare_grpo_dataset.py \
      --src "$DATA_PATH" \
      --dst "$PREFILTERED_PATH" \
      || { echo "[Dataset] 准备失败, 退出"; exit 1; }
  else
    echo "[Dataset] $PREFILTERED_PATH 已存在, 跳过 prepare"
  fi
fi

# 通信优化 (按需打开)
# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

# 限制可见卡
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

# ============== 启动信息 ==============
echo "============================================================"
echo "[MiniQ-VL GRPO] 启动训练"
echo "  Project        : $PROJECT_DIR"
echo "  GPUs           : $NPROC_PER_NODE  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  Model base     : $MODEL_PATH"
echo "  SFT checkpoint : ${SFT_CHECKPOINT:-<auto: USE_SFT_MODEL=$USE_SFT_MODEL>}"
echo "  Use base model : $USE_BASE_MODEL"
echo "  Data           : $DATA_PATH"
echo "  Prefiltered    : $PREFILTERED_PATH (auto-prepare: $PREPARE_DATASET)"
echo "  Save dir       : $SAVE_DIR"
echo "  Max samples    : $MAX_SAMPLES (0=全部)"
echo "  Effective batch: $((NPROC_PER_NODE * BATCH_SIZE * ACCUMULATION_STEPS)) prompts"
echo "  Group size     : $GROUP_SIZE  (B*K = $((BATCH_SIZE * GROUP_SIZE)) per micro-step)"
echo "  Learning rate  : $LEARNING_RATE"
echo "  dtype          : $DTYPE"
echo "  grad_ckpt      : $USE_GRAD_CHECKPOINT  (input_require_grads=$USE_INPUT_REQUIRE_GRADS)"
echo "  use_ref_model  : $USE_REF_MODEL  (KL coef=$KL_COEF)"
echo "  deepseek think : $ENABLE_THINKING (0=非思考模式, 1=思考模式)"
echo "  Reward weights : w1=$W1 w2=$W2 w3=$W3 w4=$W4 w5=$W5 w6=$W6"
echo "  wandb          : $USE_WANDB  (project=$WANDB_PROJECT)"
echo "============================================================"

# ============== 组装命令 ==============
CMD=(
  torchrun
  --standalone
  --nproc_per_node="$NPROC_PER_NODE"
  --master_port="$MASTER_PORT"
  trainer/train_grpo.py
  --model_path "$MODEL_PATH"
  --save_dir "$SAVE_DIR"
  --save_weight "$SAVE_WEIGHT"
  --max_samples "$MAX_SAMPLES"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --group_size "$GROUP_SIZE"
  --accumulation_steps "$ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE"
  --grad_clip "$GRAD_CLIP"
  --max_seq_len "$MAX_SEQ_LEN"
  --dtype "$DTYPE"
  --seed "$SEED"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --w1 "$W1" --w2 "$W2" --w3 "$W3"
  --w4 "$W4" --w5 "$W5" --w6 "$W6"
  --length_min "$LENGTH_MIN" --length_max "$LENGTH_MAX"
  --kl_coef "$KL_COEF" --clip_eps "$CLIP_EPS"
  --freeze_vision "$FREEZE_VISION" --freeze_language "$FREEZE_LANGUAGE"
  --use_grad_checkpoint "$USE_GRAD_CHECKPOINT"
  --use_compile "$USE_COMPILE"
  --use_wandb "$USE_WANDB"
  --wandb_project "$WANDB_PROJECT"
  --log_interval "$LOG_INTERVAL"
  --save_interval "$SAVE_INTERVAL"
  --from_resume "$FROM_RESUME"
  --use_sft_model "$USE_SFT_MODEL"
  --use_base_model "$USE_BASE_MODEL"
  --data_path "$DATA_PATH"
  --prefiltered_path "$PREFILTERED_PATH"
  --filter_mode "$FILTER_MODE"
  --filter_llm_model "$FILTER_LLM_MODEL"
  --clip_model "$CLIP_MODEL"
  --judge_model "$JUDGE_MODEL"
)

# ============== 可选参数 ==============
[[ "$USE_REF_MODEL" == "1" ]]                && CMD+=(--use_ref_model)
[[ "$USE_INPUT_REQUIRE_GRADS" == "1" ]]      && CMD+=(--use_enable_input_require_grads)
[[ "$ENABLE_THINKING" == "1" ]]              && CMD+=(--enable_thinking)
[[ -n "${WANDB_ENTITY:-}" ]]                 && CMD+=(--wandb_entity "$WANDB_ENTITY")
[[ -n "${SFT_CHECKPOINT:-}" ]]               && CMD+=(--sft_checkpoint "$SFT_CHECKPOINT")
[[ -n "${FILTER_CACHE:-}" ]]                 && CMD+=(--filter_cache "$FILTER_CACHE")
[[ -n "${JUDGE_API_KEY:-}" ]]                && CMD+=(--judge_api_key "$JUDGE_API_KEY")
[[ -n "${JUDGE_BASE_URL:-}" ]]               && CMD+=(--judge_base_url "$JUDGE_BASE_URL")

# ============== 执行 ==============
"${CMD[@]}"
