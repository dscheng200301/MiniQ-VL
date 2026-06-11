# MiniQ-VL

MiniQ-VL 是一个围绕 Qwen3-VL 构建的轻量多模态训练项目，覆盖监督微调（SFT）、直接偏好优化（DPO）、组相对策略优化（GRPO）与统一评估流程。

项目当前以图像描述任务为主要使用场景。模型结构由加载的 Hugging Face / ModelScope 检查点决定，仓库负责数据处理、LoRA 微调、偏好训练、奖励计算和评估。

## 当前能力

- SFT：支持 LoRA、DDP、混合精度、梯度累积、断点恢复与合并权重导出。
- DPO：使用 chosen/rejected 偏好样本，只计算助手回答区域的序列对数概率，默认启用冻结参考模型。
- GRPO：为每个提示独立生成多条回答，计算组内相对优势，并支持 CLIP、LLM Judge、长度及可选扩展奖励。
- 评估：可统一比较 Base、SFT、DPO 模型，输出 CLIPScore、Self-BLEU、LLM Judge、文本长度和失败统计。
- 测试：包含静态契约与关键训练逻辑测试。

## 项目结构

```text
MiniQ-VL/
├── dataset/
│   ├── prepare_dpo_dataset.py
│   ├── prepare_grpo_dataset.py
│   └── sft_dataset.py
├── model/
│   └── qwen_vl.py
├── scripts/
│   ├── download_eval_data.py
│   ├── train_dpo.sh
│   ├── train_grpo.sh
│   └── train_sft.sh
├── tests/
├── trainer/
│   ├── grpo_utils.py
│   ├── train_dpo.py
│   ├── train_grpo.py
│   ├── train_pretrain.py
│   └── train_sft.py
├── utils/
│   └── api_client.py
└── eval_all.py
```

更详细的代码结构与数据流见 [PROJECT.md](PROJECT.md)。

## 环境安装

建议使用 Linux、CUDA GPU 和 Python 3.10+。

```bash
pip install -r requirements.txt
```

主要依赖包括 PyTorch、Transformers、Accelerate、PEFT、PyArrow、OpenAI SDK 和 ModelScope。

## 准备模型与数据

仓库不包含模型权重与训练数据。默认路径如下：

```text
./model/Qwen3-VL-2B-Instruct
./model/clip-vit-base-patch32
./dataset/minimind-v_dataset/sft_i2t.parquet
./dataset/minimind-v_dataset/dpo_i2t.json
./dataset/minimind-v_dataset/grpo_i2t.parquet
```

可使用 ModelScope 下载基础模型和数据集：

```bash
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-VL-2B-Instruct', local_dir='./model/Qwen3-VL-2B-Instruct')"
python -c "from modelscope import snapshot_download; snapshot_download('gongjy/minimind-v_dataset', local_dir='./dataset/minimind-v_dataset')"
```

CLIP 奖励与评估默认读取本地 `./model/clip-vit-base-patch32`，也可通过命令行参数覆盖。

## 快速开始

### 1. SFT

Linux 下可直接运行：

```bash
bash scripts/train_sft.sh
```

也可自行启动：

```bash
torchrun --nproc_per_node=1 trainer/train_sft.py \
  --model_path ./model/Qwen3-VL-2B-Instruct \
  --data_path ./dataset/minimind-v_dataset/sft_i2t.parquet \
  --use_lora 1
```

默认会将训练权重写入 `./out`；启用 LoRA 合并时，还会生成可直接加载的合并模型目录。

### 2. 准备并运行 DPO

`dataset/prepare_dpo_dataset.py` 当前仍使用脚本内硬编码路径、采样数量和 API 配置。首次使用前请先修改脚本顶部相关常量，再执行：

```bash
python dataset/prepare_dpo_dataset.py
bash scripts/train_dpo.sh
```

直接运行示例：

```bash
torchrun --nproc_per_node=1 trainer/train_dpo.py \
  --model_path ./model/Qwen3-VL-2B-Instruct \
  --sft_checkpoint ./out/sft_vlm_merged \
  --data_path ./dataset/minimind-v_dataset/dpo_i2t.json \
  --use_ref_model 1
```

DPO 默认加载冻结参考模型，因此显存占用会明显高于 SFT。

### 3. 准备并运行 GRPO

生成关键词过滤后的训练集：

```bash
python dataset/prepare_grpo_dataset.py \
  --src ./dataset/minimind-v_dataset/sft_i2t.parquet \
  --dst ./dataset/minimind-v_dataset/grpo_i2t.parquet
```

启动训练：

```bash
bash scripts/train_grpo.sh
```

GRPO 会为每个提示生成 `group_size` 条回答。启用 LLM Judge、参考模型或较大生成组时，训练成本会显著增加。

### 4. 评估模型

下载 COCO Val2017 评估图像：

```bash
python scripts/download_eval_data.py
```

比较 Base、SFT 和 DPO：

```bash
python eval_all.py \
  --mode all \
  --base_model_path ./model/Qwen3-VL-2B-Instruct \
  --sft_checkpoint ./out/sft_vlm_merged \
  --dpo_checkpoint ./out/dpo_<timestamp>/dpo_final.pt \
  --image_dir ./dataset/eval_images \
  --max_samples 100
```

请显式传入当前 DPO 检查点路径，避免使用脚本中的历史时间戳默认值。评估报告默认保存到：

```text
./eval_output/comprehensive_<timestamp>/
```

完整评估说明见 [EVALUATION.md](EVALUATION.md)。

## API 配置

LLM Judge 与部分数据构造流程使用兼容 OpenAI SDK 的接口：

```bash
export API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.minimaxi.com/v1"
```

也支持 `OPENAI_API_KEY`。默认 Judge 模型为 `MiniMax-M2.7-highspeed`，可通过对应命令行参数覆盖。

## 验证

```bash
python -m pytest tests -q
python -m compileall model trainer dataset utils eval_all.py
```

涉及真实模型、GPU、外部 API 的流程仍应在目标环境中进行小规模冒烟测试。

## 已知限制

- 仓库不分发模型、数据集和 CLIP 权重。
- DPO 数据准备脚本尚未提供完整 CLI，使用前需要修改硬编码配置。
- 训练器中存在面向具体入口的内嵌数据集实现，行为应以实际运行的训练入口为准。
- GRPO 的多次生成、CLIP 奖励、LLM Judge 和可选参考模型都会提高时间与显存成本。
- `train_pretrain.py` 已提供预训练入口，但目前没有配套启动脚本，集成度低于 SFT、DPO 和 GRPO。

## 进一步阅读

- [PROJECT.md](PROJECT.md)：架构、模块职责和数据契约
- [EVALUATION.md](EVALUATION.md)：评估命令、指标与报告
- [OPTIMIZATION.md](OPTIMIZATION.md)：优化路线和实验建议
- [CHANGELOG.md](CHANGELOG.md)：近期修复与变更
