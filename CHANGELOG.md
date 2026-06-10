# 改动日志

## [v1.1.0] - 2026-06-10

### 新增 eval_all.py 综合评估

- 支持 Base / SFT / DPO 三模型在同一批图像上对比评估
- 自动识别模型格式：merged 目录直接加载，.pt checkpoint 先加载 base 再载入权重
- `--no_judge` 跳过 LLM Judge 加速评估
- `--no_clip` 跳过 CLIPScore 加速评估
- `report.md` 包含自动结果分析（CLIPScore 对比、描述长度趋势、薄弱维度诊断、训练建议）

### API 统一与速率限制

- `utils/api_client.py` 统一管理所有 LLM API 调用
- 默认使用 MiniMax-M2.7-highspeed 模型
- 内置滑动窗口速率限制：500 RPM / 20,000,000 TPM
- MiniMax thinking 模式自动禁用，确保 JSON 格式输出

### CLIP 模型本地化

- `trainer/grpo_utils.py` CLIP 默认路径改为 `./model/clip-vit-base-patch32`
- `eval_all.py` 同步使用本地 CLIP 路径

### LLM-Judge JSON 解析修复

- `.format()` 花括号转义，防止 prompt 中的 JSON 示例被误解析
- 移除 `<think>` 标签干扰
- `json.loads` 异常安全处理，非 dict 结果降级为默认分

---

## [v1.0.0] - 2026-06-07

### 新建项目

基于 MiniMind-V 魔改，将基座模型替换为 Qwen3-VL-2B-Instruct。

### 改动详情

#### 模型替换

| 文件 | 改动 |
|---|---|
| `model/qwen_vl.py` | 新建，封装 Qwen3-VL 模型加载和前向传播 |
| `model/__init__.py` | 导出 QwenVLM, QwenVLMConfig |

#### 模型加载

- 使用 `AutoModelForImageTextToText` 替代 `AutoModelForCausalLM`
- 冻结策略支持 `visual`/`vit`/`vision` 通配符匹配
- 兼容旧版 transformers (fallback 到 AutoModelForCausalLM)

#### 数据集

| 文件 | 改动 |
|---|---|
| `dataset/sft_dataset.py` | 新建，支持 parquet 格式图文数据处理 |
| `dataset/__init__.py` | 导出 VLMDataset |
