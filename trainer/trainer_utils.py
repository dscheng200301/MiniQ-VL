"""
MiniQ-VL 训练工具函数集合
魔改点:
1. 使用 wandb 替代 swanlab 进行训练监控
2. 集成 tqdm 进度条
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import math
import time
import json
import platform
import traceback as tb
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from tqdm import tqdm


def is_main_process():
    """判断是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_project_root():
    """获取项目根目录（trainer/ 的父目录）"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(path):
    """
    将相对路径转换为基于项目根目录的绝对路径
    如果路径是绝对路径或 HF Hub ID，则保持不变
    """
    if path is None:
        return None
    # HF Hub ID 或已绝对路径保持不变
    if os.path.isabs(path) or "/" in path and not path.startswith("./") and not path.startswith("../"):
        return path
    # 相对路径转换为绝对路径
    project_root = get_project_root()
    return os.path.join(project_root, path)


def Logger(content, pbar=None):
    """日志打印，支持 tqdm 进度条同步更新"""
    if is_main_process():
        if pbar is not None:
            pbar.write(content)
        else:
            print(content)


def get_lr(current_step: int, total_steps: int, lr: float) -> float:
    """余弦退火学习率调度"""
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode(auto_ddp: bool = True):
    """
    初始化分布式训练

    优先级:
    1. 已通过 torchrun / accelerate 启动 (RANK 环境变量存在)
    2. auto_ddp=True 且检测到多 GPU → 自动启动单机多卡 DDP
    3. 单 GPU 模式

    Args:
        auto_ddp: 是否在检测到多 GPU 时自动启动 DDP

    Returns:
        local_rank: 本地 GPU 编号
    """
    # 方式 1: 已通过 torchrun 启动
    if int(os.environ.get("RANK", -1)) != -1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        Logger(f"DDP initialized via torchrun, world_size={dist.get_world_size()}")
        return local_rank

    # 方式 2: 自动检测多 GPU
    num_gpus = torch.cuda.device_count()
    if auto_ddp and num_gpus > 1:
        Logger(f"Auto-DDP: detected {num_gpus} GPUs, launching distributed training")
        _launch_ddp_auto(num_gpus)
        # _launch_ddp_auto 会重新启动进程，不会执行到这里
        return 0

    # 方式 3: 单 GPU
    return 0


def _launch_ddp_auto(num_gpus: int):
    """
    自动启动单机多卡 DDP 训练

    通过 subprocess 重新以 torchrun 方式启动当前脚本，
    保留原始命令行参数
    """
    import subprocess

    # 获取当前脚本和参数
    script = os.path.abspath(sys.argv[0])
    args = sys.argv[1:]

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        script,
    ] + args

    Logger(f"Auto-DDP command: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def setup_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_params(model, ignore_patterns=None):
    """获取模型参数统计"""
    if ignore_patterns is None:
        ignore_patterns = ['visual']
    
    def should_count(name):
        return not any(p in name for p in ignore_patterns)
    
    total = sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    trainable = sum(p.numel() for n, p in model.named_parameters() 
                    if should_count(n) and p.requires_grad) / 1e6
    
    Logger(f'Model Params: {total:.2f}M')
    Logger(f'Trainable Params: {trainable:.2f}M')
    return total, trainable


def init_vlm_model(
    model_path: str = "Qwen/Qwen3-VL-2B-Instruct",
    freeze_vision: bool = True,
    freeze_language: bool = False,
    device: str = "cuda"
):
    """
    初始化 VLM 模型
    
    Args:
        model_path: Qwen-VL 模型路径
        freeze_vision: 是否冻结视觉编码器
        freeze_language: 是否冻结语言模型
        device: 设备
    
    Returns:
        model: QwenVLM 模型
        processor: 处理器
    """
    from model.qwen_vl import QwenVLM, QwenVLMConfig
    
    config = QwenVLMConfig(
        model_path=model_path,
        freeze_vision=freeze_vision,
        freeze_language=freeze_language
    )
    model = QwenVLM(config).to(device)
    processor = model.processor
    
    get_model_params(model)
    
    return model, processor


def vlm_checkpoint(
    model=None,
    optimizer=None,
    epoch: int = 0,
    step: int = 0,
    wandb_run=None,
    save_dir: str = '../checkpoints',
    save_weight: str = 'qwen_vl',
    scaler=None,
    **kwargs
):
    """
    保存/加载模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前 epoch
        step: 当前 step
        wandb_run: wandb run 对象
        save_dir: 保存目录
        save_weight: 保存权重名称
        scaler: GradScaler
    """
    os.makedirs(save_dir, exist_ok=True)
    ckp_path = f'{save_dir}/{save_weight}_checkpoint.pt'

    if model is not None:
        # 保存模式
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)

        # 获取 wandb id
        wandb_id = None
        if wandb_run is not None:
            wandb_id = getattr(wandb_run, 'id', None)

        resume_data = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }

        # 添加额外数据（如 scaler）
        for key, value in kwargs.items():
            if value is not None and hasattr(value, 'state_dict'):
                raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                raw_value = getattr(raw_value, '_orig_mod', raw_value)
                resume_data[key] = raw_value.state_dict()

        # 保存检查点（原子写入）
        ckp_tmp = ckp_path + '.tmp'
        torch.save(resume_data, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        Logger(f'Checkpoint saved: {ckp_path}')

        del resume_data
        torch.cuda.empty_cache()

    else:
        # 加载模式
        if os.path.exists(ckp_path):
            ckp_data = torch.load(ckp_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1

            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU 数量变化 ({saved_ws} → {current_ws})，step 已自动转换')

            return ckp_data

        return None


def vlm_collate_fn(batch):
    """DataLoader 整理函数"""
    from dataset.sft_dataset import vlm_collate_fn as base_collate
    return base_collate(batch)


def find_latest_checkpoint(base_dir: str, stage_prefix: str = None) -> tuple:
    """
    查找最新的检查点目录和模型路径
    
    Args:
        base_dir: 基准目录 (如 ./out)
        stage_prefix: 阶段前缀过滤 (如 "sft", "grpo", "pretrain")
    
    Returns:
        (latest_dir, checkpoint_path) 或 (None, None)
    """
    if not os.path.exists(base_dir):
        return None, None
    
    # 收集所有子目录
    subdirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # 如果指定了前缀，只保留匹配的项目
            if stage_prefix is None or item.startswith(stage_prefix):
                subdirs.append((item, item_path))
    
    if not subdirs:
        return None, None
    
    # 按名称排序（名称含时间戳，排序后最新的在最后）
    subdirs.sort(key=lambda x: x[0])
    latest_name, latest_dir = subdirs[-1]
    
    Logger(f"Found latest checkpoint directory: {latest_dir}")
    
    return latest_dir, None


class SkipBatchSampler(Sampler):
    """跳过指定 batch 的采样器"""
    
    def __init__(self, sampler, batch_size: int, skip_batches: int = 0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches
    
    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch
    
    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class TrainingProgressBar:
    """训练进度条封装类"""
    
    def __init__(self, total_steps: int, desc: str = "Training", log_interval: int = 10):
        self.total_steps = total_steps
        self.desc = desc
        self.log_interval = log_interval
        self.pbar = None
        self.start_time = None
    
    def __enter__(self):
        if is_main_process():
            self.pbar = tqdm(total=self.total_steps, desc=self.desc, unit='step')
            self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        if self.pbar is not None:
            self.pbar.close()
    
    def update(self, step: int, loss: float = None, lr: float = None):
        """更新进度"""
        if self.pbar is not None:
            # 更新描述信息
            desc = f"{self.desc} [Loss: {loss:.4f}]" if loss is not None else self.desc
            if lr is not None:
                desc += f" [LR: {lr:.2e}]"
            self.pbar.set_description(desc)
            
            # 更新进度
            self.pbar.update(1)
    
    def write(self, content: str):
        """写入日志"""
        if self.pbar is not None:
            self.pbar.write(content)
    
    def set_postfix(self, **kwargs):
        """设置后缀信息"""
        if self.pbar is not None:
            self.pbar.set_postfix(**kwargs)


def save_error_report(
    error: BaseException,
    save_dir: str = None,
    args = None,
    last_step: int = None,
    extra: dict = None,
):
    """
    训练失败时落盘错误报告（在主进程调用，避免 DDP 文件冲突）

    生成文件:
        - {save_dir}/error_report.json   结构化信息（args / 系统 / 错误摘要）
        - {save_dir}/error_traceback.txt 完整 traceback
        - {save_dir}/error_summary.md    人类可读摘要

    Args:
        error: 异常对象
        save_dir: 保存目录；为 None 时只打印
        args: argparse Namespace（会序列化成 JSON）
        last_step: 最后一次成功的 step（用于定位断点）
        extra: 其它附加信息（如 loaded_sft_path、ckp_path）
    """
    # DDP 下只在主进程写
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    tb_text = ''.join(tb.format_exception(type(error), error, error.__traceback__))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 收集 GPU 信息
    gpu_info = []
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 1),
                    "memory_allocated_gb": round(torch.cuda.memory_allocated(i) / 1024**3, 2),
                    "memory_reserved_gb": round(torch.cuda.memory_reserved(i) / 1024**3, 2),
                })
    except Exception:
        pass

    # 组装 report
    report = {
        "timestamp": timestamp,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "last_step": last_step,
        "ddp": {
            "initialized": dist.is_initialized(),
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "rank": dist.get_rank() if dist.is_initialized() else 0,
        },
        "system": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
        "gpus": gpu_info,
    }
    if args is not None:
        # Namespace → dict
        try:
            report["args"] = {k: (str(v) if not isinstance(v, (int, float, bool, str, list, dict, type(None))) else v)
                              for k, v in vars(args).items()}
        except Exception:
            report["args"] = str(args)
    if extra:
        report["extra"] = extra

    # 控制台打印
    print("\n" + "=" * 70)
    print(f"  ❌ 训练失败：{type(error).__name__}: {error}")
    print(f"  ⏱  失败时间：{timestamp}")
    if last_step is not None:
        print(f"  📍 最后成功 step：{last_step}")
    if save_dir:
        print(f"  📂 错误报告目录：{save_dir}")
    print("=" * 70 + "\n")

    if not save_dir:
        return

    try:
        os.makedirs(save_dir, exist_ok=True)

        # 1) JSON 报告
        with open(os.path.join(save_dir, "error_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        # 2) 完整 traceback
        with open(os.path.join(save_dir, "error_traceback.txt"), "w", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {type(error).__name__}: {error}\n\n")
            f.write(tb_text)

        # 3) 人类可读摘要
        md_lines = [
            f"# 训练错误报告\n",
            f"**时间**: {timestamp}",
            f"**错误类型**: `{type(error).__name__}`",
            f"**错误信息**: `{error}`",
        ]
        if last_step is not None:
            md_lines.append(f"**最后成功 step**: {last_step}")
        if gpu_info:
            md_lines.append(f"\n## GPU 状态")
            md_lines.append("| Index | Name | Total | Allocated | Reserved |")
            md_lines.append("|---|---|---|---|---|")
            for g in gpu_info:
                md_lines.append(f"| {g['index']} | {g['name']} | {g['memory_total_gb']} GB | "
                                f"{g['memory_allocated_gb']} GB | {g['memory_reserved_gb']} GB |")
        if extra:
            md_lines.append(f"\n## 附加信息")
            for k, v in extra.items():
                md_lines.append(f"- **{k}**: `{v}`")
        md_lines.append(f"\n## 完整 traceback\n```\n{tb_text}\n```")

        with open(os.path.join(save_dir, "error_summary.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        print(f"  📝 错误报告已保存：")
        print(f"     - {save_dir}/error_report.json")
        print(f"     - {save_dir}/error_traceback.txt")
        print(f"     - {save_dir}/error_summary.md")
    except Exception as write_err:
        print(f"  ⚠️  写错误报告失败：{write_err}")
        print(tb_text)