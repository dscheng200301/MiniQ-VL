from dataclasses import dataclass
from typing import Union

import torch
from torch import nn
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

from transformers import AutoModelForCausalLM


@dataclass
class QwenVLMConfig:
    model_path: str
    freeze_vision: bool = True
    freeze_language: bool = False
    max_length: int = 2048
    torch_dtype: Union[str, torch.dtype] = "auto"
    trust_remote_code: bool = True


class QwenVLM(nn.Module):
    """Thin wrapper that gives all training stages one stable model interface."""

    def __init__(self, config: QwenVLMConfig):
        super().__init__()
        self.config = config
        self.processor = AutoProcessor.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code,
        )

        load_kwargs = {"trust_remote_code": config.trust_remote_code}
        load_kwargs["torch_dtype"] = config.torch_dtype
        if AutoModelForImageTextToText is not None:
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    config.model_path, **load_kwargs
                )
            except ValueError:
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.model_path, **load_kwargs
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path, **load_kwargs
            )
        self._apply_freeze_policy()

    def _apply_freeze_policy(self):
        for name, param in self.model.named_parameters():
            is_vision = any(part in name.lower() for part in ("visual", "vision", "vit"))
            if self.config.freeze_vision and is_vision:
                param.requires_grad = False
            if self.config.freeze_language and not is_vision:
                param.requires_grad = False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
