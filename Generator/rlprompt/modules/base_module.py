import torch
from torch import nn
from typing import Dict, List, Any, Tuple

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_rewards(
        self,
        batch: Dict[str, Any],
        output_tokens: List[List[str]],
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any], Dict[Tuple, Tuple], Dict[Tuple, Tuple]]:
        raise NotImplementedError

    def _pre_steps(self, step: int) -> None:
        pass
