import torch
# import numpy as np
# import sys
# if sys.version_info >= (3, 8):
#     from typing import Union, List, Dict, Any, TypedDict, NamedTuple, Callable
# else:
#     from typing import Union, List, Dict, Any, NamedTuple, Callable
#     from typing_extensions import TypedDict
from typing import Callable
from enum import Enum


class ForwardMode(Enum):
    SQL_ON = "SQL_ON"
    SQL_OFF_GT = "SQL_OFF_GT"
    INFER = "INFER"


def get_reward_shaping_func(
    old_min: float,
    old_max: float,
    new_min: float,
    new_max: float,
    gamma: float
) -> Callable[[torch.Tensor], torch.Tensor]:
    def shaping_func(reward: torch.Tensor, Q_next_max: torch.Tensor, Q_current: torch.Tensor) -> torch.Tensor:
        R = (reward - old_min) / (old_max - old_min)
        return R + gamma * Q_next_max - Q_current

    return shaping_func

