import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, TypeVar, Union, Tuple

T = TypeVar('T')
MaybeList = Union[T, List[T]]

def gather_2d_on_last_dim(tensor: torch.Tensor, index: torch.LongTensor, shape: torch.Size) -> torch.Tensor:
    flattened_tensor = tensor.view(-1, tensor.shape[-1])
    flattened_index = index.view(-1)
    flattened_gathered_tensor = flattened_tensor[torch.arange(flattened_index.shape[0]), flattened_index]
    return flattened_gathered_tensor.view(shape)

def get_masked_mean_min_max(X: torch.Tensor, lengths: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if X.ndim != 2 or lengths.ndim != 1 or X.shape[0] != lengths.shape[0]:
        raise ValueError

    mask = get_lengths_mask(X, lengths)

    masked_min = X.masked_fill(~mask, np.inf).min(dim=1)
    masked_max = X.masked_fill(~mask, -np.inf).max(dim=1)
    masked_mean = mask_and_reduce(sequence=X, sequence_length=lengths, average_across_timesteps=True, sum_over_timesteps=False)

    return (masked_mean, masked_min.values.mean(), masked_max.values.mean())

def masked_reverse_cumsum(X: torch.Tensor, lengths: torch.LongTensor, dim: int) -> torch.Tensor:
    masked_X = X * sequence_mask(lengths, max_len=X.shape[1])

    return (masked_X.flip(dims=[dim]).cumsum(dim=dim).flip(dims=[dim]))

def get_lengths_mask(X: torch.Tensor, lengths: torch.LongTensor) -> torch.Tensor:
    if X.ndim != 2 or lengths.ndim != 1 or X.shape[0] != lengths.shape[0]:
        raise ValueError

    expanded_lengths = lengths.unsqueeze(dim=1)
    return (torch.arange(X.shape[1], device=X.device).expand(X.shape) < expanded_lengths)

def mask_and_reduce(sequence: torch.Tensor, sequence_length: Optional[torch.LongTensor], rank: int = 2, average_across_batch: bool = True, average_across_timesteps: bool = False, average_across_remaining: bool = False, sum_over_batch: bool = False, sum_over_timesteps: bool = True, sum_over_remaining: bool = True, dtype: Optional[torch.dtype] = None, time_major: bool = False) -> torch.Tensor:
    if rank < 2:
        raise ValueError('`rank` must be >= 2.')

    if time_major:
        sequence = transpose_batch_time(sequence)

    if sequence_length is not None:
        sequence = mask_sequences(sequence, sequence_length, dtype=dtype, time_major=False)

    if rank > 2:
        if average_across_remaining and sum_over_remaining:
            raise ValueError("Only one of `average_across_remaining` and `sum_over_remaining` can be set.")
        if average_across_remaining:
            for axis in sorted(list(range(2, rank)), reverse=True):
                sequence = torch.mean(sequence, dim=axis)
        elif sum_over_remaining:
            for axis in sorted(list(range(2, rank)), reverse=True):
                sequence = torch.sum(sequence, dim=axis)

    sequence = reduce_batch_time(sequence, sequence_length, average_across_batch, average_across_timesteps, sum_over_batch, sum_over_timesteps)

    reduce_time = average_across_timesteps or sum_over_timesteps
    reduce_batch = average_across_batch or sum_over_batch

    if not reduce_time and not reduce_batch and time_major:
        sequence = transpose_batch_time(sequence)

    return sequence


def reduce_batch_time(
    sequence: torch.Tensor,
    sequence_length: Optional[torch.LongTensor],
    average_across_batch: bool = True,
    average_across_timesteps: bool = False,
    sum_over_batch: bool = False,
    sum_over_timesteps: bool = True
) -> torch.Tensor:
    if average_across_timesteps and sum_over_timesteps:
        raise ValueError("Only one of `average_across_timesteps` and `sum_over_timesteps` can be set.")
    if average_across_batch and sum_over_batch:
        raise ValueError("Only one of `average_across_batch` and `sum_over_batch` can be set.")

    if sum_over_timesteps:
        sequence = torch.sum(sequence, dim=1)
    elif average_across_timesteps:
        if sequence_length is None:
            sequence = torch.mean(sequence, dim=1)
        else:
            sequence = (torch.sum(sequence, dim=1).float() / sequence_length.float())

    if sum_over_batch:
        sequence = torch.sum(sequence, dim=0)
    elif average_across_batch:
        sequence = torch.mean(sequence, dim=0)

    return sequence

def reduce_dimensions(
    tensor: torch.Tensor,
    average_axes: Optional[MaybeList[int]] = None,
    sum_axes: Optional[MaybeList[int]] = None,
    keepdims: Optional[bool] = None
) -> torch.Tensor:
    reduced_axes = set()
    if average_axes is not None:
        if not isinstance(average_axes, (list, tuple)):
            average_axes = [average_axes]
        if len(average_axes) > 0:
            for average_axis in average_axes:
                tensor = torch.mean(tensor, dim=average_axis, keepdim=True)
            reduced_axes.update(average_axes)

    if sum_axes is not None:
        if not isinstance(sum_axes, (list, tuple)):
            sum_axes = [sum_axes]
        if len(sum_axes) > 0:
            for sum_axis in sum_axes:
                tensor = torch.sum(tensor, dim=sum_axis, keepdim=True)
            reduced_axes.update(sum_axes)

            if average_axes is not None:
                if len(reduced_axes) != len(average_axes) + len(sum_axes):
                    raise ValueError('`average_axes` and `sum_axes` must not have overlapped elements.')
    if not keepdims:
        for axis in sorted(list(reduced_axes), reverse=True):
            tensor = torch.squeeze(tensor, dim=axis)
    return tensor


def sequence_mask(
    lengths: Union[torch.LongTensor, List[int]],
    max_len: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None
) -> torch.ByteTensor:
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, device=device)
    elif device is None:
        device = lengths.device
    lengths: torch.LongTensor
    if max_len is None:
        max_len = torch.max(lengths).item()

    size = lengths.size()
    row_vector = torch.arange(max_len, device=device, dtype=lengths.dtype).view(
        *([1] * len(size)), -1).expand(*size, max_len)
    mask = (row_vector < lengths.unsqueeze(-1)).to(device=device)
    if dtype is not None:
        mask = mask.to(dtype=dtype)

    return mask


def _get_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, -1) + 1e-8
    entropy = - probs * torch.log(probs)
    entropy = torch.sum(entropy, -1)
    return entropy


def transpose_batch_time(inputs: torch.Tensor) -> torch.Tensor:
    return inputs.transpose(0, 1)


def mask_sequences(sequence: Union[torch.Tensor, List[int]],
                   sequence_length: Union[torch.LongTensor, List[int]],
                   dtype: Optional[torch.dtype] = None,
                   time_major: bool = False) -> torch.Tensor:
    if not torch.is_tensor(sequence):
        sequence = torch.tensor(sequence, dtype=dtype)
    sequence: torch.Tensor

    rank = sequence.dim()
    if rank < 2:
        raise ValueError("`sequence` must be 2D or higher order.")

    if time_major:
        sequence = transpose_batch_time(sequence)
    max_time = sequence.size(1)
    if dtype is None:
        dtype = sequence.dtype
    mask = sequence_mask_custom(sequence_length, max_time, dtype=dtype)
    mask = mask.view(*mask.size(), *([1] * (rank - 2)))
    sequence = sequence * mask
    if time_major:
        sequence = transpose_batch_time(sequence)

    return sequence
