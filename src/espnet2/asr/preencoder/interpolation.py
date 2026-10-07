
"""Temporal interpolation without a learnable residual Adapter."""

import math
from typing import Tuple

import torch
from torch import nn
from typeguard import typechecked

from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder


class BatchedLFR(nn.Module):
    """Apply the backend's 7-frame/6-shift low-frame-rate stacking."""

    def __init__(self, lfr_m: int = 7, lfr_n: int = 6):
        super().__init__()
        self.m = lfr_m
        self.n = lfr_n

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, feat_dim = inputs.shape
        output_frames = math.ceil(num_frames / self.n)
        left_pad_size = (self.m - 1) // 2
        left_pad = inputs[:, :1, :].repeat(1, left_pad_size, 1)
        target_frames = (output_frames - 1) * self.n + self.m
        right_pad_size = max(0, target_frames - num_frames - left_pad_size)
        if right_pad_size:
            right_pad = inputs[:, -1:, :].repeat(1, right_pad_size, 1)
            inputs = torch.cat((left_pad, inputs, right_pad), dim=1)
        else:
            inputs = torch.cat((left_pad, inputs), dim=1)
        windows = inputs.unfold(1, self.m, self.n)
        return windows.transpose(2, 3).contiguous().view(
            batch_size, output_frames, self.m * feat_dim
        )


class Interpolation(AbsPreEncoder):
    """Match the backend frame rate using interpolation only."""

    @typechecked
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0):
        super().__init__()
        self.output_dim = output_size
        self.lfr = BatchedLFR(lfr_m=7, lfr_n=6)

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        interpolated, interpolated_lengths = self.forward_nolfr(
            inputs, input_lengths
        )
        outputs = self.lfr(interpolated)
        output_lengths = torch.ceil(
            interpolated_lengths.float() / self.lfr.n
        ).int()
        return outputs, output_lengths

    def forward_nolfr(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_frames = int(inputs.shape[1] * 1.6)
        interpolated = torch.nn.functional.interpolate(
            inputs.transpose(1, 2),
            size=target_frames,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        interpolated_lengths = (input_lengths.float() * 1.6).int()
        return interpolated, interpolated_lengths

    def output_size(self) -> int:
        return self.output_dim
