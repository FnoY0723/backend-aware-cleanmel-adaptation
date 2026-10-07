
"""Prop. residual Adapter (the paper's ``Adapter'') -- ESPnet pre-encoder.

This is the trainable residual Adapter between the CleanMel enhancement
frontend and the frozen ASR encoder. CleanMel and this Adapter are both
updated during backend adaptation; the backend remains frozen.
It is registered as an ESPnet ``preencoder`` (hence ``preencoder.*`` tensors in
the released checkpoint) and is the concrete realisation of the paper's
"residual Adapter".

What it does (forward pass):

    CleanMel 80-dim log-mel feature (16 ms shift)
        1. resample the time axis 16 ms -> 10 ms (ratio 1.6) to match the frozen
           encoder's feature rate;
        2. a small 4-layer 2-D Conv stack predicts a *residual* correction that
           is added back onto the resampled feature. The final projection is
           zero-initialised, so at init the Adapter is an identity map and only
           the backend-aware residual correction is learned;
        3. LFR (low-frame-rate stacking, m=7 / n=6) to match the encoder's
           stacked-frame input.

Only ``conv2d`` and ``out`` carry learned parameters; interpolation and LFR are
parameter-free. Ablation variants of this module live in the sibling files
``interpolation.py`` (no residual), ``interpolation_linear.py`` (feed-forward
residual) and ``linear_noresidule.py`` (non-residual conv).
"""

from typing import Tuple

import torch
from typeguard import typechecked
from torch import nn
import math
import torch.nn.utils as nn_utils

from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder


class BatchedLFR(nn.Module):
    """Low-Frame-Rate (LFR) stacking, applied per batch.

    Stacks ``m`` consecutive frames with stride ``n`` (m=7, n=6 here), i.e. each
    output frame is the concatenation of a 7-frame window and windows hop by 6
    frames. Reproduces the frozen encoder's stacked-frame input. No parameters.
    """

    def __init__(self, lfr_m: int = 7, lfr_n: int = 6):
        super().__init__()
        self.m = lfr_m
        self.n = lfr_n

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B, T, D = inputs.shape
        T_lfr = math.ceil(T / self.n)

        left_pad_len = (self.m - 1) // 2
        left_pad = inputs[:, 0:1, :].repeat(1, left_pad_len, 1)

        padded_T = T + left_pad_len
        target_T = (T_lfr - 1) * self.n + self.m
        right_pad_len = max(0, target_T - padded_T)

        if right_pad_len > 0:
            right_pad = inputs[:, -1:, :].repeat(1, right_pad_len, 1)
            x_padded = torch.cat([left_pad, inputs, right_pad], dim=1)
        else:
            x_padded = torch.cat([left_pad, inputs], dim=1)

        windows = x_padded.unfold(1, self.m, self.n)

        out = windows.transpose(2, 3).contiguous().view(B, T_lfr, self.m * D)
        return out


class LinearProjection(AbsPreEncoder):
    """The Prop. residual Adapter (paper's ``Adapter``), as an ESPnet pre-encoder."""

    RESAMPLE_RATIO = 1.6

    @typechecked
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0):
        super().__init__()

        self.output_dim = output_size

        self.conv2d = nn.Sequential(
            nn_utils.weight_norm(nn.Conv2d(1, 256, kernel_size=3, padding=1)),
            nn.GELU(),
            nn_utils.weight_norm(nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            nn.GELU(),
            nn_utils.weight_norm(nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            nn.GELU(),
            nn_utils.weight_norm(nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            nn.GELU(),
        )
        self.out = nn.Linear(256 * 80, 80)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        self.lfr = BatchedLFR(lfr_m=7, lfr_n=6)

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Residual correction + LFR stacking. Returns (feature, lengths)."""
        input_refined, resampled_lengths = self.forward_nolfr(input, input_lengths)
        output = self.lfr(input_refined)
        out_lengths = torch.ceil(resampled_lengths.float() / self.lfr.n).int()
        return output, out_lengths

    def forward_nolfr(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The core Adapter step (before LFR): resample + residual correction.

        input: [B, T, 80] CleanMel feature at 16 ms shift.
        """
        B, T, C = input.shape
        target_T = int(T * self.RESAMPLE_RATIO)

        input_resampled = torch.nn.functional.interpolate(
            input.transpose(1, 2), size=target_T, mode="linear", align_corners=False
        ).transpose(1, 2)

        refined = self.conv2d(input_resampled.unsqueeze(1))
        refined = refined.transpose(1, 2).contiguous().view(B, -1, 256 * 80)
        refined = self.out(refined)
        input_refined = refined + input_resampled

        resampled_lengths = (input_lengths.float() * self.RESAMPLE_RATIO).int()
        return input_refined, resampled_lengths

    def output_size(self) -> int:
        return self.output_dim
