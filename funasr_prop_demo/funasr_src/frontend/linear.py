
"""Prop. residual Adapter (the paper's ``Adapter'').

This file implements the trainable residual Adapter between the CleanMel
enhancement frontend and the frozen Fun-ASR-Nano encoder. CleanMel and this
Adapter are both updated during backend adaptation; the backend remains frozen.

Pipeline for the ``prop'' variant:

    raw waveform
        -> CleanMel  (produces an 80-dim enhanced log-mel, 16 ms frame shift)
        -> LinearProjection (THIS FILE)          <-- the Adapter
             1. resample the time axis  16  ms -> 10 ms  (ratio 1.6),
                so the feature rate matches what the frozen encoder expects;
             2. a small 4-layer 2-D Conv stack predicts a *residual* correction,
                added back onto the resampled feature (zero-initialised, so the
                Adapter starts as an identity map and only learns a correction);
             3. LFR (low-frame-rate stacking, m=7 / n=6) to match the encoder's
                stacked-frame input.
        -> frozen Fun-ASR-Nano encoder -> Adaptor -> Qwen3 LLM decoder

Only ``conv2d`` and ``out`` carry learned parameters; both the interpolation and
the LFR stacking are parameter-free. Because ``out`` is zero-initialised, at
init the module output equals the (interpolated) CleanMel feature exactly, and
training only has to learn the *residual* backend-aware correction.
"""

from typing import Tuple

import torch
from torch import nn
import math
import torch.nn.utils as nn_utils


class BatchedLFR(nn.Module):
    """Low-Frame-Rate (LFR) stacking, applied per batch.

    Stacks ``m`` consecutive frames with stride ``n`` (m=7, n=6 here), i.e. each
    output frame is the concatenation of a 7-frame window and windows hop by 6
    frames. This reproduces the frozen encoder's expected stacked-frame input.
    The module has no learnable parameters.
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


class LinearProjection(nn.Module):
    """The Prop. residual Adapter (paper's ``Adapter'').

    Maps the 80-dim CleanMel feature (16 ms shift) to the frozen encoder's input:
    resample to 10 ms, add a zero-initialised residual Conv2D correction, then LFR
    stack. This is the module referenced as ``self.linear_proj`` in
    ``model_cleanmel.py``. Both this Adapter and CleanMel are updated during
    backend adaptation; the ASR backend remains frozen.
    """

    RESAMPLE_RATIO = 1.6

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0, ckpt: str = None):
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

        if ckpt is not None:
            state_dict = torch.load(
                ckpt, map_location=lambda storage, loc: storage, weights_only=True
            )
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            if not isinstance(state_dict, dict):
                raise RuntimeError(f"Adapter checkpoint is not a state dictionary: {ckpt}")
            new_state_dict = {
                k[len("preencoder."):]: v
                for k, v in state_dict.items()
                if k.startswith("preencoder.")
            }
            try:
                self.load_state_dict(new_state_dict, strict=True)
            except RuntimeError as error:
                raise RuntimeError(
                    f"Cannot load complete Adapter weights from {ckpt}: {error}"
                ) from error

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
