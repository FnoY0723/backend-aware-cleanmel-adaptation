
"""Linear Projection."""

from typing import Tuple

import os
import torch
import kaldiio
from typeguard import typechecked
from torch import nn
import math
import torch.nn.utils as nn_utils

from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder

class BatchedLFR(nn.Module):
    def __init__(self, lfr_m=7, lfr_n=6):
        super().__init__()
        self.m = lfr_m
        self.n = lfr_n

    def forward(self, inputs):
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

import numpy as np
class CMVN(nn.Module):
    """
    CMVN via kaldiio: read the .mat stats and apply them as tensor ops.
    """
    def __init__(self, kaldi_cmvn_file):
        super().__init__()
        dim, means, inverse_std_variences = self.read_kaldi_cmvn(kaldi_cmvn_file)
        self.dim = dim
        self.register_buffer("means", torch.from_numpy(means).float())
        self.register_buffer("inverse_std_variences", torch.from_numpy(inverse_std_variences).float())

    def forward(self, x):
        if x.dim() not in (2, 3):
            raise ValueError(f"CMVN expects 2D/3D input, but got shape={tuple(x.shape)}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"CMVN dim mismatch: expected {self.dim}, got {x.shape[-1]}")

        means = self.means.to(dtype=x.dtype)
        inv_stds = self.inverse_std_variences.to(dtype=x.dtype)
        return (x - means) * inv_stds

    def read_kaldi_cmvn(self, kaldi_cmvn_file):
        assert os.path.exists(kaldi_cmvn_file)
        stats = kaldiio.load_mat(kaldi_cmvn_file)
        assert stats.shape[0] == 2
        dim = stats.shape[-1] - 1
        count = stats[0, dim]
        assert count >= 1
        floor = 1e-20
        means = []
        inverse_std_variences = []
        for d in range(dim):
            mean = stats[0, d] / count
            means.append(mean.item())
            varience = (stats[1, d] / count) - mean * mean
            if varience < floor:
                varience = floor
            istd = 1.0 / math.sqrt(varience)
            inverse_std_variences.append(istd)
        return dim, np.array(means), np.array(inverse_std_variences)


class LinearProjection(AbsPreEncoder):
    """Linear Projection Preencoder."""

    @typechecked
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0):
        """Initialize the module."""
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





        
        model_path = os.environ.get("FIRERED_ADAPTER_INIT")
        if model_path:
            state_dict = torch.load(
                model_path,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )
            new_state_dict = {
                k[11:]: v for k, v in state_dict.items() if k.startswith("preencoder.")
            }
            self.load_state_dict(new_state_dict, strict=False)


    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward."""
        input_refined, resampled_lengths = self.forward_nolfr(input, input_lengths)
        output = input_refined
        out_lengths = resampled_lengths

        return output, out_lengths
    
    def forward_nolfr(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward."""
        B, T, C = input.shape
        ratio = 1.6
        target_T = int(T * ratio)
        

        input_resampled = torch.nn.functional.interpolate(
            input.transpose(1, 2), 
            size=target_T, 
            mode='linear', 
            align_corners=False
        ).transpose(1, 2)

        input_refined = self.conv2d(input_resampled.unsqueeze(1))
        input_refined = input_refined.transpose(1, 2).contiguous().view(B, -1, 256*80)
        input_refined = self.out(input_refined)
        input_refined = input_refined + input_resampled 
      
        resampled_lengths = (input_lengths.float() * ratio).int()

        return input_refined, resampled_lengths

    def output_size(self) -> int:
        """Get the output size."""
        return self.output_dim
