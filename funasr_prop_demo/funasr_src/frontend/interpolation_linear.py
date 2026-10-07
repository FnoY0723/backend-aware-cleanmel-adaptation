
"""Linear Projection."""

from typing import Tuple

import torch

from torch import nn
import math
import torch.nn.utils as nn_utils



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

class InterpolationLinear(nn.Module):
    """Linear Projection Preencoder."""

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0, ckpt: str = None):
        """Initialize the module."""
        super().__init__()

        self.output_dim = output_size
        
        self.feedforward = nn.Sequential(
            nn.Linear(input_size, input_size*8),
            nn.GELU(),
            nn.Linear(input_size*8, input_size),
        )
        nn.init.zeros_(self.feedforward[-1].bias)
        nn.init.zeros_(self.feedforward[-1].weight)

        
        self.lfr = BatchedLFR(lfr_m=7, lfr_n=6)
        self.lfr = BatchedLFR(lfr_m=7, lfr_n=6)
        if ckpt is not None:
            state_dict = torch.load(ckpt, map_location=lambda storage, loc: storage, weights_only=True)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('preencoder.'):
                    new_state_dict[k[11:]] = v
            missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
            print(f"===missing_keys in LinearProjection===")
            print(f"{missing_keys}")

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward."""
        input_refined, resampled_lengths = self.forward_nolfr(input, input_lengths)
        output = self.lfr(input_refined)
        out_lengths = torch.ceil(resampled_lengths.float() / self.lfr.n).int()

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

        input_refined = self.feedforward(input_resampled) 
        input_refined = input_refined + input_resampled 
      
        resampled_lengths = (input_lengths.float() * ratio).int()

        return input_refined, resampled_lengths

    def output_size(self) -> int:
        """Get the output size."""
        return self.output_dim
