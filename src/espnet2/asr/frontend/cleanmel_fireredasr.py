from typing import *
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, init
from torch.nn.common_types import _size_1_t

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.stft import FunasrInputSTFT, FunasrTargetMel
from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask
from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams


class LinearGroup(nn.Module):

    def __init__(self, in_features: int, out_features: int, num_groups: int, bias: bool = True) -> None:
        super(LinearGroup, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.weight = Parameter(torch.empty((num_groups, out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """shape [..., group, feature]"""
        x = torch.einsum("...gh,gkh->...gk", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, num_groups={self.num_groups}, bias={True if self.bias is not None else False}"

class LayerNorm(nn.LayerNorm):

    def __init__(self, seq_last: bool, **kwargs) -> None:
        """
        Arg s:
            seq_last (bool): whether the sequence dim is the last dim
        """
        super().__init__(**kwargs)
        self.seq_last = seq_last

    def forward(self, input: Tensor) -> Tensor:
        if self.seq_last:
            input = input.transpose(-1, 1)
        o = super().forward(input)
        if self.seq_last:
            o = o.transpose(-1, 1)
        return o

class CausalConv1d(nn.Conv1d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t | str = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        look_ahead: int = 0,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.look_ahead = look_ahead
        assert look_ahead <= self.kernel_size[0] - 1, (look_ahead, self.kernel_size)

    def forward(self, x: Tensor, state: Dict[int, Any] = None) -> Tensor:
        B, H, T = x.shape
        if state is None or id(self) not in state:
            x = F.pad(x, pad=(self.kernel_size[0] - 1 - self.look_ahead, self.look_ahead))
        else:
            x = torch.concat([state[id(self)], x], dim=-1)
        if state is not None:
            state[id(self)] = x[..., -self.kernel_size + 1:]
        x = super().forward(x)
        return x

class CleanMelLayer(nn.Module):

    def __init__(
            self,
            dim_hidden: int,
            dim_squeeze: int,
            n_freqs: int,
            dropout: Tuple[float, float, float] = (0, 0, 0),
            f_kernel_size: int = 5,
            f_conv_groups: int = 8,
            padding: str = 'zeros',
            full: nn.Module = None,
            mamba_state: int = None,
            mamba_conv_kernel: int = None,
            online: bool = False,
    ) -> None:
        super().__init__()
        self.online = online
        self.n_freqs = n_freqs
        self.fconv1 = nn.ModuleList([
            LayerNorm(seq_last=True, normalized_shape=dim_hidden),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])
        self.norm_full = LayerNorm(seq_last=False, normalized_shape=dim_hidden)
        self.full_share = False if full == None else True
        self.squeeze = nn.Sequential(nn.Conv1d(in_channels=dim_hidden, out_channels=dim_squeeze, kernel_size=1), nn.SiLU())
        self.dropout_full = nn.Dropout2d(dropout[2]) if dropout[2] > 0 else None
        self.full = LinearGroup(n_freqs, n_freqs, num_groups=dim_squeeze) if full == None else full
        self.unsqueeze = nn.Sequential(nn.Conv1d(in_channels=dim_squeeze, out_channels=dim_hidden, kernel_size=1), nn.SiLU())
        self.fconv2 = nn.ModuleList([
            LayerNorm(seq_last=True, normalized_shape=dim_hidden),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])

        self.norm_mamba = LayerNorm(seq_last=False, normalized_shape=dim_hidden)
        if online:
            self.mamba = Mamba(d_model=dim_hidden, d_state=mamba_state, d_conv=mamba_conv_kernel, layer_idx=0)
        else:
            self.mamba = nn.ModuleList([
                Mamba(d_model=dim_hidden, d_state=mamba_state, d_conv=mamba_conv_kernel, layer_idx=0),
                Mamba(d_model=dim_hidden, d_state=mamba_state, d_conv=mamba_conv_kernel, layer_idx=1),
            ])
        self.dropout_mamba = nn.Dropout(dropout[0])

    
    def forward(self, x: Tensor, inference: bool = False) -> Tensor:
        x = x + self._fconv(self.fconv1, x)
        x = x + self._full(x)
        x = x + self._fconv(self.fconv2, x)  

        if self.online:
            x = x + self._mamba(x, self.mamba, self.norm_mamba, self.dropout_mamba, inference)
        else:
            x_fw = self._mamba(x, self.mamba[0], self.norm_mamba, self.dropout_mamba, inference)
            x_bw = self._mamba(x.flip(dims=[2]), self.mamba[1], self.norm_mamba, self.dropout_mamba, inference)
            x = x + (x_fw + x_bw.flip(dims=[2])) / 2 

        return x

    def _mamba(self, x: Tensor, mamba: Mamba, norm: nn.Module, dropout: nn.Module, inference: bool = False):
        B, F, T, H = x.shape
        x = norm(x)
        x = x.reshape(B * F, T, H)
        if inference:
            inference_params = InferenceParams(T, B * F)
            xs = []
            for i in range(T):
                inference_params.seqlen_offset = i
                xi = mamba.forward(x[:, [i], :], inference_params)
                xs.append(xi)
            x = torch.concat(xs, dim=1)
        else:
            x = mamba.forward(x)
        x = x.reshape(B, F, T, H)
        return dropout(x)

    def _fconv(self, ml: nn.ModuleList, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B * T, H, F)
        for m in ml:
            x = m(x)
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)
        return x

    def _full(self, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = self.norm_full(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B * T, H, F)
        x = self.squeeze(x)
        if self.dropout_full:
            x = x.reshape(B, T, -1, F)
            x = x.transpose(1, 3)
            x = self.dropout_full(x)
            x = x.transpose(1, 3)
            x = x.reshape(B * T, -1, F)
        x = self.full(x)
        x = self.unsqueeze(x)
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)
        return x

    def extra_repr(self) -> str:
        return f"full_share={self.full_share}"



class CleanMel(nn.Module):

    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        n_layers: int,
        n_freqs: int,
        n_mels: int = 80,
        layer_linear_freq: int = 1,
        encoder_kernel_size: int = 5,
        dim_hidden: int = 192,
        dropout: Tuple[float, float, float] = (0, 0, 0),
        f_kernel_size: int = 5,
        f_conv_groups: int = 8,
        padding: str = 'zeros',
        mamba_state: int = 16,
        mamba_conv_kernel: int = 4,
        online: bool = True,
        sr: int = 16000,
        n_fft: int = 512,
    ):
        super().__init__()
        self.layer_linear_freq = layer_linear_freq
        self.online = online
        self.encoder = nn.Conv1d(in_channels=dim_input, out_channels=dim_hidden, kernel_size=encoder_kernel_size, padding='same', padding_mode=padding)
        full = None
        layers = []
        for l in range(n_layers):
            layer = CleanMelLayer(
                dim_hidden=dim_hidden,
                dim_squeeze=8 if l < layer_linear_freq else dim_hidden,
                n_freqs=n_freqs if l < layer_linear_freq else n_mels,
                dropout=dropout,
                f_kernel_size=f_kernel_size,
                f_conv_groups=f_conv_groups,
                padding=padding,
                full=full if l > layer_linear_freq else None,
                online=online,
                mamba_conv_kernel=mamba_conv_kernel,
                mamba_state=mamba_state,    
            )
            if hasattr(layer, 'full'):
                full = layer.full
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        import torchaudio
        linear2mel, _ = torchaudio.compliance.kaldi.get_mel_banks(
            80, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0
        )
        linear2mel = torch.nn.functional.pad(linear2mel, (0, 1), mode="constant", value=0)
        denom = linear2mel.sum(dim=1, keepdim=True)
        linear2mel = linear2mel / (denom + 1e-10)
        self.register_buffer("linear2mel", linear2mel.T.clone().detach().to(torch.float32))
        self.norm_after_linear2mel = LayerNorm(seq_last=False, normalized_shape=dim_hidden)

        self.after_norm = LayerNorm(seq_last=False, normalized_shape=dim_hidden)
        self.decoder = nn.Linear(in_features=dim_hidden, out_features=1)


    def forward(self, x: Tensor, inference: bool = False) -> Tensor:
        B, F, T, H0 = x.shape
        x = self.encoder(x.reshape(B * F, T, H0).permute(0, 2, 1)).permute(0, 2, 1)
        
        H = x.shape[2]
        x = x.reshape(B, F, T, H)
        
        for i in range(self.layer_linear_freq):
            m = self.layers[i]
            x = m(x, inference).contiguous()
        
        x = torch.einsum("bfth,fm->bmth", x, self.linear2mel)
        x = self.norm_after_linear2mel(x)
        
        for i in range(self.layer_linear_freq, len(self.layers)):
            m = self.layers[i]
            x = m(x, inference).contiguous()
        
        B, F, T, H = x.shape
        x = self.after_norm(x)
        y = self.decoder(x).squeeze(-1) 
   
        return y.contiguous()


class CleanMelFrontend(AbsFrontend):
    def __init__(self, model_path: str = None, depth: int = 16, hidden: int = 120, fs: int = 16000, frame_length: int = 32, frame_shift: int = 16, **kwargs) -> None:
        super().__init__()
        self.stft = FunasrInputSTFT(
            frame_length= frame_length,
            frame_shift= frame_shift,
        )

        self.target_mel = FunasrTargetMel(
            frame_length= frame_length,
            frame_shift= frame_shift,
        )
    
        self.cleanmel = CleanMel(
            dim_input=2,
            dim_output=1,
            n_layers=depth,
            dim_hidden=hidden,
            layer_linear_freq=1,
            f_kernel_size=5,
            f_conv_groups=8,
            n_freqs=257,
            mamba_state=16,
            mamba_conv_kernel=4,
            online=False,
            sr=16000,
            n_fft=512
        )
        
        if model_path is not None:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            if not isinstance(state_dict, dict):
                raise RuntimeError(f"CleanMel checkpoint is not a state dictionary: {model_path}")
            # Composite adaptation and pretraining checkpoints also contain
            # other modules. Select CleanMel, then require its complete state.
            prefix = next(
                (p for p in ("frontend.cleanmel.", "arch.")
                 if any(k.startswith(p) for k in state_dict)),
                "",
            )
            new_state_dict = {
                k[len(prefix):]: v for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            try:
                self.cleanmel.load_state_dict(new_state_dict, strict=True)
            except RuntimeError as error:
                raise RuntimeError(
                    f"Cannot load complete CleanMel weights from {model_path}: {error}"
                ) from error

        
    def output_size(self) -> int:
        return 80
    
    def safe_log(self, x: Tensor) -> Tensor:
        return torch.log(torch.clip(x, min=1e-5))
     
    
    def forward(self, wav: Tensor, wav_length: Tensor) -> Tensor:
        stft, x_norm = self.stft(wav)
        noisy_mel = self.target_mel(wav, x_norm)
        mask = torch.sigmoid(self.cleanmel(stft))
        enhanced_mel = torch.square(mask * torch.sqrt(noisy_mel))
        output_lengths = torch.div(wav_length - 512, 256, rounding_mode="trunc") + 1
        enhanced_mel.masked_fill_(
            make_pad_mask(output_lengths, enhanced_mel, 2), 0.0
        )
        floor = torch.tensor(1e6, device=enhanced_mel.device, dtype=enhanced_mel.dtype)
        enhanced_mel = torch.max(enhanced_mel * (1 << 30), floor).log()
        return enhanced_mel.transpose(1, 2), output_lengths
