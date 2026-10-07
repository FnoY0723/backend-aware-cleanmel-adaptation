import os
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence
import kaldiio
import humanfriendly
from torchaudio.functional import melscale_fbanks

from espnet2.asr.frontend.abs_frontend import AbsFrontend

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
        assert x.shape[-1] == self.dim, "CMVN dim mismatch"
        out = x - self.means
        out = out * self.inverse_std_variences
        return out

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

class FireRedASR2S_Feat(AbsFrontend):
    """
    """
    def __init__(
        self,
        cmvn_file: str = None,
        fs: int = 16000,
        window: str = "povey",
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        filter_length_min: int = -1,
        filter_length_max: int = -1,
        dither: float = 0.0, 
        snip_edges: bool = True,
        upsacle_samples: bool = True,
        remove_dc_offset: bool = True,
        preemphasis: float = 0.97,
        **kwargs,
    ):
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        self.fs = fs
        self.window_type = window
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.filter_length_min = filter_length_min
        self.filter_length_max = filter_length_max
        self.cmvn_file = cmvn_file
        self.dither = dither
        self.snip_edges = snip_edges
        self.upsacle_samples = upsacle_samples
        self.remove_dc_offset = remove_dc_offset
        self.preemphasis = preemphasis

        self.cmvn = CMVN(self.cmvn_file) if self.cmvn_file is not None else None

        self.n_win = int(self.fs * (self.frame_length / 1000.0))
        self.n_hop = int(self.fs * (self.frame_shift / 1000.0))
        self.n_fft = 2 ** math.ceil(math.log2(self.n_win))

        if self.window_type == "povey":
            window_tensor = torch.hann_window(self.n_win, periodic=False).pow(0.85)
        elif self.window_type == "hamming":
            window_tensor = torch.hamming_window(self.n_win, periodic=False, alpha=0.54, beta=0.46)
        else:
            window_tensor = torch.ones(self.n_win)
        self.register_buffer("window", window_tensor)

        mel_basis = melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=20.0,
            f_max=fs / 2,
            n_mels=n_mels,
            sample_rate=fs,
            mel_scale="htk"
        )
        self.register_buffer("mel_basis", mel_basis)

    def output_size(self) -> int:
        return self.n_mels

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input.size(0)
        feats = []
        feats_lens = []

        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = input[i][:waveform_length]

            if self.upsacle_samples:
                waveform = waveform * 32768.0

            if self.snip_edges:
                frames = waveform.unfold(-1, self.n_win, self.n_hop).contiguous()
            else:
                raise NotImplementedError("snip_edges=False is not fully implemented.")

            if self.dither != 0.0:
                noise = torch.randn_like(frames) * self.dither
                frames = frames + noise

            if self.remove_dc_offset:
                frames = frames - frames.mean(dim=-1, keepdim=True)

            if self.preemphasis != 0.0:
                first_sample = frames[..., 0:1] * (1.0 - self.preemphasis)
                rest_samples = frames[..., 1:] - self.preemphasis * frames[..., :-1]
                frames = torch.cat([first_sample, rest_samples], dim=-1)

            frames = frames * self.window

            complex_stft = torch.fft.rfft(frames, n=self.n_fft)
            power_spec = complex_stft.real.pow(2) + complex_stft.imag.pow(2)
            mel_spec = torch.matmul(power_spec, self.mel_basis)
            log_mel_fbank = torch.log(mel_spec.clamp(min=1e-10))
            
            mat = log_mel_fbank

            if self.cmvn is not None:
                mat = self.cmvn(mat)

            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens, device=input.device)
        if batch_size == 1:
            feats_pad = feats[0][None, :, :]
        else:
            feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)

        return feats_pad, feats_lens
