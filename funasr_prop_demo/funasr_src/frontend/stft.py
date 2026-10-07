import torch
import librosa
import torch.nn as nn
import random
from torch import Tensor
from typing import Optional
from torchaudio.transforms import Spectrogram
import sys
from torchaudio.transforms import Spectrogram, MelScale
from torch.nn.utils.rnn import pad_sequence

def soxnorm(wav: torch.Tensor, gain, factor=None):
    """sox norm, used in Vocos codes;
    """
    wav = wav.float()
    if factor is None:
        linear_gain = 10 ** (gain / 20)
        peak = torch.abs(wav).max().item()
        # Preserve the original non-silent scaling exactly. Silence stays zero
        # with a finite factor that can also be reused by the target transform.
        factor = linear_gain / peak if peak > 0.0 else 1.0
        wav = wav * factor
    else:
        wav = wav * factor
    
    assert torch.all(wav.abs() <= 1), f"out wavform is not in [-1, 1], {wav.abs().max()}, {factor}"
    return wav, factor


class InputSTFT(nn.Module):
    """
    The STFT of the input signal of CleanMel (STFT coefficients);
    In online mode, the recursive normalization is used.
    """
    def __init__(
        self, 
        n_fft: int,
        n_win: int, 
        n_hop: int, 
        center: bool,
        normalize: bool,
        onesided: bool,
        online: bool = False):
        super().__init__()
        
        self.online = online
        self.stft=Spectrogram(
            n_fft=n_fft,
            win_length=n_win,
            hop_length=n_hop,
            normalized=normalize,
            center=center,
            onesided=onesided,
            power=None,
        )
    
    def forward(self, x):
        x, x_norm = soxnorm(x, random.randint(-6, -1) if self.training else -3)
        x = self.stft(x)
        x = torch.view_as_real(x)
        return x , x_norm


class LibrosaMelScale(nn.Module):
    r"""Pytorch implementation of librosa mel scale to align with common ESPNet ASR models; 
    You might need to define .
    """
    def __init__(self, n_mels, sample_rate, f_min, f_max, n_stft, norm=None, mel_scale="slaney"):
        super(LibrosaMelScale, self).__init__()
        
        _mel_options = dict(
            sr=sample_rate,
            n_fft=(n_stft - 1) * 2,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max if f_max is not None else float(sample_rate // 2),
            htk=mel_scale=="htk",
            norm=norm
        )
        
        fb = torch.from_numpy(librosa.filters.mel(**_mel_options).T).float()
        self.register_buffer("fb", fb)
    
    def forward(self, specgram):
        mel_specgram = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)
        return mel_specgram


class TargetMel(nn.Module):
    """
    This class generates the enhancement TARGET mel spectrogram;
    """
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        n_win: int,
        n_hop: int,
        n_mels: int,
        f_min: int,
        f_max: int,
        power: int,
        center: bool,
        normalize: bool,
        onesided: bool,
        mel_norm: str | None,
        mel_scale: str,
        librosa_mel: bool = True,
        online: bool = False,
        ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.online = online
        self.stft = Spectrogram(
            n_fft=n_fft,
            win_length=n_win,
            hop_length=n_hop,
            power=None if online else power,
            normalized=normalize,
            center=center,
            onesided=onesided,
        )
        mel_method = LibrosaMelScale if librosa_mel else MelScale
        self.mel_scale = mel_method(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_stft=n_fft // 2 + 1,
            norm=mel_norm,
            mel_scale=mel_scale,
        )
        
    def forward(self, x: Tensor, x_norm=None):      
        if self.online:
            spectrogram = self.stft(x)
            spectrogram = spectrogram / (x_norm + 1e-8)
            spectrogram = spectrogram.abs().pow(2)
        else:
            x, _ = soxnorm(x, None, x_norm)
            spectrogram = self.stft(x)
        mel_specgram = self.mel_scale(spectrogram)
        return mel_specgram


class InputMel(nn.Module):
    """
    This class generates the enhancement TARGET mel spectrogram;
    """
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        n_win: int,
        n_hop: int,
        n_mels: int,
        f_min: int,
        f_max: int,
        power: int,
        center: bool,
        normalize: bool,
        onesided: bool,
        mel_norm: str | None,
        mel_scale: str,
        librosa_mel: bool = True,
        online: bool = False,
        ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.online = online
        self.stft = Spectrogram(
            n_fft=n_fft,
            win_length=n_win,
            hop_length=n_hop,
            power=None if online else power,
            normalized=normalize,
            center=center,
            onesided=onesided,
        )
        mel_method = LibrosaMelScale if librosa_mel else MelScale
        self.mel_scale = mel_method(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_stft=n_fft // 2 + 1,
            norm=mel_norm,
            mel_scale=mel_scale,
        )
        
    def forward(self, x: Tensor):      
        x, x_norm = soxnorm(x, random.randint(-6, -1) if self.training else -3)
        spectrogram= self.stft(x)
        mel_specgram = self.mel_scale(spectrogram)
        log_mel_specgram = torch.log(torch.clip(mel_specgram, min=1e-10))  
        return log_mel_specgram, x_norm


from frontend.kaldi_spec import kaldi_spectrogram
class FunasrInputSTFT(nn.Module):
    """
    The STFT of the input signal of CleanMel (STFT coefficients);
    In online mode, the recursive normalization is used.
    """
    def __init__(
        self, 
        fs: int = 16000,
        window: str = "hamming",
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
        upsacle_samples: bool = True,
        snip_edges: bool = True,
        n_fft: int = 512,
        ):
        super().__init__()
        self.fs = fs
        self.window = window
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.dither = dither
        self.upsacle_samples = upsacle_samples
        self.snip_edges = snip_edges

    def forward(self, x):
        x, x_norm = soxnorm(x, -3)
        batch_size = x.size(0)
        feats = []
        for i in range(batch_size):
            waveform = x[i]
            if self.upsacle_samples:
                waveform = waveform * (1 << 15)
            waveform = waveform.unsqueeze(0)
            import numpy as np
            output = kaldi_spectrogram(
                waveform,
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
                dither=self.dither,
                energy_floor=0.0,
                window_type=self.window,
                sample_frequency=self.fs,
                snip_edges=self.snip_edges,
            )

            feats.append(output.permute(1, 0, 2))

        if batch_size == 1:
            x = feats[0][None, :, :, :]
        else:
            x = pad_sequence(feats, batch_first=True, padding_value=0.0)
        
        x = x/32768
        return x , x_norm

from typing import Tuple
import torchaudio.compliance.kaldi as kaldi
class WavFrontend(nn.Module):
    """Conventional frontend structure for ASR."""

    def __init__(
        self,
        cmvn_file: str = None,
        fs: int = 16000,
        window: str = "hamming",
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        filter_length_min: int = -1,
        filter_length_max: int = -1,
        lfr_m: int = 1,
        lfr_n: int = 1,
        dither: float = 1.0,
        snip_edges: bool = True,
        upsacle_samples: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.fs = fs
        self.window = window
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.filter_length_min = filter_length_min
        self.filter_length_max = filter_length_max
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.cmvn_file = cmvn_file
        self.dither = dither
        self.snip_edges = snip_edges
        self.upsacle_samples = upsacle_samples


    def output_size(self) -> int:
        return self.n_mels * self.lfr_m

    def forward(
        self,
        input: torch.Tensor,
        input_lengths,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = input[i][:waveform_length]
            if self.upsacle_samples:
                waveform = waveform * (1 << 15)
            waveform = waveform.unsqueeze(0)
            mat = kaldi.fbank(
                waveform,
                num_mel_bins=self.n_mels,
                frame_length=min(self.frame_length,waveform_length/self.fs*1000),
                frame_shift=self.frame_shift,
                dither=self.dither,
                energy_floor=0.0,
                window_type=self.window,
                sample_frequency=self.fs,
                snip_edges=self.snip_edges,
                use_log_fbank=False,
            )
            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        if batch_size == 1:
            feats_pad = feats[0][None, :, :]
        else:
            feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)
        return feats_pad, feats_lens


class FunasrTargetMel(nn.Module):
    """
    This class generates the enhancement TARGET mel spectrogram;
    """
    def __init__(
        self,
        fs: int = 16000,
        window: str = "hamming",
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
        n_fft: int = 512,
        ):
        super().__init__()
        self.n_fft = n_fft
        self.funasr_frontend = WavFrontend(
            fs=fs,
            window=window,
            n_mels=n_mels,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
        )
        
    def forward(self, x: Tensor, x_norm=None):      
        x, _ = soxnorm(x, None, x_norm)
        input_length = torch.tensor([x.size(1)] * x.size(0), dtype=torch.long)
        mel_specgram, _ = self.funasr_frontend(x, input_length)
        mel_specgram = mel_specgram / (32768**2)
        return mel_specgram.permute(0, 2, 1)


