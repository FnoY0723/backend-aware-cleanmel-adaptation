"""Feature extraction used by the released CleanMel frontend."""

from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
import torchaudio.compliance.kaldi as kaldi

from espnet2.asr.frontend.kaldi_spec import kaldi_spectrogram


def soxnorm(
    wav: torch.Tensor, gain: torch.Tensor | float, factor: torch.Tensor | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply per-utterance dBFS normalization and return its scale factor."""
    wav = wav.float()
    squeezed = wav.ndim == 1
    if squeezed:
        wav = wav.unsqueeze(0)
    batch_size = wav.size(0)
    if factor is None:
        gain = torch.as_tensor(gain, dtype=wav.dtype, device=wav.device)
        gain = gain.expand(batch_size) if gain.ndim == 0 else gain.reshape(-1)
        if gain.numel() != batch_size:
            raise ValueError("Gain count must match batch size")
        linear_gain = torch.pow(10.0, gain / 20.0).view(batch_size, 1)
        factor = linear_gain / torch.abs(wav).amax(dim=1, keepdim=True).clamp_min(1e-10)
    else:
        factor = torch.as_tensor(factor, dtype=wav.dtype, device=wav.device)
        if factor.ndim == 0:
            factor = factor.expand(batch_size, 1)
        elif factor.ndim == 1:
            factor = factor.unsqueeze(1)
        if factor.shape[0] != batch_size:
            raise ValueError("Normalization-factor count must match batch size")
    wav = wav * factor
    if squeezed:
        return wav.squeeze(0), factor.squeeze(0)
    return wav, factor


class FunasrInputSTFT(nn.Module):
    """Produce normalized complex STFT input for CleanMel."""

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
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.dither = dither
        self.upsacle_samples = upsacle_samples
        self.snip_edges = snip_edges

    def forward(self, x: Tensor, x_lengths=None):
        gains = torch.full((x.size(0),), -3.0, device=x.device)
        x, x_norm = soxnorm(x, gains)
        feats = []
        for waveform in x:
            if self.upsacle_samples:
                waveform = waveform * (1 << 15)
            spectrum = kaldi_spectrogram(
                waveform.unsqueeze(0),
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
                dither=self.dither,
                energy_floor=0.0,
                window_type=self.window,
                sample_frequency=self.fs,
                snip_edges=self.snip_edges,
            )
            feats.append(spectrum.permute(1, 0, 2))
        output = feats[0].unsqueeze(0) if len(feats) == 1 else pad_sequence(
            feats, batch_first=True, padding_value=0.0
        )
        return output / 32768, x_norm


class WavFrontend(nn.Module):
    """Kaldi fbank used to construct the CleanMel target Mel."""

    def __init__(
        self,
        fs: int = 16000,
        window: str = "hamming",
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
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
        self.dither = dither
        self.snip_edges = snip_edges
        self.upsacle_samples = upsacle_samples

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        feats, lengths = [], []
        for waveform, waveform_length in zip(inputs, input_lengths):
            waveform = waveform[:waveform_length]
            if self.upsacle_samples:
                waveform = waveform * (1 << 15)
            feat = kaldi.fbank(
                waveform.unsqueeze(0),
                num_mel_bins=self.n_mels,
                frame_length=min(self.frame_length, waveform_length / self.fs * 1000),
                frame_shift=self.frame_shift,
                dither=self.dither,
                energy_floor=0.0,
                window_type=self.window,
                sample_frequency=self.fs,
                snip_edges=self.snip_edges,
                use_log_fbank=False,
            )
            feats.append(feat)
            lengths.append(feat.size(0))
        output = feats[0].unsqueeze(0) if len(feats) == 1 else pad_sequence(
            feats, batch_first=True, padding_value=0.0
        )
        return output, torch.as_tensor(lengths)


class FunasrTargetMel(nn.Module):
    """Produce the normalized Mel target used during CleanMel training."""

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
        self.funasr_frontend = WavFrontend(
            fs=fs,
            window=window,
            n_mels=n_mels,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
        )

    def forward(self, x: Tensor, x_norm=None):
        x, _ = soxnorm(x, 0.0, x_norm)
        input_lengths = torch.full(
            (x.size(0),), x.size(1), dtype=torch.long, device=x.device
        )
        mel, _ = self.funasr_frontend(x, input_lengths)
        return (mel / (32768**2)).permute(0, 2, 1)
