"""Differentiable Kaldi-compatible complex spectrogram extraction."""

import torch
from torch import Tensor
import torchaudio.compliance.kaldi as kaldi

POVEY = "povey"


def kaldi_spectrogram(
    waveform: Tensor,
    blackman_coeff: float = 0.42,
    channel: int = -1,
    dither: float = 0.0,
    energy_floor: float = 1.0,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    min_duration: float = 0.0,
    preemphasis_coefficient: float = 0.97,
    raw_energy: bool = True,
    remove_dc_offset: bool = True,
    round_to_power_of_two: bool = True,
    sample_frequency: float = 16000.0,
    snip_edges: bool = True,
    window_type: str = POVEY,
) -> Tensor:
    """Return Kaldi-windowed complex spectra as real/imaginary pairs."""
    device, dtype = waveform.device, waveform.dtype
    waveform, window_shift, window_size, padded_window_size = (
        kaldi._get_waveform_and_window_properties(
            waveform,
            channel,
            sample_frequency,
            frame_shift,
            frame_length,
            round_to_power_of_two,
            preemphasis_coefficient,
        )
    )
    if len(waveform) < min_duration * sample_frequency:
        return torch.empty(0, device=device, dtype=dtype)
    strided_input, _ = kaldi._get_window(
        waveform,
        padded_window_size,
        window_size,
        window_shift,
        window_type,
        blackman_coeff,
        snip_edges,
        raw_energy,
        energy_floor,
        dither,
        remove_dc_offset,
        preemphasis_coefficient,
    )
    return torch.view_as_real(torch.fft.rfft(strided_input))
