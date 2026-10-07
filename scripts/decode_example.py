"""Diagnostic greedy-CTC decoding with the archived FunASR adaptation overlay.

This auxiliary-head diagnostic is not the original backend decoder used for
reported manuscript recognition results. It does not reproduce the main-table
scores. Use funasr_prop_demo/run_demo.py for the archived FunASR LLM inference
integration. Neither script provides the full evaluation/scoring pipeline.

Requires a compatible ESPnet base installation, the frozen backend checkpoint,
the released CleanMel--Adapter weights, and a user-supplied waveform. CleanMel
forward inference uses CUDA Mamba kernels in the recorded environment.

Example::

    python scripts/decode_example.py \
        --backend /path/to/Fun-ASR-Nano-2512/model.pt \
        --wav /path/to/your/utterance.wav --device cuda
"""

import argparse
import os
import tempfile
from pathlib import Path

import torch
import yaml

from espnet2.tasks.asr_funasrnano import ASRTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=Path, required=True,
                        help="frozen official FunASR-nano model.pt")
    parser.add_argument("--wav", type=Path, required=True,
                        help="16 kHz mono waveform to decode")
    parser.add_argument("--adapted", type=Path,
                        default=Path("checkpoints/funasr_prop_frontend.pth"))
    parser.add_argument("--config", type=Path,
                        default=Path("configs/expanded/funasr_prop.yaml"))
    parser.add_argument("--no-adapt", action="store_true",
                        help="do NOT load the released frontend (contrast run)")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_waveform(path: Path) -> torch.Tensor:
    """Return a (num_samples,) float32 tensor resampled to 16 kHz mono."""
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio).mean(dim=1)
    if sample_rate != 16000:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return waveform


def build_model(args: argparse.Namespace):
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config.pop("config", None)
    config["frontend_conf"]["model_path"] = None
    config["output_dir"] = str(Path(tempfile.gettempdir()) / "funasr_decode_example")
    os.environ["FUNASR_MODEL_PATH"] = str(args.backend.resolve())
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
        temporary_config = Path(stream.name)
    try:
        model, _ = ASRTask.build_model_from_file(temporary_config, None, "cpu")
    finally:
        temporary_config.unlink(missing_ok=True)

    if not args.no_adapt:
        adapted = torch.load(args.adapted, map_location="cpu", weights_only=True)
        if not adapted or any(
            not key.startswith(("frontend.", "preencoder.")) for key in adapted
        ):
            raise RuntimeError("Adapted checkpoint contains unexpected modules")
        missing, unexpected = model.load_state_dict(adapted, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    return model


def ctc_greedy_decode(token_ids, blank_id, token_list) -> str:
    """Collapse repeats, drop blanks, and detokenize BPE/char tokens."""
    tokens = []
    previous = None
    for index in token_ids:
        if index != previous and index != blank_id:
            tokens.append(token_list[index])
        previous = index
    text = "".join(tokens).replace("\u2581", " ").strip()
    return text


def main() -> None:
    args = parse_args()
    if not args.backend.is_file():
        raise FileNotFoundError(args.backend)
    if not args.wav.is_file():
        raise FileNotFoundError(args.wav)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Mamba CleanMel frontend")

    model = build_model(args).eval().to(args.device)
    waveform = load_waveform(args.wav).to(args.device)
    lengths = torch.tensor([waveform.numel()], device=args.device)

    with torch.inference_mode():
        encoder_out, encoder_lens, _, _, _ = model.encode(
            waveform.unsqueeze(0), lengths
        )
        logits = model.ctc.ctc_lo(encoder_out)
        token_ids = logits[0, : encoder_lens[0]].argmax(dim=-1).tolist()

    hypothesis = ctc_greedy_decode(token_ids, model.blank_id, model.token_list)
    tag = "CTC diagnostic, uninitialized adapted frontend" if args.no_adapt else "CTC diagnostic, adapted frontend"
    print(f"[{tag}] {args.wav.name}")
    print(f"HYP: {hypothesis}")


if __name__ == "__main__":
    main()
