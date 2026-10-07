"""Smoke-test the released FunASR frontend with the official Nano-2512 backend."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
import yaml

from espnet2.tasks.asr_funasrnano import ASRTask


TESTED_BACKEND_SHA256 = "81fec8616083c69377f3ceef36aba3655660ee0ca69a5d4a1e9810cd340ca499"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=Path)
    parser.add_argument(
        "--model-id", default="FunAudioLLM/Fun-ASR-Nano-2512"
    )
    parser.add_argument("--revision", default="master")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--adapted",
        type=Path,
        default=Path("checkpoints/funasr_prop_frontend.pth"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/expanded/funasr_prop.yaml"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=32000)
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--skip-backend-hash-check", action="store_true")
    return parser.parse_args()


def resolve_backend(args: argparse.Namespace) -> Path:
    if args.backend is not None:
        return args.backend
    from modelscope import snapshot_download

    download_kwargs = {
        "model_id": args.model_id,
        "revision": args.revision,
        "allow_file_pattern": ["model.pt", "config.yaml", "configuration.json"],
    }
    if args.model_dir is not None:
        download_kwargs["local_dir"] = str(args.model_dir)
    model_dir = Path(snapshot_download(**download_kwargs))
    return model_dir / "model.pt"


def main() -> None:
    args = parse_args()
    args.backend = resolve_backend(args)
    if not args.backend.is_file():
        raise FileNotFoundError(args.backend)
    if not args.adapted.is_file():
        raise FileNotFoundError(args.adapted)
    backend_sha256 = file_sha256(args.backend)
    if (
        not args.skip_backend_hash_check
        and backend_sha256 != TESTED_BACKEND_SHA256
    ):
        raise RuntimeError(
            f"Unexpected backend SHA-256: {backend_sha256}; "
            f"expected {TESTED_BACKEND_SHA256}"
        )

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config.pop("config", None)
    config["frontend_conf"]["model_path"] = None
    config["output_dir"] = str(Path(tempfile.gettempdir()) / "funasr_smoke_test")

    os.environ["FUNASR_MODEL_PATH"] = str(args.backend.resolve())
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
        temporary_config = Path(stream.name)

    try:
        model, _ = ASRTask.build_model_from_file(temporary_config, None, "cpu")
    finally:
        temporary_config.unlink(missing_ok=True)

    adapted = torch.load(args.adapted, map_location="cpu", weights_only=True)
    if not adapted or any(
        not key.startswith(("frontend.", "preencoder.")) for key in adapted
    ):
        raise RuntimeError("Adapted checkpoint contains unexpected modules")
    missing, unexpected = model.load_state_dict(adapted, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    missing_adapted = [
        key
        for key in missing
        if key.startswith(("frontend.", "preencoder."))
    ]
    if missing_adapted:
        raise RuntimeError(f"Missing adapted keys: {missing_adapted}")
    model_state = model.state_dict()
    if any(not torch.equal(model_state[key], value) for key, value in adapted.items()):
        raise RuntimeError("Adapted tensors differ after loading")

    report = {
        "backend_sha256": backend_sha256,
        "adapted_tensors": len(adapted),
        "unexpected_keys": len(unexpected),
        "missing_adapted_keys": len(missing_adapted),
    }
    if not args.skip_forward:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the Mamba forward smoke test")
        model = model.eval().to(args.device)
        waveform = torch.zeros(1, args.samples, device=args.device)
        lengths = torch.tensor([args.samples], device=args.device)
        with torch.inference_mode():
            encoder_out, encoder_lens, adapted_feats, adapted_lens, _ = model.encode(
                waveform, lengths
            )
            logits = model.ctc.ctc_lo(encoder_out)
        if not torch.isfinite(encoder_out).all() or not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite values found in the forward pass")
        report.update(
            encoder_shape=list(encoder_out.shape),
            encoder_lengths=encoder_lens.tolist(),
            adapted_shape=list(adapted_feats.shape),
            adapted_lengths=adapted_lens.tolist(),
            ctc_shape=list(logits.shape),
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
