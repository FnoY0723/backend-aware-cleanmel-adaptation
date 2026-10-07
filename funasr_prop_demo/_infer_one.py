"""Run FunASR-Nano inference for a single variant and emit JSON lines.

variant = "prop"        : CleanMel front-end + residual Adapter (our method)
variant = "unprocessed" : the frozen FunASR-Nano backend, no enhancement

This worker is spawned once per variant by ``run_demo.py`` so that the two
model classes register cleanly in separate processes.
"""
import argparse
from pathlib import Path
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "funasr_src")
sys.path.insert(0, SRC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["prop", "unprocessed"], required=True)
    ap.add_argument("--model-dir", default="FunAudioLLM/Fun-ASR-Nano-2512",
                    help="ModelScope id or a local snapshot directory of Fun-ASR-Nano-2512")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--examples", required=True)
    args = ap.parse_args()

    wavs = sorted(str(p) for p in Path(args.examples).iterdir()
                  if p.is_file() and p.suffix.lower() in {".wav", ".flac"})
    if not wavs:
        ap.error("--examples contains no .wav or .flac files")

    import torch
    from funasr.register import tables
    from funasr import AutoModel

    if args.variant == "prop":
        import model_cleanmel as M
    else:
        import model as M
    tables.model_classes["FunASRNano"] = M.FunASRNano

    model, kwargs = AutoModel.build_model(
        model=args.model_dir, device=args.device, trust_remote_code=False
    )
    model.eval()

    if args.variant == "prop":
        fe = kwargs.get("frontend", None)
        if fe is not None:
            def _passthrough(input, input_lengths, **kw):
                return input, input_lengths
            fe.forward = _passthrough

    for wav in wavs:
        key = os.path.splitext(os.path.basename(wav))[0]
        res = model.inference(data_in=[wav], **kwargs)
        text = res[0][0]["text"]
        print("RESULT " + json.dumps(
            {"variant": args.variant, "key": key, "text": text}, ensure_ascii=False
        ), flush=True)


if __name__ == "__main__":
    main()
