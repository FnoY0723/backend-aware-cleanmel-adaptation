"""One-command demo: run FunASR-Nano Prop. vs. unprocessed and compare.

Usage
-----
    python run_demo.py --examples /path/to/audio  # run both variants
    python run_demo.py --examples /path/to/audio --variant prop

The first run downloads ``FunAudioLLM/Fun-ASR-Nano-2512`` from ModelScope
(cached under ~/.cache/modelscope). The fine-tuned CleanMel + Adapter weights
used by the Prop. variant are read from ``../checkpoints/funasr_prop_frontend.pth``
(override with the FUNASR_PROP_CKPT environment variable).
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "_infer_one.py")

# Optional references are supplied by the user as a JSON object.



def run_variant(variant, model_dir, device, examples):
    cmd = [sys.executable, WORKER, "--variant", variant,
           "--model-dir", model_dir, "--device", device, "--examples", examples]
    print(f"\n>>> running variant: {variant}", flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"variant {variant} failed (exit {proc.returncode})")
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            rec = json.loads(line[len("RESULT "):])
            out[rec["key"]] = rec["text"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["prop", "unprocessed", "both"], default="both")
    ap.add_argument("--model-dir", default="FunAudioLLM/Fun-ASR-Nano-2512")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--examples", required=True,
                    help="Directory containing user-provided .wav or .flac files")
    ap.add_argument("--references", help="Optional UTF-8 JSON object: filename stem -> reference text")
    args = ap.parse_args()
    args.examples = os.path.abspath(args.examples)
    if not os.path.isdir(args.examples):
        ap.error("--examples must be an existing directory")
    if not any(name.lower().endswith((".wav", ".flac")) for name in os.listdir(args.examples)):
        ap.error("--examples contains no .wav or .flac files")
    references = {}
    if args.references:
        with open(args.references, encoding="utf-8") as stream:
            references = json.load(stream)
        if not isinstance(references, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in references.items()):
            ap.error("--references must contain a JSON object mapping strings to strings")

    variants = ["unprocessed", "prop"] if args.variant == "both" else [args.variant]
    results = {v: run_variant(v, args.model_dir, args.device, args.examples) for v in variants}

    keys = sorted({k for res in results.values() for k in res})
    print("\n" + "=" * 88)
    print("FunASR-Nano  unprocessed (frozen backend)  vs.  Prop. (CleanMel + Adapter)")
    print("=" * 88)
    for k in keys:
        print(f"\n[{k}]")
        print(f"  ref         : {references.get(k, '(no reference)')}")
        for v in variants:
            print(f"  {v:<12}: {results[v].get(k, '')}")


if __name__ == "__main__":
    main()
