# FunASR-Nano Prop. inference integration demo

This demo compares the released CleanMel + residual Adapter frontend with the
unprocessed frozen FunASR-Nano backend. Supply your own `.wav`/`.flac` audio:
no corpus clips are distributed here. This demo is part of
[Backend-Aware CleanMel Adaptation for Robust Recognition with Frozen ASR Models](https://github.com/FnoY0723/backend-aware-cleanmel-adaptation).

## Install

```bash
python -m pip install -r requirements.txt
```

The archived demo record lists PyTorch 2.5.1, FunASR 1.3.30, Transformers 4.51.3,
ModelScope 1.38.1, Mamba 2.2.4, and causal-conv1d 1.5.0.post8. Its tested wheel
naming record is CUDA 12.x / Python 3.11 / cxx11abiFALSE. Install Mamba/causal-conv1d
wheels matching the actual local PyTorch/CUDA/Python/ABI combination. The authors
confirm that the archived inference demo passed before the AAAI submission. The
corrected release has only undergone the CPU regressions described below; full
CUDA inference has not been rerun. CPU CleanMel forward compatibility is not claimed.

Official wheel releases: https://github.com/state-spaces/mamba/releases and
https://github.com/Dao-AILab/causal-conv1d/releases.

## Run

```bash
python run_demo.py --examples /path/to/your/audio --device cuda:0
python run_demo.py --examples /path/to/your/audio --variant prop
python run_demo.py --examples /path/to/your/audio --variant unprocessed
```

For optional reference text, use `--references /path/to/references.json` with
`{"utterance_stem": "reference transcript"}`. Without that file the demo prints
hypotheses without reference text. This demo does not score WER/CER.

The backend defaults to `FunAudioLLM/Fun-ASR-Nano-2512` and may be downloaded by
FunASR from ModelScope. Use `--model-dir /path/to/local/snapshot` to select a
verified local artifact. See `../ARTIFACT_PROVENANCE.md` for the archived backend
hash; a mutable hub default does not identify the historical revision.

The included adapted weights are loaded from
`../checkpoints/funasr_prop_frontend.pth`; override with `FUNASR_PROP_CKPT`.
CleanMel and the residual Adapter were both adapted; the downstream backend
remains fixed in the inference comparison. The worker registers the two model
classes in separate processes, and the Prop. path passes waveforms directly to
CleanMel before the residual Adapter and original backend decoder.

The demonstration is not the full benchmark evaluation pipeline. Its archived
`model_cleanmel.py` integration is used for inference here; its `forward` method
is not presented as a substitute for the ESPnet training overlay.

## Release corrections and checks

The current release fixes the frontend time-padding mask and silent-input
normalization, removes an unused Tk dependency, and rejects incomplete or
incompatible adapted-module checkpoints. See `../RELEASE_CHANGES.md`. Archived
checkpoint bytes and score tables are unchanged; the corrected release has not
been benchmarked again. CPU regression checks are separate from full GPU inference:

```bash
# Run from the repository root, preferably in a separate test environment.
python -m pip install -r requirements-test.txt
python scripts/test_release_regressions.py
```
