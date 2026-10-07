# Backend-Aware CleanMel Adaptation

Research code for **Backend-Aware CleanMel Adaptation for Robust Recognition with Frozen ASR Models**.

Repository: [https://github.com/FnoY0723/backend-aware-cleanmel-adaptation](https://github.com/FnoY0723/backend-aware-cleanmel-adaptation).

The method adapts **both CleanMel and a residual Adapter**, while keeping the
large-scale ASR backend frozen. This package provides the method implementation,
archived training configurations, a fine-tuned FunASR frontend checkpoint, and
an inference integration with the original FunASR-Nano LLM decoder.

## Contents

- `funasr_prop_demo/`: FunASR-Nano inference with the adapted frontend or the
  unprocessed frozen backend.
- `src/espnet2/`: method-specific ESPnet training overlay for FunASR and FireRed.
- `configs/`: adaptation recipes, ablation deltas, and pretraining records.
- `checkpoints/funasr_prop_frontend.pth`: adapted CleanMel + residual Adapter.
- `scripts/`: CPU regression checks, CTC diagnostics, and fixed-system bootstrap.
- `results/prop_summary.csv`: archived reported scores, not newly measured scores.

No dataset audio, frozen backend weights, CleanMel initialization checkpoint, or
adapted FireRed checkpoint is included. Obtain third-party data/models from their
official providers. Optional Conv-TasNet baseline code is excluded because its
redistribution permission has not been established; see `THIRD_PARTY_NOTICES.md`.

## FunASR inference

Use a compatible Linux/CUDA environment. The archived demo uses Python 3.11,
PyTorch 2.5.1, FunASR 1.3.30, Transformers 4.51.3, Mamba 2.2.4, and
causal-conv1d 1.5.0.post8. Install Mamba and causal-conv1d wheels matching your
PyTorch/CUDA/Python/ABI combination; see `funasr_prop_demo/README.md`.

From the repository root:

```bash
python -m pip install -r funasr_prop_demo/requirements.txt
python funasr_prop_demo/run_demo.py --examples /path/to/audio --device cuda:0
```

The directory must contain `.wav` or `.flac` files. Use `--variant prop` or
`--variant unprocessed` to run one path. Optional `--references references.json`
accepts a UTF-8 object mapping filename stems to transcripts. The demo outputs
text; it does not calculate benchmark WER/CER.

The default backend is `FunAudioLLM/Fun-ASR-Nano-2512` and may be downloaded by
FunASR. Prefer a verified local model snapshot:

```bash
python funasr_prop_demo/run_demo.py --examples /path/to/audio \
    --model-dir /path/to/Fun-ASR-Nano-2512 --device cuda:0
```

The included frontend checkpoint is loaded by default. Set `FUNASR_PROP_CKPT`
to select another checkpoint. Frontend and Adapter parameters must match the
model; incomplete or incompatible parameters fail during loading.

## Local regression checks

These checks require PyTorch but do not download ASR models, install Mamba,
or run training. They execute the actual numerical helper/frontend code with
controlled inputs and substitute test components for the neural network and
feature extractor. They do **not** validate full CleanMel or LLM inference.

```bash
python -m pip install -r requirements-test.txt
python scripts/test_release_regressions.py
```

The CPU checks cover short-utterance frequency preservation, temporal padding,
normalization, and adapted-module checkpoint loading. See `LOCAL_VALIDATION.json`
for the recorded environment, results, and untested paths. The authors confirm
that the archived inference demo ran successfully before the AAAI submission.
That historical inference test is distinct from the CPU regression checks:
full CUDA inference and benchmark scores have not been revalidated after the
release corrections below.

The release corrects a time-padding mask that previously operated on the Mel
axis, and protects demo normalization for silent audio. These are numerical
changes; the released checkpoint and archived score table are unchanged.
`RELEASE_CHANGES.md` describes their scope. Do not treat the archived scores as
new evaluations of the corrected release.

## Training overlay

Copy `src/espnet2/` into a compatible ESPnet installation. Entry points are:

```bash
python -m espnet2.bin.asr_funasrnano_train --config CONFIG.yaml ...
python -m espnet2.bin.asr_fireredasr_train --config CONFIG.yaml ...
```

The remaining arguments depend on your prepared ESPnet data directories.
`configs/expanded/*_run.sh` files are historical invocation records, not complete
standalone training scripts. The original ESPnet base revision, base training
launcher, external CTC module, local data recipes, and CleanMel initialization
weights are not included. The overlay is therefore not a one-command
reproduction environment. The demo's model integration is for inference;
use the ESPnet overlay for the archived training implementation.

| Setting | Archived record |
|---|---|
| FunASR adaptation | Seed 2026; AdamW 1e-4; 500 warmup steps; up to 20 epochs; 1,000 iterations/epoch; accumulation 10 |
| Released FunASR checkpoint | Original README identifies average of epochs 11--20 |
| FireRed adaptation | Configuration allows up to 50 epochs with accumulation 5; actual stopping point is not recovered |
| CleanMel initialization | 150-epoch pretraining record; epoch-149 initialization; external training project/data classes |

`ARTIFACT_PROVENANCE.md` records recoverable model identities and missing historical
records. Do not generalize the FunASR schedule or averaging to other experiments.
The confirmed FT Conv-TasNet baseline objective was `CTC + (-SI-SNR)` with 1:1
weights; its actual training wrapper is not available in this release.

## Diagnostics and reported results

After installing the ESPnet overlay, an encoder/CTC loading check is available:

```bash
python scripts/smoke_test_funasr.py --backend /path/to/model.pt \
    --skip-forward --device cpu
```

For a full encoder forward check use `--device cuda` without `--skip-forward`.
The script checks the archived backend weight SHA-256 by default.
`scripts/decode_example.py` is an auxiliary greedy-CTC diagnostic, not the original
backend decoder or the manuscript's evaluation pipeline. Its `--no-adapt` option
uses an uninitialized frontend, not the unprocessed-backend baseline.

The archived results lack utterance-level hypotheses/references and the complete
historical scoring manifests and normalizers. `scripts/bootstrap_ci.py` accepts
TSV columns `utt_id`, `errors`, and `reference_units`:

```bash
python scripts/bootstrap_ci.py counts.tsv --resamples 2000 --seed 2027
```

Its percentile interval is for one fixed system under utterance resampling,
not a paired system difference or training-seed uncertainty estimate.

## Attribution and release status

See `THIRD_PARTY_NOTICES.md` and `third_party_licenses/` for upstream license
records and attribution. No general reuse license has been specified for the original code contributions
or adapted weights. The applicable third-party terms remain in force; the
repository as a whole is not labeled MIT or Apache-2.0.

`MANIFEST.sha256` identifies distributed files. `archive_source_manifest.json`
records the unchanged original submission archive, including files omitted here.
Citation metadata are provided in `CITATION.cff` for this software. No publication
DOI or journal acceptance is claimed.
