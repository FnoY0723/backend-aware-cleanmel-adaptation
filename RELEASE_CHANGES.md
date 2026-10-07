# Changes from the existing AAAI code archive

Initially prepared 2026-10-06; documentation updated 2026-10-07.
Changes affect the distributed code copies only; the source archive remains
unchanged and its file hashes are recorded in `archive_source_manifest.json`.

- Removed four CHiME-4/RealMAN `.flac` recordings from the copied package. Added an
  audio-input notice. No recordings were removed from the original archive.
- Replaced obsolete demonstration claims with user-provided audio instructions.
  `run_demo.py` now requires `--examples` and accepts an optional reference JSON;
  missing/empty audio inputs fail before model loading/downloading. `_infer_one.py`
  supports that path and case-insensitive `.wav`/`.flac` extensions.
- Moved the three unchanged custom encoder files from `asr/frontend/encoder/` to
  `asr/encoder/`, matching the existing task imports.
- Corrected README and Adapter docstrings to state that both CleanMel and the
  residual Adapter are trainable during adaptation.
- Corrected `decode_example.py` documentation/output labels: auxiliary CTC greedy
  decoding is diagnostic, not the manuscript's backend decoder. It requires a
  user-provided waveform; its uninitialized-frontend option is not the unprocessed
  baseline. The model computation and diagnostic decoding logic are unchanged.
- Documented author-confirmed FT Conv-TasNet loss `CTC + (-SI-SNR)` with 1:1 weights,
  while explicitly recording that its actual loss wrapper is not archived. The
  potentially inconsistent historical expanded config is preserved as evidence,
  not silently rewritten as a reconstructed exact experiment.
- Preserved all adapted checkpoint bytes, score CSV values, dependency pins,
  main-method recipe/configuration contents, and main-method model computation code.
- Added artifact provenance and uncertainty records, third-party license copies
  from verified upstream URLs, and release-status notes. No overall reuse license
  was assigned to original code or adapted weights.
- Refreshed the package file manifest after local validation. Full GPU inference,
  training and benchmark scoring were not rerun for this packaging operation.

## Core-method exclusions

The core package omits `src/espnet2/asr/frontend/Conv_TasNet.py`,
`configs/expanded/funasr_ft_convtasnet.yaml`, and
`configs/expanded/funasr_ft_convtasnet_run.sh`. The extended local review copy and
original archive retain them. The archived task modules contain no Conv-TasNet
import/registration, so no task-source changes were required and the main-method
imports were preserved. `CONVTASNET_SOURCE_MATCH.json` records comparison with the
excluded original file, not a file claimed to remain in this core package.

This omission removes that particular unresolved code-permission issue from the
core package; it does not select a root license or certify every original
contribution/weight's distribution rights. Third-party license copies remain.

## Runtime corrections and CPU regression checks (2026-10-06)

These corrections supersede the initial packaging statement that model
computation was unchanged. They affect both the core and extended local review copies;
the original submission archive, included checkpoint bytes, recipes, and archived
result values remain unchanged.

- Corrected the padding-mask dimension from the Mel axis to the time axis in the
  FunASR inference demo and the FunASR/FireRed ESPnet frontends. At the masking
  point the tensor is `(batch, 80, time)`. Previously a 61-frame utterance could
  have Mel bins 61--79 masked at every frame. Batch padding is now masked in time.
- Removed the unused unconditional `tkinter` import from the inference frontend.
- Protected demo waveform normalization against a zero peak; non-silent inputs
  retain the previous normalization calculation.
- Corrected the remaining Adapter class documentation: both CleanMel and the
  residual Adapter are trained during adaptation.
- Made adapted-module loading fail on missing, unexpected, or incompatible
  module parameters, instead of continuing with partially loaded weights.
- Added CPU regression checks using the actual source functions/forward code,
  controlled features and neural-network substitutes. Added a separate test
  requirements file; archived inference/training dependency pins are unchanged.
- Added `.gitignore` for environment files, caches, local data, downloaded models,
  and audio. The supplied adapted checkpoint remains eligible for version control.

The time-mask change affects short utterances and padded batches. In the demo,
an unpadded single utterance with at least 80 frontend frames had an all-false
old mask; this specific issue did not remove its Mel bins. This observation does
not quantify any historical benchmark impact. The corrected code has not been
used to replace or relabel archived experimental results. Full CUDA inference,
training, and benchmark scoring are outside the completed CPU checks.

## Repository documentation (2026-10-07)

- Updated the project title to **Backend-Aware CleanMel Adaptation for Robust Recognition with Frozen ASR Models** and linked the author-created repository:
  https://github.com/FnoY0723/backend-aware-cleanmel-adaptation.
- Added software citation metadata for Ying Fang, Nian Shao, and Xiaofei Li.
  No article DOI, acceptance claim, or unverified release date was added.
- Recorded the author-confirmed successful inference test before the AAAI
  submission separately from the targeted CPU regressions of the corrected code.
- Clarified that no general reuse license has been specified for original code
  or adapted weights; existing third-party license terms remain unchanged.

This documentation update does not change the numerical implementation, weights,
archived results, or the ten CPU regression tests. Upload status is recorded
separately in `LOCAL_VALIDATION.json`.
