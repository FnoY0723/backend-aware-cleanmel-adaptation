# Artifact and software provenance

Prepared 2026-10-06; documentation updated 2026-10-07. Derived from the existing `AAAI_2027_fail/code_supplement` archive.
Claims below distinguish locally verified artifact identities, archive statements,
and information not present in that archive. Current upstream repository revisions
were not substituted for the historical experimental versions.

| Component | Recoverable identity | Evidence and scope |
|---|---|---|
| Included FunASR adapted frontend | `funasr_prop_frontend.pth`; 35,399,567 bytes; SHA-256 `2f9d57b232a6f4109cd301f27a650bcf6f385c30f594096b6524148db673363a` | Hash and byte size verified locally; original README identifies `frontend.*`/`preencoder.*`, average of epochs 11--20. Training logs/optimizer states are not supplied. |
| Frozen FunASR backend | ModelScope `FunAudioLLM/Fun-ASR-Nano-2512`; `model.pt`; recorded 2,127,426,538 bytes; SHA-256 `81fec8616083c69377f3ceef36aba3655660ee0ca69a5d4a1e9810cd340ca499` | Original README and `scripts/smoke_test_funasr.py`. Not included or re-downloaded for this audit. The script records revision `master`, not an immutable commit. The checksum pins the weight file, not all tokenizer/config/source files. |
| CleanMel initialization | `epoch149_metric-0.0098.ckpt`; 150-epoch pretraining record | `configs/pretraining/README.md` and YAML; initialization weights and their hash absent. This modified hidden-size-120 model is not asserted identical to a currently published upstream checkpoint. |
| FireRedASR2-AED and -LLM | Public model identifiers and repositories listed below | No immutable historical revision, checkpoint hash, or adapted FireRed weights present. |
| Conv-TasNet initialization | Default path `checkpoints/convtasnet/model.ckpt` | Wrapper only; pretrained weight artifact/version/hash not present. |
| PyTorch/torchaudio | 2.5.1 / 2.5.1 | Archived top-level and demo requirements; README records CUDA 12.4 for core environment. |
| FunASR library | 1.3.30 | Demo requirements only; not proof of every historical training run's installed version. |
| Transformers / ModelScope | 4.51.3 / 1.38.1 | Demo requirements; top-level also pins ModelScope 1.38.1. |
| Mamba / causal-conv1d | 2.2.4 / 1.5.0.post8 | Demo requirements; demo README records Python 3.11, CUDA 12.x, cxx11abiFALSE wheel names. |
| NumPy / PyYAML | 1.24.4 / 6.0.2 | Top-level requirements. |
| Lightning packages | `pytorch-lightning==2.6.1`, `lightning==2.6.5` | Both archived pins retained; these are distinct package records, not an inferred single historical environment lock. |
| ESPnet base source | Unknown historical commit/version | Overlay imports external ESPnet classes including training CTC implementation. Exact base source is not archived. |
| CleanMel/Fun-ASR/FireRed source | Unknown historical Git commits | No `.git` metadata, lockfile, or commit identifiers found in supplied archive. |

Official project/model locations retained from the archive:

- CleanMel: https://github.com/Audio-WestlakeU/CleanMel
- Fun-ASR: https://github.com/QwenAudio/Fun-ASR
- FunASR toolkit: https://github.com/modelscope/FunASR
- Fun-ASR-Nano-2512: https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512
- Fun-ASR-Nano-2512 mirror: https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512
- FireRedASR2S: https://github.com/FireRedTeam/FireRedASR2S
- AED: https://www.modelscope.cn/models/xukaituo/FireRedASR2-AED
- AED mirror: https://huggingface.co/FireRedTeam/FireRedASR2-AED
- LLM: https://www.modelscope.cn/models/xukaituo/FireRedASR2-LLM
- LLM mirror: https://huggingface.co/FireRedTeam/FireRedASR2-LLM
- ESPnet: https://github.com/espnet/espnet

Different hosting mirrors need not serve byte-identical weights/configuration.
Use the recorded weight hash when selecting FunASR weights for the existing CTC
smoke test; no equivalent FireRed hash can be supplied from this archive.

Still needed for exact historical reproduction: the original ESPnet base revision
and CTC module; training/evaluation environment records; FireRed source/checkpoint
identifiers and adapted weights; CleanMel initialization; the FT Conv-TasNet
CTC + negative-SI-SNR (1:1) wrapper; benchmark channel/manifests and actual scoring
normalizers. These missing records are not replaced with newly reconstructed
material labeled as historical evidence.

## Corrected release versus archived experiments

The 2026-10-06 release corrections change time-padding masking in the FunASR
and FireRed frontends and silent-input normalization in the inference demo.
Adapted weights and reported scores remain byte-for-byte unchanged. CPU regression
checks validate the corrections, not the historical benchmark scores or the full
CUDA inference path. The original submission code remains available locally as
the historical reference; details are in `RELEASE_CHANGES.md`.

## Historical inference test and current validation

The authors confirm that the archived inference demo was executed successfully
before the AAAI submission. This is a historical author-confirmed test, distinct
from the ten CPU regressions of the corrected release. Full CleanMel/Mamba and
backend LLM inference after the numerical corrections has not been rerun in the
current validation environment. The repository for this work is
https://github.com/FnoY0723/backend-aware-cleanmel-adaptation.
