# Third-party attribution and license status

Repository: https://github.com/FnoY0723/backend-aware-cleanmel-adaptation.

No general reuse license has been specified for the original code contributions
or adapted weights. Existing third-party licenses remain applicable to their
respective components; the repository as a whole is not labeled MIT or
Apache-2.0. Original source headers and upstream license records are retained.

The core package excludes the Conv-TasNet implementation and its baseline
config/run record because redistribution permission for that implementation
has not been established. Its source-match evidence is retained for provenance.

## Upstream license records

Copies of the current upstream license texts were retrieved on 2026-10-06 into
`third_party_licenses/`; `sources.json` records the source URL and content hash.
These texts preserve the upstream terms; they are not new license grants from the
paper authors and do not establish which historical source revision was used.

| Upstream | Current license text | Corresponding archive area / relationship |
|---|---|---|
| Audio-WestlakeU/CleanMel | Apache-2.0 | Modified CleanMel frontend and pretraining integration |
| QwenAudio/Fun-ASR | Apache-2.0 | Fun-ASR model integration and backend model-family source |
| FireRedTeam/FireRedASR2S | Apache-2.0 | FireRed encoder/decoder integration |
| espnet/espnet | Apache-2.0 | ESPnet task/model overlay and required external toolkit |
| modelscope/FunASR | MIT | External FunASR Python toolkit used by the demo |

The table maps known dependencies/integration areas, not a verified file-by-file
copyright or derivative-work inventory. Complete historical source revisions were not retained in the original archive.
The current license records do not establish an unknown historical file's origin
or grant a new license to unrelated files.

## Component scope and outstanding license choices

1. `src/espnet2/asr/frontend/Conv_TasNet.py` reliably matches
   [JusperLee/Conv-TasNet, `Conv-TasNet_lightning/model.py`](https://github.com/JusperLee/Conv-TasNet/blob/1d05c4730e59034a0f3dfc8d0ee987f9988820b5/Conv-TasNet_lightning/model.py).
   Its Kai Li / 2020-07-09 18:22:54 header matches exactly, and nine top-level
   classes/functions have identical Python ASTs (see `CONVTASNET_SOURCE_MATCH.json`).
   The checked upstream tree contains no LICENSE/NOTICE file. The current
   Hugging Face model card's MIT label does not establish licensing of the
   separate GitHub implementation. This identifies a matching implementation
   lineage, not the historical experimental checkout. This
   implementation and its archived baseline configs are therefore excluded from
   the core package. Their presence in the extended local review copy does not
   imply a code-redistribution license.
2. No general reuse license has yet been specified for original contributions
   or `checkpoints/funasr_prop_frontend.pth`. Contact the authors regarding reuse
   beyond the applicable third-party terms; no license choice is inferred here.
3. Applicable upstream license texts and original source headers are retained,
   and local modifications are recorded in `RELEASE_CHANGES.md`. The original
   archive did not retain complete source histories.
4. The CHiME-4/RealMAN example recordings have been omitted only from this copy.
   Users obtain datasets and pretrained third-party models from official providers
   under their own terms. No corpus redistribution permission is presumed.

No credentials or author-machine absolute paths were found by the limited static
scan described in `LOCAL_VALIDATION.json`. That scan is not a guarantee that every
possible secret format or redistribution obligation was identified.
