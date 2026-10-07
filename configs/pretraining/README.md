# CleanMel pretraining record

The shared initialization was trained for 150 epochs on eight GPUs with
`configs/pretraining/cleanmel_16ms.yaml`. The ASR adaptation experiments initialize
from `epoch149_metric-0.0098.ckpt`. This initialization checkpoint is documented but
not included in the archive; the included FunASR checkpoint contains the subsequently
fine-tuned CleanMel and Adapter weights. Training corpora must be supplied separately.

Recorded epoch-149 validation values:

| Metric | Value |
|---|---:|
| `val/MRM` | 0.00979769 |
| `val/logmel_mse_all` | 0.23060510 |
| `val/logmel_L1` | 0.21569057 |
| `val/metric` | -0.00979769 |

The model has 5.369M parameters. The recorded forward complexity is 180.05 GFLOPs
for a four-second waveform, or approximately 45.0 GFLOPs/s. TensorBoard event files,
optimizer states, the pretraining checkpoint, and training corpora are excluded from
the archive.
