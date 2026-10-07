"""Utterance-level percentile-bootstrap CI for corpus WER/CER.

Input TSV columns: utt_id, errors, reference_units.  The edit and reference counts
must be computed after applying the same text normalizer to hypothesis and reference.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load_counts(path: Path) -> tuple[np.ndarray, np.ndarray]:
    errors, refs = [], []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            errors.append(float(row["errors"]))
            refs.append(float(row["reference_units"]))
    return np.asarray(errors), np.asarray(refs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("counts", type=Path)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    errors, refs = load_counts(args.counts)
    if len(errors) == 0 or np.any(refs < 0) or refs.sum() == 0:
        raise ValueError("invalid or empty count file")

    rng = np.random.default_rng(args.seed)
    samples = np.empty(args.resamples)
    for sample in range(args.resamples):
        indices = rng.integers(0, len(errors), len(errors))
        samples[sample] = 100.0 * errors[indices].sum() / refs[indices].sum()

    point = 100.0 * errors.sum() / refs.sum()
    low, high = np.percentile(samples, [2.5, 97.5])
    print(f"error_rate={point:.4f}")
    print(f"ci95=[{low:.4f}, {high:.4f}]")
    print(f"half_width={(high - low) / 2:.4f}")


if __name__ == "__main__":
    main()
