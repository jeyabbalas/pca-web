#!/usr/bin/env python3
"""Generate the demo app's bundled dataset: sklearn's digits (1797x64).

Writes two little-endian binaries into examples/demo/public/:
  digits.bin        — 1797*64 uint8 pixel values (0..16, row-major)
  digits-labels.bin — 1797 uint8 digit labels (0..9)

Re-run with:  .venv/bin/python python/generate_demo_data.py
"""

import os

import numpy as np
from sklearn.datasets import load_digits

OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "examples", "demo", "public"
)


def main():
    digits = load_digits()
    X = digits.data  # (1797, 64), float64 with integer values 0..16
    y = digits.target  # (1797,), int 0..9
    assert X.shape == (1797, 64), X.shape
    assert X.min() >= 0 and X.max() <= 16
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "digits.bin"), "wb") as f:
        f.write(np.ascontiguousarray(X, dtype=np.uint8).tobytes())
    with open(os.path.join(OUT_DIR, "digits-labels.bin"), "wb") as f:
        f.write(np.ascontiguousarray(y, dtype=np.uint8).tobytes())
    print(f"wrote {X.shape[0]}x{X.shape[1]} digits ({X.shape[0] * X.shape[1]} bytes) to {OUT_DIR}")


if __name__ == "__main__":
    main()
