#!/usr/bin/env python3
"""Export a trained detector for the glasses, and the scores wake.c must reproduce.

    python export.py OUT_DIR [WAV ...]

Reads OUT_DIR/wake.pt, writes OUT_DIR/wake_model.h, and for each WAV the score of
every look the glasses would take (every 50 ms once 1.5 s is in) to
OUT_DIR/<name>.scores, one per line.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import HOP, SAMPLES, Detector, export, features, load  # noqa: E402


def main() -> int:
    out = Path(sys.argv[1])
    model = Detector()
    model.load_state_dict(torch.load(out / "wake.pt", map_location="cpu"))
    model.eval()
    export(model, out / "wake_model.h")
    for name in sys.argv[2:]:
        audio = load(Path(name))
        # As the glasses quantise: 16-bit samples.
        audio = np.round(audio * 32768).clip(-32768, 32767) / 32768
        # Looks end at sample 400 + 160 * (149 + 5k): every fifth frame once 150 are in.
        ends = [SAMPLES + 5 * HOP * k for k in range(1, 10000) if SAMPLES + 5 * HOP * k <= len(audio)]
        windows = np.stack([audio[e - SAMPLES:e] for e in ends]).astype(np.float32) if ends else np.zeros((0, SAMPLES))
        with torch.no_grad():
            scores = torch.sigmoid(model(features(torch.from_numpy(windows)))).numpy() if len(ends) else []
        (out / (Path(name).stem + ".scores")).write_text("".join(f"{s:.6f}\n" for s in scores))
        print(Path(name).name, len(scores), "looks, highest", f"{max(scores):.3f}" if len(scores) else "-")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
