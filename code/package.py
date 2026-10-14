#!/usr/bin/env python3
"""Put the wake word model and its dataset in one folder (and a zip), with a manifest.

    python package.py WAKE_DIR OUT_DIR

WAKE_DIR is where gen_data.py / train.py ran (data/, out/, the scripts). The
manifest says, for every clip, its label, the voice that said it and the licence
that voice comes under, so commercial use can keep only what is allowed.
"""

from __future__ import annotations

import csv
import shutil
import sys
import zipfile
from pathlib import Path

SOURCES = [
    ("en_US-libritts_r", "Piper en_US-libritts_r-medium (LibriTTS-R speakers)", "CC BY 4.0", "yes"),
    ("en_GB-vctk", "Piper en_GB-vctk-medium (VCTK speakers)", "CC BY 4.0", "yes"),
    ("en_US-l2arctic", "Piper en_US-l2arctic-medium (L2-ARCTIC speakers)", "CC BY-NC 4.0", "no"),
    ("mac_", "macOS system voice (say)", "Apple macOS licence: personal, non-commercial", "no"),
]


def source_of(name: str):
    stem = name[4:] if name.startswith("llm_") else name
    for prefix, voice, licence, commercial in SOURCES:
        if stem.startswith(prefix):
            return voice, licence, commercial
    return "unknown", "unknown", "no"


def main() -> int:
    wake, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists():
        shutil.rmtree(out)
    (out / "model").mkdir(parents=True)
    for name in ("wake.pt", "wake_model.h", "report.txt"):
        shutil.copy(wake / "out" / name, out / "model" / name)
    (out / "code").mkdir()
    for name in ("gen_data.py", "gen_negatives.py", "train.py", "export.py", "package.py", "wake.c", "wake.h",
                 "wake_test.c"):
        if (wake / name).exists():
            shutil.copy(wake / name, out / "code" / name)
    shutil.copytree(wake / "data" / "pos", out / "dataset" / "pos")
    shutil.copytree(wake / "data" / "neg", out / "dataset" / "neg")
    if (wake / "data" / "negative_sentences.txt").exists():
        shutil.copy(wake / "data" / "negative_sentences.txt", out / "dataset" / "negative_sentences.txt")
    counts = {}
    with open(out / "dataset" / "manifest.csv", "w", newline="", encoding="utf-8") as manifest:
        writer = csv.writer(manifest)
        writer.writerow(["file", "label", "kind", "voice", "licence", "commercial_use"])
        for label in ("pos", "neg"):
            for clip in sorted((out / "dataset" / label).glob("*.wav")):
                voice, licence, commercial = source_of(clip.stem)
                kind = ("wake word" if label == "pos" else
                        "everyday sentence (LLM-written)" if clip.stem.startswith("llm_") else
                        "near miss" if clip.stem.endswith("_close") or "_neg" in clip.stem else "sentence")
                writer.writerow([f"{label}/{clip.name}", label, kind, voice, licence, commercial])
                key = (label, commercial)
                counts[key] = counts.get(key, 0) + 1
    summary = "\n".join(f"{label} commercial_use={c}: {n}" for (label, c), n in sorted(counts.items()))
    (out / "dataset" / "counts.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(out.rglob("*")):
            zipped.write(path, path.relative_to(out.parent))
    print("wrote", archive, archive.stat().st_size // 1048576, "MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
