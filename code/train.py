#!/usr/bin/env python3
"""Train the "Hey Cerebel" detector that runs inside cerebel-mic on the glasses.

    python train.py DATA_DIR OUT_DIR [--epochs N]

DATA_DIR has pos/ and neg/ from gen_data.py. Writes OUT_DIR/wake_model.h (weights
for wake.c) and OUT_DIR/report.txt.

Everything here must match wake.c exactly: 16 kHz sound, 25 ms Hann windows every
10 ms, a 512-point FFT, 40 log-mel bands, 1.5 s of them (150 frames) at a time, a
small temporal-convolution network (TC-ResNet style) that ends in one score.
"""

from __future__ import annotations

import argparse
import math
import random
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RATE = 16000
WIN, HOP, NFFT, MELS = 400, 160, 512, 40
FRAMES = 150  # 1.5 s
SAMPLES = (FRAMES - 1) * HOP + WIN
CH = 48


def mel_matrix() -> np.ndarray:
    """Triangular filters on the HTK mel scale, 60 Hz to 7600 Hz, applied to NFFT/2+1 power bins."""
    def mel(f): return 2595.0 * math.log10(1.0 + f / 700.0)
    def hz(m): return 700.0 * (10 ** (m / 2595.0) - 1.0)
    points = [hz(mel(60) + (mel(7600) - mel(60)) * i / (MELS + 1)) for i in range(MELS + 2)]
    bins = [p * NFFT / RATE for p in points]
    matrix = np.zeros((MELS, NFFT // 2 + 1), dtype=np.float32)
    for m in range(MELS):
        left, centre, right = bins[m], bins[m + 1], bins[m + 2]
        for k in range(NFFT // 2 + 1):
            if left < k <= centre:
                matrix[m, k] = (k - left) / (centre - left)
            elif centre < k < right:
                matrix[m, k] = (right - k) / (right - centre)
    return matrix


WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)).astype(np.float32)  # periodic Hann
MEL = mel_matrix()


def features(audio: torch.Tensor) -> torch.Tensor:
    """(batch, SAMPLES) in [-1, 1] -> (batch, MELS, FRAMES) log-mel, as wake.c computes it."""
    frames = audio.unfold(1, WIN, HOP) * torch.from_numpy(WINDOW).to(audio.device)
    spectrum = torch.fft.rfft(frames, n=NFFT)
    power = spectrum.real ** 2 + spectrum.imag ** 2
    mel = power @ torch.from_numpy(MEL).to(audio.device).T
    return torch.log(mel + 1e-6).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, stride: int) -> None:
        super().__init__()
        self.depth = nn.Conv1d(CH, CH, 9, stride=stride, padding=4, groups=CH, bias=False)
        self.depth_norm = nn.BatchNorm1d(CH)
        self.point = nn.Conv1d(CH, CH, 1, bias=False)
        self.point_norm = nn.BatchNorm1d(CH)
        self.stride = stride

    def forward(self, x):
        y = F.relu(self.depth_norm(self.depth(x)))
        y = self.point_norm(self.point(y))
        skip = x[:, :, ::self.stride] if self.stride > 1 else x
        return F.relu(y + skip[:, :, :y.shape[2]])


class Detector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(MELS))
        self.register_buffer("std", torch.ones(MELS))
        self.first = nn.Conv1d(MELS, CH, 3, stride=2, padding=1, bias=False)
        self.first_norm = nn.BatchNorm1d(CH)
        self.blocks = nn.ModuleList([Block(1), Block(2), Block(2), Block(2)])
        self.out = nn.Linear(CH, 1)

    def forward(self, mel):  # (batch, MELS, FRAMES) log-mel
        x = (mel - self.mean[None, :, None]) / self.std[None, :, None]
        x = F.relu(self.first_norm(self.first(x)))
        for block in self.blocks:
            x = block(x)
        return self.out(x.mean(dim=2)).squeeze(1)


# --- sound ----------------------------------------------------------------------------

def load(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        return np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768


def trim(audio: np.ndarray) -> np.ndarray:
    """Cut the silence the voices leave around the words."""
    level = np.convolve(np.abs(audio), np.ones(160) / 160, mode="same")
    loud = np.nonzero(level > 0.02 * level.max())[0]
    return audio[max(0, loud[0] - 800):loud[-1] + 800] if len(loud) else audio


def coloured_noise(n: int) -> np.ndarray:
    white = np.random.randn(n).astype(np.float32)
    tilt = random.choice((0.0, 0.5, 1.0, 2.0))  # white, between, pink, brown-ish
    spectrum = np.fft.rfft(white)
    f = np.arange(len(spectrum), dtype=np.float32) + 1
    noise = np.fft.irfft(spectrum / f ** (tilt / 2), n)
    return (noise / (np.abs(noise).max() + 1e-6)).astype(np.float32)


def room(audio: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        return audio
    length = int(RATE * random.uniform(0.05, 0.4))
    decay = np.exp(-np.arange(length) / (length / random.uniform(3, 7)))
    impulse = np.random.randn(length).astype(np.float32) * decay
    impulse[0] = 1.0
    wet = np.convolve(audio, impulse)[:len(audio)]
    return wet / (np.abs(wet).max() + 1e-6) * np.abs(audio).max()


def microphone(audio: np.ndarray) -> np.ndarray:
    """A small glasses microphone: no deep bass, uneven treble, some level."""
    from scipy.signal import butter, lfilter
    low = random.uniform(80, 300)
    high = random.uniform(3500, 7800)
    b, a = butter(2, [low / (RATE / 2), high / (RATE / 2)], btype="band")
    return lfilter(b, a, audio).astype(np.float32)


# Voices whose output may not be used commercially (see package.py): left out with --commercial.
NOT_COMMERCIAL = ("en_US-l2arctic", "mac_", "llm_en_US-l2arctic", "llm_mac_")


class Sounds:
    commercial = False

    def __init__(self, folder: Path, split: str) -> None:
        pos = sorted((folder / "pos").glob("*.wav"))
        neg = sorted((folder / "neg").glob("*.wav"))
        if Sounds.commercial:
            pos = [p for p in pos if not p.stem.startswith(NOT_COMMERCIAL)]
            neg = [p for p in neg if not p.stem.startswith(NOT_COMMERCIAL)]
        # Speakers kept apart for checking: every tenth Piper speaker and every fifth macOS voice.
        def held(path):
            parts = path.stem.split("_")
            return (parts[-2].isdigit() and int(parts[-2]) % 10 == 7) or (parts[0] == "mac" and hash(parts[1]) % 5 == 0)
        keep = (lambda p: held(p)) if split == "check" else (lambda p: not held(p))
        self.pos = [trim(load(p)) for p in pos if keep(p)]
        self.neg = [trim(load(p)) for p in neg if keep(p)]
        self.talk = np.concatenate(self.neg)  # one long stream of not-the-wake-word

    def background(self, n: int) -> np.ndarray:
        kind = random.random()
        if kind < 0.35:
            start = random.randrange(0, len(self.talk) - n)
            return self.talk[start:start + n] * random.uniform(0.05, 0.6)
        if kind < 0.8:
            return coloured_noise(n) * random.uniform(0.002, 0.08)
        return np.zeros(n, dtype=np.float32)

    def place(self, words: np.ndarray, end_at: int) -> np.ndarray:
        out = self.background(SAMPLES)
        words = words[-end_at:] if len(words) > end_at else words
        start = end_at - len(words)
        out[start:start + len(words)] += words * random.uniform(0.3, 1.0)
        return out

    def example(self, positive: bool) -> np.ndarray:
        if positive:
            words = room(random.choice(self.pos))
            # The detector is asked at the end of the phrase, give or take a little.
            sound = self.place(words, SAMPLES - random.randint(0, int(0.4 * RATE)))
        else:
            kind = random.random()
            if kind < 0.35:  # half a wake word ("Hey ...") must not count
                words = random.choice(self.pos)
                words = words[:int(len(words) * random.uniform(0.2, 0.55))]
                sound = self.place(room(words), random.randint(len(words) // 2 + 1, SAMPLES))
            elif kind < 0.8:
                words = room(random.choice(self.neg))
                sound = self.place(words, random.randint(min(len(words), SAMPLES) // 2 + 1, SAMPLES))
            else:
                sound = self.background(SAMPLES)
        sound = microphone(sound) * (10 ** (random.uniform(-25, 6) / 20))
        return np.clip(sound, -1, 1)

    def batch(self, size: int, share: float = 0.35):
        labels = [random.random() < share for _ in range(size)]
        audio = np.stack([self.example(label) for label in labels])
        return torch.from_numpy(audio), torch.tensor(labels, dtype=torch.float32)


# --- export ---------------------------------------------------------------------------

def folded(conv, norm):
    """Batch norm folded into the convolution before it: weight, bias."""
    scale = norm.weight / torch.sqrt(norm.running_var + norm.eps)
    weight = conv.weight * scale[:, None, None]
    bias = norm.bias - norm.running_mean * scale
    return weight.detach().cpu().numpy(), bias.detach().cpu().numpy()


def export(model: Detector, path: Path) -> None:
    model = model.cpu().eval()
    arrays = {"WAKE_MEAN": model.mean.numpy(), "WAKE_STD": model.std.numpy()}
    arrays["WAKE_FIRST_W"], arrays["WAKE_FIRST_B"] = folded(model.first, model.first_norm)
    for i, block in enumerate(model.blocks):
        arrays[f"WAKE_B{i}_DW"], arrays[f"WAKE_B{i}_DB"] = folded(block.depth, block.depth_norm)
        arrays[f"WAKE_B{i}_PW"], arrays[f"WAKE_B{i}_PB"] = folded(block.point, block.point_norm)
    arrays["WAKE_OUT_W"] = model.out.weight.detach().numpy()
    arrays["WAKE_OUT_B"] = model.out.bias.detach().numpy()
    lines = ["/* Generated by cerebel_wake/train.py: \"Hey Cerebel\" detector weights. */",
             f"#define WAKE_CH {CH}", f"#define WAKE_MELS {MELS}", f"#define WAKE_FRAMES {FRAMES}",
             f"static const int WAKE_STRIDES[{len(model.blocks)}] = {{{', '.join(str(b.stride) for b in model.blocks)}}};"]
    for name, value in arrays.items():
        flat = ", ".join(f"{v:.7g}f" for v in np.asarray(value, dtype=np.float32).ravel())
        lines.append(f"static const float {name}[{np.asarray(value).size}] = {{{flat}}};")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- training -------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("out")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--commercial", action="store_true", help="train only on voices that allow commercial use")
    args = parser.parse_args()
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Sounds.commercial = args.commercial
    train, check = Sounds(Path(args.data), "train"), Sounds(Path(args.data), "check")
    print(f"train: {len(train.pos)} positive, {len(train.neg)} negative; check: {len(check.pos)} / {len(check.neg)}")

    model = Detector().to(device)
    with torch.no_grad():  # normalise the features with their own statistics
        sample = torch.cat([features(train.batch(64)[0].to(device)) for _ in range(8)])
        model.mean.copy_(sample.mean(dim=(0, 2)))
        model.std.copy_(sample.std(dim=(0, 2)) + 1e-3)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, 3e-3, total_steps=args.epochs * args.steps)
    check_sets = [check.batch(256) for _ in range(4)]
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for _ in range(args.steps):
            audio, labels = train.batch(96)
            scores = model(features(audio.to(device)))
            # A false wake is worse than a missed one: negatives weigh more.
            weight = torch.where(labels.to(device) > 0, 1.0, 2.0)
            loss = F.binary_cross_entropy_with_logits(scores, labels.to(device), weight=weight)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            got = [(torch.sigmoid(model(features(a.to(device)))).cpu(), l) for a, l in check_sets]
        probs = torch.cat([g for g, _ in got])
        labels = torch.cat([l for _, l in got])
        recall = (probs[labels > 0] > 0.5).float().mean().item()
        false = (probs[labels == 0] > 0.5).float().mean().item()
        print(f"epoch {epoch + 1}: loss {total / args.steps:.4f}  wake {recall:.3f}  false {false:.4f}", flush=True)
        if recall - 5 * false > best:
            best = recall - 5 * false
            torch.save(model.state_dict(), out / "wake.pt")

    model.load_state_dict(torch.load(out / "wake.pt"))
    model.eval()
    # False wakes per hour: a long stream of talk and noise, scored every 50 ms as on the glasses.
    pieces = []
    for i in range(len(check.neg) * 3):
        pieces += [microphone(room(check.neg[i % len(check.neg)])) * random.uniform(0.2, 1.0),
                   coloured_noise(RATE // 2) * 0.01]
    stream = np.concatenate(pieces)
    hours = len(stream) / RATE / 3600
    starts = np.arange(0, len(stream) - SAMPLES, 800)
    scores = []
    with torch.no_grad():
        for i in range(0, len(starts), 512):
            chunk = torch.from_numpy(np.stack([stream[s:s + SAMPLES] for s in starts[i:i + 512]]))
            scores.append(torch.sigmoid(model(features(chunk.to(device)))).cpu().numpy())
    scores = np.concatenate(scores)
    report = [f"check speakers: {len(check.pos)} positive clips, {len(check.neg)} negative clips",
              f"negative stream: {hours * 60:.1f} minutes"]
    with torch.no_grad():
        pos_scores = torch.cat([torch.sigmoid(model(features(check.batch(256, share=1.0)[0].to(device)))).cpu() for _ in range(4)])
    for threshold in (0.5, 0.7, 0.8, 0.9, 0.95):
        fires, last = 0, -99
        for i, score in enumerate(scores):
            if score > threshold and i - last > 20:  # one wake per second at most
                fires += 1
                last = i
        report.append(f"threshold {threshold}: wakes on {(pos_scores > threshold).float().mean().item():.1%} of "
                      f"'Hey Cerebel'; {fires / hours:.1f} false wakes per hour of talk")
    (out / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    export(model, out / "wake_model.h")
    print("wrote", out / "wake_model.h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
