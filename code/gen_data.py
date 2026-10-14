#!/usr/bin/env python3
"""Make training sound for the "Hey Cerebel" wake word, all synthetic.

    python gen_data.py OUT_DIR [--voices DIR]

Positives: "Hey Cerebel" (and spellings that steer the voice to the right sound)
said by every speaker of three multi-speaker Piper voices (904 US, 109 UK, 24
non-native English speakers, Mandarin speakers among them) and the system voices of
macOS, at several speeds. Negatives: everyday sentences and words that sound close
("Hey Siri", "Isabel", "celebrate", "cerebral" ...) from the same voices.

Writes 16 kHz mono 16-bit WAVs to OUT_DIR/pos and OUT_DIR/neg.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

POSITIVE = ["Hey Cerebel", "Hey Cerebel.", "Hey, Cerebel!", "Hey Cerebel?", "Hey Serra bell", "Hey Sarah-bel",
            "Hey Sereh bell", "Hey Cerebel,"]
CLOSE = ["Hey Siri", "Hey Sarah", "Hey there", "Hey Isabel", "Hey Annabel", "Hey Mirabel", "Hey Rebel", "Hey Gabriel",
         "Hey Cindy", "Hey Celeste", "Hey Ceres", "Hey Sara Beth", "Hey Kimberly", "Hey Serena", "Hey Serengeti",
         "Hey Siri, call Bella", "celebrate", "cerebral", "terrible", "Sarah's bell", "hey seriously", "hey sir, a bell",
         "Hey Sabelle", "Hey Carol", "Hey Severe", "Hey there, bell", "Hey Cortana", "Hey Google", "OK Google",
         "Alexa", "Hey Siri, play music", "hey sorry about", "very well", "Sarah, tell", "a cherry belle",
         "Hey cereal", "Hey Cyril", "they serve well", "say a bell", "hey Sara, well"]
SENTENCES = """Can you send me the report by Friday
I think we should take the train instead
The weather is really nice today
Let's grab lunch after the meeting
Did you see the game last night
My phone battery is almost dead
We need to book a bigger room
Please turn the lights off when you leave
How much does the new model cost
She said the package arrived this morning
I'll call you back in five minutes
The coffee here is surprisingly good
What time does the store close
He forgot his keys again
Could you speak a little louder please
The project deadline moved to next week
I love the color of that jacket
Where did you park the car
Let me check my calendar first
The kids are playing in the garden
This song reminds me of college
We should celebrate your promotion
That was a terrible movie honestly
The cerebral cortex handles thinking
Hey everyone, thanks for coming
Hey, how are you doing today
Hey, can you pass me the salt
Hey there, nice to meet you
Siri, set a timer for ten minutes
OK, let's get started then
Sorry, I didn't catch your name
Isabel and Annabel are sisters
The bell rang at the end of class
Sarah will be late to the meeting
Seriously, that is the best pizza
Turn left at the next intersection
I need to buy groceries tonight
Can we move this to Thursday afternoon
The flight was delayed by two hours
Tell me more about your new job
Hey Siri, what's the weather like
Hey Google, play some jazz
Alexa, turn on the kitchen lights
My favorite season is autumn
Let's take a photo together
Start recording the lecture please
What's the capital of Australia
How far is the nearest gas station
Remind me to call mom tomorrow
I'm not sure that's a good idea
Thank you so much for your help""".splitlines()


def to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == 16000:
        return audio
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(16000, rate)
    return resample_poly(audio, 16000 // g, rate // g)


def save(path: Path, audio: np.ndarray) -> None:
    audio = audio / max(1e-6, np.abs(audio).max()) * 0.8
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes((audio * 32767).astype("<i2").tobytes())


def piper_voices(folder: Path):
    from piper import PiperVoice, SynthesisConfig

    for model in sorted(folder.glob("*.onnx")):
        voice = PiperVoice.load(str(model))
        speakers = voice.config.num_speakers or 1
        rate = voice.config.sample_rate

        def say(text, speaker, speed, voice=voice, rate=rate):
            config = SynthesisConfig(speaker_id=speaker if speakers > 1 else None, length_scale=speed,
                                     noise_scale=random.uniform(0.5, 0.9), noise_w_scale=random.uniform(0.6, 1.0))
            chunks = [chunk.audio_float_array for chunk in voice.synthesize(text, syn_config=config)]
            return to_16k(np.concatenate(chunks), rate) if chunks else None

        yield model.stem, speakers, say


def mac_voices():
    if sys.platform != "darwin":
        return []
    listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout.splitlines()
    novelty = {"Albert", "Bad", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Wobble", "Good", "Jester", "Organ",
               "Superstar", "Trinoids", "Whisper", "Zarvox", "Deranged", "Hysterical"}
    names = []
    for line in listing:
        parts = line.split("#")[0].split()
        if len(parts) >= 2 and parts[-1].startswith(("en_", "zh_")) and parts[0] not in novelty:
            names.append(" ".join(parts[:-1]))
    return sorted(set(names))


def mac_say(text: str, name: str, rate: int) -> np.ndarray | None:
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "s.wav"
        subprocess.run(["say", "-v", name, "-r", str(rate), "-o", str(out), "--file-format=WAVE",
                        "--data-format=LEI16@16000", text], capture_output=True)
        if not out.exists():
            return None
        with wave.open(str(out), "rb") as source:
            return np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--voices", default=str(Path(__file__).resolve().parent / "voices"))
    parser.add_argument("--per-speaker", type=int, default=3, help="positive clips per Piper speaker")
    args = parser.parse_args()
    random.seed(7)
    pos, neg = Path(args.out) / "pos", Path(args.out) / "neg"
    pos.mkdir(parents=True, exist_ok=True)
    neg.mkdir(parents=True, exist_ok=True)
    counts = [0, 0]

    def keep(kind: int, name: str, audio) -> None:
        if audio is None or len(audio) < 1600:
            return
        save((pos if kind else neg) / f"{name}.wav", audio.astype(np.float32))
        counts[kind] += 1

    for label, speakers, say in piper_voices(Path(args.voices)):
        for speaker in range(speakers):
            for n in range(args.per_speaker):
                keep(1, f"{label}_{speaker}_{n}", say(random.choice(POSITIVE), speaker, random.uniform(0.8, 1.3)))
            # Negatives from the same speaker: what it must not wake for.
            keep(0, f"{label}_{speaker}_close", say(random.choice(CLOSE), speaker, random.uniform(0.8, 1.3)))
            keep(0, f"{label}_{speaker}_talk", say(random.choice(SENTENCES), speaker, random.uniform(0.85, 1.2)))
            if speaker % 50 == 0:
                print(label, speaker, "/", speakers, "pos", counts[1], "neg", counts[0], flush=True)
    for name in mac_voices():
        for n, rate in enumerate((150, 175, 200, 225)):
            keep(1, f"mac_{name}_{n}".replace(" ", "_"), mac_say(random.choice(POSITIVE), name, rate))
        for n, text in enumerate(random.sample(CLOSE, 6) + random.sample(SENTENCES, 4)):
            keep(0, f"mac_{name}_neg{n}".replace(" ", "_"), mac_say(text, name, random.choice((160, 190, 220))))
    print("done: positives", counts[1], "negatives", counts[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
