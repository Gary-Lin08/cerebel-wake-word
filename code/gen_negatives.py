#!/usr/bin/env python3
"""More of what the wake word must not wake for: everyday speech from many voices.

    python gen_negatives.py OUT_DIR [--sentences N] [--llm URL]

Asks the local language model (llama-server, OpenAI-compatible) for varied everyday
sentences, English and some Chinese, then has random Piper speakers and macOS voices
say them. Writes 16 kHz WAVs into OUT_DIR/neg (next to gen_data.py's).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

import numpy as np

from gen_data import mac_say, mac_voices, piper_voices, save

TOPICS = ["work meetings", "family dinner", "shopping", "sports", "travel plans", "weather", "cooking", "school",
          "a doctor's visit", "friends chatting", "phone calls", "driving directions", "movies and music",
          "technology and phones", "office small talk", "restaurants", "the gym", "pets", "money and bills",
          "a party", "public transport", "a coffee shop", "talking to a voice assistant like Siri or Alexa",
          "greetings and names (Sarah, Isabel, Bella, Cyril, Serena, Gabriel)", "science class"]


def sentences(url: str, count: int) -> list[str]:
    found: set[str] = set()
    while len(found) < count:
        topic = random.choice(TOPICS)
        prompt = (f"Write 25 different short things people actually say out loud about {topic}. "
                  "Mix questions, commands and casual remarks, 3 to 14 words each. "
                  "One per line, no numbering, no quotes.")
        body = json.dumps({"model": "cerebel-llm", "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": 600, "temperature": 1.0}).encode()
        request = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as reply:
            text = json.loads(reply.read())["choices"][0]["message"]["content"]
        for line in text.splitlines():
            line = re.sub(r"^\W*\d*[.)]?\s*", "", line).strip().strip('"')
            if 2 <= len(line.split()) <= 16 and "cerebel" not in line.lower():
                found.add(line)
        print(len(found), "sentences", flush=True)
    return sorted(found)[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--sentences", type=int, default=3000)
    parser.add_argument("--llm", default="http://127.0.0.1:8088/v1")
    parser.add_argument("--voices", default=str(Path(__file__).resolve().parent / "voices"))
    args = parser.parse_args()
    random.seed(11)
    neg = Path(args.out) / "neg"
    neg.mkdir(parents=True, exist_ok=True)
    lines = sentences(args.llm, args.sentences)
    (Path(args.out) / "negative_sentences.txt").write_text("\n".join(lines), encoding="utf-8")
    voices = list(piper_voices(Path(args.voices)))
    macs = mac_voices()
    made = 0
    for i, line in enumerate(lines):
        for copy in range(2):
            if random.random() < 0.85:
                label, speakers, say = random.choice(voices)
                speaker = random.randrange(speakers)
                audio = say(line, speaker, random.uniform(0.8, 1.25))
                name = f"llm_{label}_{speaker}_{i}_{copy}"
            else:
                voice = random.choice(macs)
                audio = mac_say(line, voice, random.choice((160, 185, 210)))
                name = f"llm_mac_{voice.replace(' ', '_')}_{i}_{copy}"
            if audio is not None and len(audio) > 1600:
                save(neg / f"{name}.wav", np.asarray(audio, dtype=np.float32))
                made += 1
        if i % 200 == 0:
            print(i, "/", len(lines), "said;", made, "clips", flush=True)
    print("done:", made, "negative clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
