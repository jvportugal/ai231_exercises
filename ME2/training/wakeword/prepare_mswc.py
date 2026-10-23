#!/usr/bin/env python3
"""Extract the exact wake-word split audio from an unpacked MSWC English tree."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import soundfile as sf


HERE = Path(__file__).resolve().parent
MANIFESTS = ("pos_train.json", "pos_test.json", "neg_train.json", "oos_test.json")


def records():
    seen = {}
    for name in MANIFESTS:
        with (HERE / "splits" / name).open(encoding="utf-8") as handle:
            for row in json.load(handle):
                seen[row["stem"]] = row
    return list(seen.values())


def locate(root: Path, word: str, basename: str) -> Path | None:
    candidates = (
        root / word / (basename + ".opus"),
        root / "clips" / word / (basename + ".opus"),
        root / "en" / word / (basename + ".opus"),
        root / "en" / "clips" / word / (basename + ".opus"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = [path for path in root.rglob(basename + ".opus")
               if path.parent.name == word]
    return matches[0] if len(matches) == 1 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path,
                        help="unpacked MSWC English audio directory")
    parser.add_argument("--destination", type=Path, default=HERE / "data" / "wavs")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)

    missing = []
    rows = records()
    for index, row in enumerate(rows, 1):
        stem = row["stem"]
        word = row["word"]
        basename = stem[len(word) + 1:]
        source = locate(args.source, word, basename)
        if source is None:
            missing.append(f"{word}/{basename}.opus")
            continue
        audio, _ = librosa.load(source, sr=16000, mono=True)
        sf.write(args.destination / (stem + ".wav"), audio, 16000,
                 subtype="PCM_16")
        if index % 500 == 0:
            print(f"prepared {index}/{len(rows)}", flush=True)

    if missing:
        preview = "\n".join(missing[:20])
        raise SystemExit(
            f"missing {len(missing)} of {len(rows)} required clips; first entries:\n{preview}")
    print(f"prepared {len(rows)} exact split clips in {args.destination}")


if __name__ == "__main__":
    main()
