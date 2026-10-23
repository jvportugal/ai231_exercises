#!/usr/bin/env python3
"""Download the public parquet files consumed by the RV90 training scripts."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination", type=Path,
        default=HERE / "data" / "ai231-me2-voice-commands",
    )
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="airimonda/ai231-me2-voice-commands",
        repo_type="dataset",
        local_dir=args.destination,
        allow_patterns=(
            "data/*.parquet",
            "supplemental_fil/*.parquet",
            "synthetic_negatives/*.parquet",
            "README.md",
            "LICENSE.md",
        ),
    )
    print(args.destination)


if __name__ == "__main__":
    main()
