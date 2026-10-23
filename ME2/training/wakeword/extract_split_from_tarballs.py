#!/usr/bin/env python3
"""Restore only the committed wake split clips from local MSWC tarballs.

This helper is for an existing sharded MSWC download. ``--map`` is the JSON
mapping from clip stem to shard number produced while indexing those archives.
It avoids unpacking hundreds of thousands of unrelated clips.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFESTS = ("pos_train.json", "pos_test.json", "neg_train.json", "oos_test.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", required=True,
                        help="directory containing tb<number>.tar.gz")
    parser.add_argument("--map", required=True,
                        help="JSON with a stem_tb mapping")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--splits", default=os.path.join(HERE, "splits"))
    parser.add_argument("--allow-missing", action="store_true",
                        help="extract mapped clips even if local shards are incomplete")
    args = parser.parse_args()

    required = set()
    for name in MANIFESTS:
        with open(os.path.join(args.splits, name), encoding="utf-8") as stream:
            required.update(row["stem"] for row in json.load(stream))
    with open(args.map, encoding="utf-8") as stream:
        mapping = json.load(stream)
    mapping = mapping.get("stem_tb", mapping)
    by_shard = defaultdict(list)
    missing = []
    for stem in sorted(required):
        shard = mapping.get(stem)
        if shard is None:
            missing.append(stem)
        else:
            by_shard[str(shard)].append(stem)
    if missing and not args.allow_missing:
        raise SystemExit("%d required stems absent from map (first: %s)" %
                         (len(missing), ", ".join(missing[:5])))
    if missing:
        print("warning: %d required stems absent from local shard map" % len(missing),
              flush=True)

    destination = os.path.abspath(os.path.expanduser(args.destination))
    os.makedirs(destination, exist_ok=True)
    written = skipped = 0
    for shard, stems in sorted(by_shard.items(), key=lambda item: int(item[0])):
        archive = os.path.join(args.archives, "tb%s.tar.gz" % shard)
        print("shard %s: %d clips" % (shard, len(stems)), flush=True)
        with tarfile.open(archive, "r:gz") as bundle:
            names = {member.name.lstrip("/"): member for member in bundle
                     if member.isfile()}
            for stem in stems:
                filename = stem + ".wav"
                target = os.path.join(destination, filename)
                if os.path.exists(target) and os.path.getsize(target) > 44:
                    skipped += 1
                    continue
                member = names.get(filename)
                if member is None:
                    raise RuntimeError("%s missing from %s" % (filename, archive))
                source = bundle.extractfile(member)
                if source is None:
                    raise RuntimeError("cannot read %s from %s" % (filename, archive))
                with source, open(target, "wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                written += 1
    print("ready=%d newly_extracted=%d destination=%s" %
          (written + skipped, written, destination))


if __name__ == "__main__":
    main()
