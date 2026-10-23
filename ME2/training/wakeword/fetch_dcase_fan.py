#!/usr/bin/env python3
"""Fetch and safely extract the official DCASE 2020 development fan data.

The archive is deliberately kept outside Git. Its DCASE dataset license is
CC BY-NC-SA 4.0, so check that the intended use is compatible before use.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import urllib.request
import zipfile

URL = "https://zenodo.org/records/3678171/files/dev_data_fan.zip?download=1"
FILENAME = "dev_data_fan.zip"
MD5 = "649bdfc06263ae7a838963f43b6641e6"


def checksum(path):
    digest = hashlib.md5()  # nosec B324 - publisher's file-integrity checksum
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination):
    partial = destination + ".part"
    offset = os.path.getsize(partial) if os.path.exists(partial) else 0
    request = urllib.request.Request(
        url, headers={"Range": "bytes=%d-" % offset} if offset else {})
    with urllib.request.urlopen(request) as response:
        resumed = bool(offset and getattr(response, "status", None) == 206)
        mode = "ab" if resumed else "wb"
        with open(partial, mode) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    os.replace(partial, destination)


def safe_extract(archive, destination):
    os.makedirs(destination, exist_ok=True)
    root = os.path.realpath(destination)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = os.path.realpath(os.path.join(root, member.filename))
            if os.path.commonpath((root, target)) != root:
                raise RuntimeError("unsafe archive member: %s" % member.filename)
            if member.is_dir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.exists(target) and os.path.getsize(target) == member.file_size:
                continue
            with bundle.open(member) as source, open(target, "wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True,
                        help="directory for the archive and extracted files")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()
    destination = os.path.abspath(os.path.expanduser(args.destination))
    os.makedirs(destination, exist_ok=True)
    archive = os.path.join(destination, FILENAME)
    if not os.path.exists(archive) or checksum(archive) != MD5:
        print("downloading %s" % URL, flush=True)
        download(URL, archive)
    actual = checksum(archive)
    if actual != MD5:
        raise RuntimeError("checksum mismatch: expected %s, got %s" % (MD5, actual))
    print("verified %s" % actual, flush=True)
    if not args.no_extract:
        extracted = os.path.join(destination, "extracted")
        safe_extract(archive, extracted)
        print("extracted to %s" % extracted, flush=True)


if __name__ == "__main__":
    main()
