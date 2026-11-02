"""Preserve additive reports without replacing the immutable weight archive."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files or any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Expected complete regular report files")
    destination.mkdir(parents=True, exist_ok=False)
    verified = []
    for path in files:
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as original, target.open("xb") as copied:
            shutil.copyfileobj(original, copied)
            copied.flush()
            os.fsync(copied.fileno())
        checksum = digest(path)
        if checksum != digest(target):
            raise ValueError("Report copy checksum mismatch")
        verified.append({"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": checksum})
    manifest = {"status": "verified", "verification": "sha256_source_and_destination",
                "destination": str(destination), "files": verified,
                "scope": "Additive postprocessing; original weight archive manifest unchanged"}
    payload = json.dumps(manifest, indent=2).encode() + b"\n"
    for directory in (destination, source):
        with (directory / "supplement-manifest.json").open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    print(json.dumps({"status": "verified", "file_count": len(verified), "destination": str(destination)}))


if __name__ == "__main__":
    main()
