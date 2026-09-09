#!/usr/bin/env python3
"""Create a deterministic SHA-256 inventory for a completed artifact tree."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checksum_output", type=Path, required=True)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="Exclude a root-relative path glob. May be repeated.",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    checksum_output = args.checksum_output.expanduser().resolve()
    excluded = {output, checksum_output}
    paths = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.resolve() in excluded:
            continue
        relative = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in args.exclude):
            continue
        paths.append(path)
    paths.sort()
    files = []
    checksum_lines = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest = sha256(path)
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "suffix": path.suffix.lower(),
            }
        )
        checksum_lines.append(f"{digest}  {relative}")
    document = {
        "schema_version": 1,
        "root": str(root),
        "generated_at": datetime.now().astimezone().isoformat(),
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checksum_output.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({key: document[key] for key in ("root", "file_count", "total_size_bytes")}, indent=2))


if __name__ == "__main__":
    main()
