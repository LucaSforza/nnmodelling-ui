from __future__ import annotations

import argparse
import email
import re
import subprocess
import sys
import zipfile
from pathlib import Path

EXPECTED_DISTRIBUTION = "nnm-tiny-decoder-llm"


def _validate_wheel(path: Path) -> None:
    if not path.is_file() or path.suffix != ".whl":
        raise ValueError(f"wheel file does not exist: {path}")
    with zipfile.ZipFile(path) as archive:
        metadata_files = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise ValueError("wheel must contain exactly one dist-info/METADATA")
        metadata = email.message_from_bytes(archive.read(metadata_files[0]))
        distribution = re.sub(r"[-_.]+", "-", metadata.get("Name", "")).lower()
        if distribution != EXPECTED_DISTRIBUTION:
            raise ValueError(f"expected {EXPECTED_DISTRIBUTION}, found {metadata.get('Name')!r}")


def install(path: Path) -> None:
    wheel = path.expanduser().resolve()
    _validate_wheel(wheel)
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "--reinstall", "--no-deps", str(wheel)],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Install a selected Tiny Shakespeare model wheel.")
    parser.add_argument("wheel", type=Path, help="local wheel artifact to install")
    args = parser.parse_args()
    install(args.wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
