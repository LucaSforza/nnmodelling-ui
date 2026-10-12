from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import zipfile
from urllib.parse import quote
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


JOB_ID = "1e8594bb-3d8c-43e7-ae96-56bc94c461d9"
FILENAME = "nnm_tiny_decoder_llm-0.1.0-py3-none-any.whl"
SHA256 = "154379d2eb73fb9d3d8dc1990b11fd1d0d67bf2d7c9835452501aef1e83ab601"


def _validate_wheel(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise ValueError("downloaded artifact is not a valid wheel archive")
    with zipfile.ZipFile(path) as archive:
        metadata_files = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise ValueError("wheel must contain exactly one dist-info/METADATA")
        metadata = archive.read(metadata_files[0]).decode("utf-8")
        if "Name: nnm_tiny_decoder_llm\n" not in metadata:
            raise ValueError("wheel distribution identity is not nnm_tiny_decoder_llm")


def download(base_url: str, destination: Path, job_id: str = JOB_ID, sha256: str | None = None) -> str:
    if not job_id or "/" in job_id or "\\" in job_id:
        raise ValueError("job ID must be a nonempty path segment")
    url = f"{base_url.rstrip('/')}/v1/jobs/{quote(job_id, safe='')}/wheel"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        with urlopen(url, timeout=30) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if sha256 is not None and digest.lower() != sha256.lower():
            raise ValueError(f"wheel SHA256 mismatch: expected {sha256}, received {digest}")
        _validate_wheel(temporary)
        os.replace(temporary, destination)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        temporary.unlink(missing_ok=True)
        raise
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a completed Tiny Shakespeare model wheel.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765", help="local backend URL (default: %(default)s)")
    parser.add_argument("--job-id", help="completed backend job ID (defaults to the recorded historical job)")
    parser.add_argument("--sha256", help="optional expected SHA256 for the selected job artifact")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "wheels")
    args = parser.parse_args()
    target = args.output_dir / FILENAME
    job_id = args.job_id or JOB_ID
    expected_sha256 = args.sha256 if args.sha256 is not None else (SHA256 if args.job_id is None else None)
    try:
        digest = download(args.base_url, target, job_id, expected_sha256)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
        print(f"wheel download failed: {error}", file=sys.stderr)
        return 1
    print(f"downloaded {target} (sha256 {digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
