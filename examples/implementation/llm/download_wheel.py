from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


JOB_ID = "1e8594bb-3d8c-43e7-ae96-56bc94c461d9"
FILENAME = "nnm_tiny_decoder_llm-0.1.0-py3-none-any.whl"
SHA256 = "154379d2eb73fb9d3d8dc1990b11fd1d0d67bf2d7c9835452501aef1e83ab601"


def download(base_url: str, destination: Path) -> None:
    url = f"{base_url.rstrip('/')}/v1/jobs/{JOB_ID}/wheel"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        with urlopen(url, timeout=30) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != SHA256:
            raise ValueError(f"wheel SHA256 mismatch: expected {SHA256}, received {digest}")
        os.replace(temporary, destination)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify the completed clustered Tiny Decoder LLM wheel.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765", help="local backend URL (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "wheels")
    args = parser.parse_args()
    target = args.output_dir / FILENAME
    try:
        download(args.base_url, target)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
        print(f"wheel download failed: {error}", file=sys.stderr)
        return 1
    print(f"verified {target} (sha256 {SHA256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
