from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


JOB_ID = "a64143c8-07c2-40e7-8f89-37c33d64970b"
FILENAME = "nnmodel_a64143c8_07c2_40e7_8f89_37c33d64970b-0.1.0-py3-none-any.whl"
SHA256 = "efd2e36435cce8ba5e22121925cbdff1af13773f6b5494d912fdbd9a260de33d"


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
    parser = argparse.ArgumentParser(description="Download and verify the completed full-training model wheel.")
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
