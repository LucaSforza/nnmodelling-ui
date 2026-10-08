from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


HERE = Path(__file__).parent
ARTIFACT = HERE / "artifact.json"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def load_artifact(path: Path = ARTIFACT) -> dict[str, str]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    required = {"job_id", "filename", "distribution", "module", "sha256"}
    if not isinstance(artifact, dict) or set(artifact) != required:
        raise ValueError("artifact.json must contain job_id, filename, distribution, module, and sha256")
    if any(not isinstance(artifact[key], str) or not artifact[key] for key in required):
        raise ValueError("artifact.json fields must be nonempty strings")
    if not re.fullmatch(r"[A-Za-z0-9-]+", artifact["job_id"]):
        raise ValueError("artifact job_id contains unsafe characters")
    if Path(artifact["filename"]).name != artifact["filename"] or not artifact["filename"].endswith(".whl"):
        raise ValueError("artifact filename must be a wheel filename without a path")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", artifact["distribution"]):
        raise ValueError("artifact distribution contains unsafe characters")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", artifact["module"]):
        raise ValueError("artifact module must be a Python import name")
    if not SHA256_PATTERN.fullmatch(artifact["sha256"]):
        raise ValueError("artifact sha256 must be 64 lowercase hexadecimal characters")
    return artifact


def download(base_url: str, destination: Path, artifact: dict[str, str]) -> None:
    url = f"{base_url.rstrip('/')}/v1/jobs/{artifact['job_id']}/wheel"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            with urlopen(url, timeout=30) as response:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
        received = digest.hexdigest()
        if received != artifact["sha256"]:
            raise ValueError(f"wheel SHA256 mismatch: expected {artifact['sha256']}, received {received}")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify the trained VAE wheel.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765", help="local backend URL (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=HERE / "wheels")
    args = parser.parse_args()
    try:
        artifact = load_artifact()
        target = args.output_dir / artifact["filename"]
        download(args.base_url, target, artifact)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"wheel download failed: {error}", file=sys.stderr)
        return 1
    print(f"verified {target} (sha256 {artifact['sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
