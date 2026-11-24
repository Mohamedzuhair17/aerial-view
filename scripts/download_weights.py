"""Download explicitly trusted model files into ./weights.

Usage:
  python scripts/download_weights.py changeformer.pt=https://... --sha256 changeformer.pt=<hex>
Only HTTPS URLs and the three known checkpoint filenames are accepted.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ALLOWED = {"changeformer.pt", "effnet_b0_damage.pth"}
MAX_BYTES = 8 * 1024**3
DEST = Path(__file__).resolve().parent.parent / "weights"


def download(name: str, url: str, expected_sha256: str | None = None):
    if name not in ALLOWED:
        raise SystemExit(f"Unsupported checkpoint filename: {name}")
    if not url.lower().startswith("https://"):
        raise SystemExit("Only HTTPS download URLs are accepted")
    DEST.mkdir(exist_ok=True)
    target = DEST / name
    temporary = DEST / (name + ".part")
    digest = hashlib.sha256()
    total = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "AerialBackendModelDownloader/2.0"})
        with urllib.request.urlopen(request, timeout=30) as response, temporary.open("wb") as output:
            if not response.geturl().lower().startswith("https://"):
                raise SystemExit("Download redirected away from HTTPS; refusing the response")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    raise SystemExit("Checkpoint exceeds the 8 GiB download limit")
                digest.update(chunk)
                output.write(chunk)
        actual = digest.hexdigest()
        if expected_sha256 and actual.lower() != expected_sha256.lower():
            raise SystemExit(f"SHA-256 mismatch for {name}: expected {expected_sha256}, got {actual}")
        temporary.replace(target)
        print(f"Downloaded {name} ({total:,} bytes), SHA-256={actual}")
    finally:
        temporary.unlink(missing_ok=True)


def main():
    args = sys.argv[1:]
    checksums = {}
    if "--sha256" in args:
        i = args.index("--sha256")
        pairs = args[i + 1:]
        args = args[:i]
        for item in pairs:
            name, checksum = item.split("=", 1)
            checksums[name] = checksum
    if not args:
        raise SystemExit("Provide at least one filename=https://url argument")
    for arg in args:
        name, url = arg.split("=", 1)
        download(name, url, checksums.get(name))


if __name__ == "__main__":
    main()
