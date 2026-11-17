"""Download a dataset's recordings into data/raw, once, with checksums."""

import hashlib
import urllib.request
from pathlib import Path

from src.paths import RAW


def fetch(recordings, into: Path = RAW) -> None:
    into.mkdir(parents=True, exist_ok=True)
    for recording in recordings:
        path = into / recording.filename
        if not path.exists():
            urllib.request.urlretrieve(recording.url, path)
        print(f"{path.name:<22} {path.stat().st_size / 1e6:5.1f} MB")
    write_checksums(into)


def write_checksums(folder: Path) -> None:
    lines = [f"{sha256(p)}  {p.name}" for p in sorted(folder.glob("*.mat"))]
    (folder / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    from src.acquisition import cwru

    fetch(cwru.RECORDINGS)
