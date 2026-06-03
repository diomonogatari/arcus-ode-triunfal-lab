import subprocess
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
RUST_DIR = ROOT / "rustscan"
DATA_DIR = RUST_DIR / "data"
RAW6 = DATA_DIR / "raw6.bin"
RAW7 = DATA_DIR / "raw7.bin"
BIN = RUST_DIR / "target" / "release" / "arcus_scan"
ODE = ROOT / "ode.pt"


def write_raw():
    obj = torch.load(str(ODE), map_location="cpu", weights_only=False)
    raw6 = obj["model"]["transformer.h.6.mlp.c_proj.weight"].numpy().tobytes()
    raw7 = obj["model"]["transformer.h.7.mlp.c_proj.weight"].numpy().tobytes()
    RAW6.write_bytes(raw6)
    RAW7.write_bytes(raw7)


import hashlib


RAW_HASH = DATA_DIR / ".ode.sha256"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def ensure_raw():
    current_hash = sha256_file(ODE)

    if RAW_HASH.exists():
        saved = RAW_HASH.read_text().strip()

        if (
            saved == current_hash
            and RAW6.exists()
            and RAW7.exists()
        ):
            return

    print("extracting raw tensors...")

    write_raw()

    RAW_HASH.write_text(current_hash)


def ensure_binary():
    src = RUST_DIR / "src" / "main.rs"
    cargo = RUST_DIR / "Cargo.toml"
    if BIN.exists():
        bin_mtime = BIN.stat().st_mtime
        if src.exists() and cargo.exists():
            if src.stat().st_mtime <= bin_mtime and cargo.stat().st_mtime <= bin_mtime:
                return
    subprocess.run(
    [
        "cargo",
        "build",
        "--release",
        "--quiet",
    ],
        cwd=RUST_DIR,
        check=True,
    )


def run_scan():
    ensure_raw()
    ensure_binary()

    start = time.perf_counter()

    subprocess.run([str(BIN), str(RAW6), str(RAW7)], check=True)

    elapsed = time.perf_counter() - start
    print(f"\nscan completed in {elapsed:.2f}s")


if __name__ == "__main__":
    run_scan()
