#!/usr/bin/env python3
"""Download face-swap models (inswapper and GFPGAN) with checksum verification.

Usage:
    python download_faceswap_models.py [output_dir]

Models downloaded:
    - inswapper_128.onnx (529MB) - InsightFace face swapping model
    - GFPGANv1.4.pth (333MB) - Face enhancement/restoration model

The output directory defaults to ./models if not specified.
Set FACESWAP_MODEL_PATH environment variable to this directory.

Security: All models are verified with SHA256 checksums to prevent
supply chain attacks and detect corrupted downloads.
"""

import os
import sys
import hashlib
import urllib.request
from pathlib import Path


# SHA256 checksums for model integrity verification
# These checksums were computed from the official model releases
MODELS = {
    "inswapper_128.onnx": {
        "url": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        "size_mb": 529,
        "description": "InsightFace face swapping model",
        "sha256": "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"
    },
    "GFPGANv1.4.pth": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "size_mb": 333,
        "description": "GFPGAN face enhancement model",
        "sha256": "e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad"
    }
}


def download_with_progress(url: str, dest_path: Path, total_mb: int):
    """Download a file with progress indication."""
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            downloaded_mb = downloaded / (1024 * 1024)
            print(f"\r  Progress: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb}MB)", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest_path), reporthook)
    print()  # New line after progress


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        # Read in 8KB chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_checksum(filepath: Path, expected_hash: str) -> bool:
    """
    Verify file checksum matches expected value.

    Args:
        filepath: Path to the file to verify
        expected_hash: Expected SHA256 hash (lowercase hex)

    Returns:
        True if checksum matches, False otherwise
    """
    print(f"  Verifying SHA256 checksum...", end="", flush=True)
    actual_hash = compute_sha256(filepath)
    if actual_hash.lower() == expected_hash.lower():
        print(f" ✓ Valid")
        return True
    else:
        print(f" ✗ MISMATCH!")
        print(f"    Expected: {expected_hash}")
        print(f"    Actual:   {actual_hash}")
        return False


def download_models(output_dir: str = "./models"):
    """Download all face-swap models with checksum verification."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading face-swap models to: {output_path.absolute()}")
    print("=" * 60)

    for name, info in MODELS.items():
        model_path = output_path / name
        expected_hash = info.get("sha256")

        if model_path.exists():
            print(f"\n{name} already exists ({info['size_mb']}MB)")
            # Verify checksum of existing file
            if expected_hash:
                if not verify_checksum(model_path, expected_hash):
                    print(f"  ⚠ Existing file has invalid checksum!")
                    print(f"  Re-downloading {name}...")
                    model_path.unlink()
                else:
                    continue  # File exists and is valid
            else:
                print(f"  ✓ Exists (no checksum available)")
                continue

        print(f"\nDownloading {name} ({info['size_mb']}MB)...")
        print(f"  {info['description']}")

        try:
            download_with_progress(info["url"], model_path, info["size_mb"])

            # Verify checksum after download
            if expected_hash:
                if not verify_checksum(model_path, expected_hash):
                    print(f"  ✗ SECURITY ERROR: Downloaded file has invalid checksum!")
                    print(f"  This could indicate a compromised download source.")
                    model_path.unlink()
                    sys.exit(1)

            print(f"  ✓ Downloaded and verified successfully")
        except Exception as e:
            print(f"  ✗ Failed to download: {e}")
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            sys.exit(1)

    print("\n" + "=" * 60)
    print("All models downloaded and verified successfully!")
    print(f"\nTo enable face-swap in power-node, set:")
    print(f"  export FACESWAP_MODEL_PATH={output_path.absolute()}")
    print("\nOr add to your config.yaml:")
    print(f"  faceswap:")
    print(f"    enabled: true")
    print(f"    model_path: {output_path.absolute()}")


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./models"
    download_models(output_dir)


if __name__ == "__main__":
    main()
