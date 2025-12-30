#!/usr/bin/env python3
"""Download face-swap models (inswapper and GFPGAN).

Usage:
    python download_faceswap_models.py [output_dir]

Models downloaded:
    - inswapper_128.onnx (529MB) - InsightFace face swapping model
    - GFPGANv1.4.pth (333MB) - Face enhancement/restoration model

The output directory defaults to ./models if not specified.
Set FACESWAP_MODEL_PATH environment variable to this directory.
"""

import os
import sys
import urllib.request
from pathlib import Path


MODELS = {
    "inswapper_128.onnx": {
        "url": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        "size_mb": 529,
        "description": "InsightFace face swapping model"
    },
    "GFPGANv1.4.pth": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "size_mb": 333,
        "description": "GFPGAN face enhancement model"
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


def download_models(output_dir: str = "./models"):
    """Download all face-swap models."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading face-swap models to: {output_path.absolute()}")
    print("=" * 60)

    for name, info in MODELS.items():
        model_path = output_path / name

        if model_path.exists():
            print(f"\n✓ {name} already exists ({info['size_mb']}MB)")
            continue

        print(f"\nDownloading {name} ({info['size_mb']}MB)...")
        print(f"  {info['description']}")

        try:
            download_with_progress(info["url"], model_path, info["size_mb"])
            print(f"  ✓ Downloaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to download: {e}")
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            sys.exit(1)

    print("\n" + "=" * 60)
    print("All models downloaded successfully!")
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
