#!/usr/bin/env python3
"""
Shared utilities for Power Node inference scripts.
Contains security constants and common functions used by both GGUF and PyTorch modes.
"""

from urllib.parse import urlparse
import subprocess

# Security constants - allowed hosts for face-swap image downloads
ALLOWED_HOSTS = [
    # Picshapes infrastructure
    'storage.picshapes.com',
    'api.gelotto.io',
    'api.picshapes.com',
    # Meme templates
    'i.imgflip.com',
    'imgflip.com',
    # Image hosting
    'i.imgur.com',
    'imgur.com',
    # Reddit
    'i.redd.it',
    'preview.redd.it',
    # GIFs
    'media.giphy.com',
    'giphy.com',
    'media.tenor.com',
    'tenor.com',
    # Social media CDNs
    'pbs.twimg.com',
    'cdn.discordapp.com',
    # Stock photos
    'images.unsplash.com',
]
MAX_DOWNLOAD_SIZE = 20 * 1024 * 1024  # 20MB
FACESWAP_IDLE_TIMEOUT = 300  # 5 minutes


def validate_image_url(url):
    """Validate URL is from allowed hosts (SSRF protection)."""
    parsed = urlparse(url)
    if parsed.scheme != 'https':
        raise ValueError(f"Only HTTPS URLs allowed, got: {parsed.scheme}")
    if parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"URL host not allowed: {parsed.hostname}")
    return url


def download_with_limit(url, max_bytes=MAX_DOWNLOAD_SIZE, timeout=30):
    """Download URL content with size limit to prevent DoS."""
    import requests

    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()

    # Check Content-Length header if available
    content_length = resp.headers.get('content-length')
    if content_length and int(content_length) > max_bytes:
        raise ValueError(f"File too large: {content_length} bytes (max: {max_bytes})")

    # Stream download with size check
    chunks = []
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=8192):
        downloaded += len(chunk)
        if downloaded > max_bytes:
            raise ValueError(f"Download exceeded size limit: {downloaded} bytes (max: {max_bytes})")
        chunks.append(chunk)

    return b''.join(chunks)


def detect_vram_gb():
    """Detect GPU VRAM in GB."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split('\n')[0]) // 1024
    except:
        pass
    return 8
