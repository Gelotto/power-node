"""Shared constants for the faceswap module."""

# Security constant - maximum image dimension to prevent OOM attacks
# Reduced from 4096 to 2048 for better memory safety:
# - Z-Image-Turbo max output is ~2048
# - Face-swap works best on smaller images anyway
# - 2048x2048 = 12MB per image vs 4096x4096 = 48MB
MAX_IMAGE_DIMENSION = 2048
