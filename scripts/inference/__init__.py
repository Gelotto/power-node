# Power Node Inference Scripts
# This package contains the inference scripts for different model types and modes.
#
# Scripts:
#   - inference_gguf_zimage.py    - Z-Image-Turbo via stable-diffusion.cpp (8-16GB VRAM)
#   - inference_gguf_flux.py      - FLUX.1-schnell via stable-diffusion.cpp (10-16GB VRAM)
#   - inference_pytorch_zimage.py - Z-Image-Turbo via diffusers (14GB+ VRAM, Blackwell)
#   - inference_pytorch_flux.py   - FLUX.1-schnell via diffusers (16GB+ VRAM, Blackwell)
#
# Shared utilities are in shared_utils.py (security constants, download helpers, etc.)
