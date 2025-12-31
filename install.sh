#!/bin/bash
#
# Power Node Installation Script
# https://github.com/Gelotto/power-node
#
set -e

# Parse command line arguments
SKIP_VIDEO=false
SKIP_FACESWAP=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --no-video) SKIP_VIDEO=true ;;
        --no-faceswap) SKIP_FACESWAP=true ;;
        --minimal) SKIP_VIDEO=true; SKIP_FACESWAP=true ;;
        --help|-h)
            echo "Usage: install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-video      Skip video model (auto-downloaded for 12GB+ VRAM PyTorch GPUs)"
            echo "  --no-faceswap   Skip face-swap model (auto-downloaded for 6GB+ VRAM)"
            echo "  --minimal       Skip all optional models (video + face-swap)"
            echo "  --help, -h      Show this help message"
            exit 0
            ;;
    esac
    shift
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

INSTALL_DIR="${POWER_NODE_DIR:-$HOME/.power-node}"
API_URL="${API_URL:-https://api.gelotto.io}"
HF_BASE="https://huggingface.co"
GITHUB_REPO="Gelotto/power-node"

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              Power Node Installation Script                   ║"
echo "║         Gelotto Distributed GPU Compute Network               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# =============================================================================
# System Requirements
# =============================================================================
echo -e "${YELLOW}[1/7] Checking system requirements...${NC}"

if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. Please install NVIDIA drivers first.${NC}"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 not found.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${RED}ERROR: Python 3.10+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

if ! command -v curl &> /dev/null; then
    echo -e "${RED}ERROR: curl not found.${NC}"
    exit 1
fi

# Check for ffmpeg (required for video encoding)
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${YELLOW}  ffmpeg not found. Installing...${NC}"
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y ffmpeg
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y ffmpeg
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm ffmpeg
    elif command -v zypper &> /dev/null; then
        sudo zypper install -y ffmpeg
    else
        echo -e "${YELLOW}  Warning: Could not install ffmpeg automatically.${NC}"
        echo "  Video generation will require ffmpeg. Install manually."
    fi

    if command -v ffmpeg &> /dev/null; then
        echo -e "${GREEN}  ✓ ffmpeg installed${NC}"
    fi
fi

# Check if venv module is available, auto-install if missing
if ! python3 -c "import venv" 2>/dev/null; then
    echo -e "${YELLOW}  Python venv module not found. Installing...${NC}"

    # Detect package manager and install
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y python3-venv
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y python3-venv
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm python
    elif command -v zypper &> /dev/null; then
        sudo zypper install -y python3-venv
    else
        echo -e "${RED}ERROR: Could not auto-install python3-venv.${NC}"
        echo "Please install it manually and re-run this script."
        exit 1
    fi

    # Verify it worked
    if ! python3 -c "import venv" 2>/dev/null; then
        echo -e "${RED}ERROR: Failed to install python3-venv.${NC}"
        exit 1
    fi
    echo -e "${GREEN}  ✓ python3-venv installed${NC}"
fi

echo -e "${GREEN}  ✓ All system requirements met${NC}"

# =============================================================================
# GPU Detection
# =============================================================================
echo -e "${YELLOW}[2/7] Detecting GPU...${NC}"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
VRAM_GB=$(( (VRAM_MB + 512) / 1024 ))
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')

echo -e "  GPU: ${GREEN}$GPU_NAME${NC}"
echo -e "  VRAM: ${GREEN}${VRAM_GB}GB${NC}"
echo -e "  Compute Capability: ${GREEN}${COMPUTE_CAP:0:1}.${COMPUTE_CAP:1}${NC}"

# =============================================================================
# Validate GPU
# =============================================================================
echo -e "${YELLOW}[3/7] Validating GPU requirements...${NC}"

if [ $VRAM_GB -lt 8 ]; then
    echo -e "${RED}ERROR: Minimum 8GB VRAM required. Your GPU has ${VRAM_GB}GB.${NC}"
    exit 1
fi

IS_BLACKWELL=false
if [ "$COMPUTE_CAP" -ge 120 ] 2>/dev/null; then
    IS_BLACKWELL=true
fi

if [ "$IS_BLACKWELL" = true ] && [ $VRAM_GB -lt 14 ]; then
    echo -e "${RED}ERROR: Blackwell GPUs require 14GB+ VRAM for PyTorch backend.${NC}"
    exit 1
fi

if [ "$IS_BLACKWELL" = true ]; then
    SERVICE_MODE="pytorch"
    echo -e "  Mode: ${BLUE}PyTorch${NC} (Blackwell GPU)"
else
    SERVICE_MODE="gguf"
    echo -e "  Mode: ${BLUE}GGUF${NC} (stable-diffusion.cpp)"
fi

echo -e "${GREEN}  ✓ GPU validated${NC}"

# =============================================================================
# Create Directory Structure
# =============================================================================
echo -e "${YELLOW}[4/7] Creating directory structure...${NC}"

mkdir -p "$INSTALL_DIR"/{bin,config,logs,scripts,models/{diffusion,vae,text_encoder}}

echo -e "  Install directory: ${GREEN}$INSTALL_DIR${NC}"

# =============================================================================
# Download Binary or Build
# =============================================================================
echo -e "${YELLOW}[5/7] Installing Power Node binary...${NC}"

ARCH=$(uname -m)
OS=$(uname -s | tr '[:upper:]' '[:lower:]')

# Normalize architecture names
if [ "$ARCH" = "aarch64" ]; then
    ARCH="arm64"
fi
# x86_64 stays as x86_64

# Try to download pre-built binary
BINARY_URL="https://github.com/$GITHUB_REPO/releases/latest/download/power-node-${OS}-${ARCH}"
echo "  Binary URL: $BINARY_URL"

# Download binary directly (GitHub releases use redirects)
echo "  Downloading pre-built binary..."
if curl -fsSL "$BINARY_URL" -o "$INSTALL_DIR/bin/power-node"; then
    chmod +x "$INSTALL_DIR/bin/power-node"
    echo -e "${GREEN}  ✓ Binary downloaded${NC}"
else
    echo -e "${YELLOW}  Download failed, building from source...${NC}"
    rm -f "$INSTALL_DIR/bin/power-node"
    BUILD_FROM_SOURCE=true
fi

if [ "$BUILD_FROM_SOURCE" = true ]; then
    echo "  Pre-built binary not available, building from source..."

    if ! command -v go &> /dev/null; then
        echo -e "${RED}ERROR: Go not found. Please install Go 1.21+${NC}"
        echo "  https://go.dev/doc/install"
        exit 1
    fi

    if ! command -v git &> /dev/null; then
        echo -e "${RED}ERROR: git not found. Required for building from source.${NC}"
        exit 1
    fi

    TEMP_DIR=$(mktemp -d)
    trap "rm -rf '$TEMP_DIR'" EXIT
    git clone --quiet "https://github.com/$GITHUB_REPO.git" "$TEMP_DIR/power-node"
    (cd "$TEMP_DIR/power-node" && go build -o "$INSTALL_DIR/bin/power-node" ./cmd/power-node)
    rm -rf "$TEMP_DIR"
    trap - EXIT
fi

# Verify binary installation
if [ ! -f "$INSTALL_DIR/bin/power-node" ] || [ ! -x "$INSTALL_DIR/bin/power-node" ]; then
    echo -e "${RED}ERROR: Binary not found or not executable${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Binary installed${NC}"

# =============================================================================
# Setup Python Environment
# =============================================================================
echo -e "${YELLOW}[6/7] Setting up Python environment...${NC}"

# Validate venv completeness (not just existence)
VENV_VALID=true
if [ ! -f "$INSTALL_DIR/venv/bin/activate" ] || \
   [ ! -f "$INSTALL_DIR/venv/bin/pip" ] || \
   [ ! -f "$INSTALL_DIR/venv/bin/python3" ]; then
    VENV_VALID=false
fi

if [ "$VENV_VALID" = false ]; then
    echo "  Creating Python virtual environment..."
    rm -rf "$INSTALL_DIR/venv"  # Remove incomplete venv

    if ! python3 -m venv "$INSTALL_DIR/venv"; then
        echo -e "${RED}ERROR: Failed to create Python virtual environment.${NC}"
        echo -e "Try: ${YELLOW}sudo apt install python3-venv${NC}"
        exit 1
    fi

    # Verify creation was successful
    if [ ! -f "$INSTALL_DIR/venv/bin/activate" ]; then
        echo -e "${RED}ERROR: Virtual environment creation incomplete.${NC}"
        echo "The venv module may be broken. Try reinstalling python3-venv."
        exit 1
    fi
else
    echo "  Using existing Python virtual environment"
fi

source "$INSTALL_DIR/venv/bin/activate"

if ! pip install --upgrade pip --quiet; then
    echo -e "${RED}ERROR: Failed to upgrade pip${NC}"
    exit 1
fi

if [ "$SERVICE_MODE" = "gguf" ]; then
    # Auto-install build tools if missing
    if ! command -v gcc &> /dev/null || ! command -v g++ &> /dev/null; then
        echo -e "${YELLOW}  Build tools (gcc/g++) not found. Installing...${NC}"
        if command -v apt-get &> /dev/null; then
            sudo apt-get update -qq && sudo apt-get install -y build-essential cmake
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y gcc gcc-c++ cmake
        elif command -v pacman &> /dev/null; then
            sudo pacman -S --noconfirm base-devel cmake
        else
            echo -e "${RED}ERROR: Could not auto-install build tools.${NC}"
            echo "Please install gcc, g++, and cmake manually."
            exit 1
        fi
        echo -e "${GREEN}  ✓ Build tools installed${NC}"
    fi

    # Auto-install CUDA toolkit if nvcc missing (for CUDA acceleration)
    if ! command -v nvcc &> /dev/null; then
        echo -e "${YELLOW}  CUDA toolkit (nvcc) not found. Installing...${NC}"
        CUDA_INSTALLED=false
        if command -v apt-get &> /dev/null; then
            if sudo apt-get update -qq && sudo apt-get install -y nvidia-cuda-toolkit; then
                CUDA_INSTALLED=true
            fi
        elif command -v dnf &> /dev/null; then
            if sudo dnf install -y cuda-toolkit; then
                CUDA_INSTALLED=true
            fi
        fi

        # Reload PATH to pick up nvcc
        export PATH="/usr/local/cuda/bin:$PATH"

        if command -v nvcc &> /dev/null; then
            echo -e "${GREEN}  ✓ CUDA toolkit installed${NC}"
        elif [ "$CUDA_INSTALLED" = true ]; then
            echo -e "${YELLOW}  Warning: CUDA installed but nvcc not in PATH${NC}"
            echo "  You may need to add /usr/local/cuda/bin to your PATH"
        else
            echo -e "${YELLOW}  Warning: Could not install CUDA toolkit automatically${NC}"
            echo "  Install manually: sudo apt install nvidia-cuda-toolkit"
            echo "  Continuing without CUDA (will use CPU-only mode)"
        fi
    fi

    echo "  Installing stable-diffusion-cpp-python..."
    echo "  (This may take several minutes for CUDA compilation)"

    if command -v nvcc &> /dev/null; then
        echo "  CUDA detected: $(nvcc --version | grep release | awk '{print $6}')"
        if ! CMAKE_ARGS="-DSD_CUDA=ON" pip install 'stable-diffusion-cpp-python>=0.4.0'; then
            echo -e "${RED}ERROR: Failed to compile stable-diffusion-cpp-python${NC}"
            echo "  This usually means CUDA headers are missing or incompatible."
            exit 1
        fi
    else
        echo -e "${YELLOW}  Warning: nvcc not found, installing CPU-only (slower inference)${NC}"
        if ! pip install 'stable-diffusion-cpp-python>=0.4.0'; then
            echo -e "${RED}ERROR: Failed to install stable-diffusion-cpp-python${NC}"
            exit 1
        fi
    fi

    pip install pillow --quiet || { echo -e "${RED}ERROR: Failed to install pillow${NC}"; exit 1; }

    # Verify installation
    if ! python3 -c "from stable_diffusion_cpp import StableDiffusion" 2>/dev/null; then
        echo -e "${RED}ERROR: stable-diffusion-cpp-python installed but cannot be imported.${NC}"
        exit 1
    fi
    echo -e "${GREEN}  ✓ ML packages installed${NC}"
else
    echo "  Installing PyTorch and dependencies for Blackwell GPU..."
    # Blackwell GPUs (sm_120) require PyTorch nightly with CUDA 12.8+
    if ! pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --quiet; then
        echo -e "${RED}ERROR: Failed to install PyTorch${NC}"
        exit 1
    fi
    # Pin diffusers>=0.32.0 for Wan video model support
    # imageio + imageio-ffmpeg required for video export (export_to_video)
    if ! pip install transformers 'diffusers>=0.32.0' safetensors accelerate tqdm pillow ftfy imageio imageio-ffmpeg --quiet; then
        echo -e "${RED}ERROR: Failed to install ML dependencies${NC}"
        exit 1
    fi
    echo -e "${GREEN}  ✓ ML packages installed${NC}"
fi

deactivate
echo -e "${GREEN}  ✓ Python environment ready${NC}"

# =============================================================================
# Install Face-Swap Dependencies
# =============================================================================
echo -e "\n${YELLOW}[6.5/7] Installing face-swap dependencies...${NC}"

# Install Python development headers (required for insightface Cython compilation)
# This provides Python.h which is needed to build insightface's C extensions
if command -v dpkg &> /dev/null; then
    if ! dpkg -s python3-dev >/dev/null 2>&1; then
        echo "  Installing python3-dev (required for insightface)..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get install -y python3-dev >/dev/null 2>&1 || echo "  [Could not auto-install python3-dev]"
        fi
    fi
fi

source "$INSTALL_DIR/venv/bin/activate"

# Core dependencies (required)
echo "  Installing core face-swap packages..."
if pip install insightface==0.7.3 opencv-python-headless requests imageio imageio-ffmpeg --quiet 2>&1; then
    echo -e "  ${GREEN}✓${NC} Core packages installed"
else
    echo -e "  ${YELLOW}!${NC} Core packages failed - face-swap may not work"
    echo "  Hint: If you see 'Python.h not found', run: sudo apt-get install python3-dev"
fi

# ONNX runtime (try GPU first, fallback to CPU)
echo "  Installing ONNX runtime..."
if pip install onnxruntime-gpu --quiet 2>&1; then
    echo -e "  ${GREEN}✓${NC} ONNX Runtime (GPU)"
elif pip install onnxruntime --quiet 2>&1; then
    echo -e "  ${YELLOW}✓${NC} ONNX Runtime (CPU only - slower)"
else
    echo -e "  ${YELLOW}!${NC} ONNX Runtime failed - face-swap won't work"
fi

# GFPGAN for face enhancement (optional, may fail due to complex deps)
echo "  Installing GFPGAN (face enhancement)..."

# Check Python version - 3.13+ has exec() compatibility issue with basicsr
# See: https://github.com/TencentARC/GFPGAN/pull/619
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
if [ "$PYTHON_MINOR" -ge 13 ]; then
    echo "  [Python 3.13+ detected - using compatible basicsr fork]"
    # Install basicsr from Disty0's fork with Python 3.13 fix (master branch)
    # Then install gfpgan without deps since basicsr is already installed
    if pip install git+https://github.com/Disty0/BasicSR.git --quiet 2>&1 && \
       pip install gfpgan --no-deps --quiet 2>&1; then
        echo -e "  ${GREEN}✓${NC} GFPGAN installed (Python 3.13 compatible)"
    else
        echo -e "  ${YELLOW}!${NC} GFPGAN failed - enhancement disabled (swap still works)"
    fi
else
    # Python 3.12 and earlier - normal install works fine
    if pip install gfpgan --quiet 2>&1; then
        echo -e "  ${GREEN}✓${NC} GFPGAN installed"
    else
        echo -e "  ${YELLOW}!${NC} GFPGAN failed - enhancement disabled (swap still works)"
    fi
fi

deactivate
echo -e "${GREEN}  ✓ Face-swap dependencies ready${NC}"

# =============================================================================
# Download Models
# =============================================================================
echo -e "${YELLOW}[7/7] Downloading models...${NC}"

# Check available disk space
AVAILABLE_KB=$(df -k "$INSTALL_DIR" | tail -1 | awk '{print $4}')
AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))

if [ "$SERVICE_MODE" = "gguf" ]; then
    REQUIRED_GB=12  # ~9GB models + buffer
else
    REQUIRED_GB=35  # ~31GB PyTorch models + buffer
fi

if [ "$AVAILABLE_GB" -lt "$REQUIRED_GB" ]; then
    echo -e "${RED}ERROR: Not enough disk space.${NC}"
    echo "  Required: ${REQUIRED_GB}GB, Available: ${AVAILABLE_GB}GB"
    echo "  Free up space in $(dirname $INSTALL_DIR) and try again."
    exit 1
fi

download_file() {
    local url="$1"
    local dest="$2"
    local name="$3"

    if [ -f "$dest" ]; then
        echo -e "  ${GREEN}✓${NC} $name (cached)"
        return 0
    fi

    echo "  Downloading $name..."
    if ! curl -L "$url" -o "$dest" --progress-bar --fail; then
        echo -e "${RED}ERROR: Failed to download $name${NC}"
        echo "  URL: $url"
        rm -f "$dest"
        exit 1
    fi

    # Verify file was downloaded
    if [ ! -s "$dest" ]; then
        echo -e "${RED}ERROR: Downloaded file is empty: $name${NC}"
        rm -f "$dest"
        exit 1
    fi
}

# Check disk space and warn if below threshold
# Returns 0 if OK to proceed, 1 if should skip
check_disk_space_warning() {
    local required_gb=$1
    local label=$2
    local available_kb=$(df -k "$INSTALL_DIR" | tail -1 | awk '{print $4}')
    local available_gb=$((available_kb / 1024 / 1024))

    if [ "$available_gb" -lt "$required_gb" ]; then
        echo -e "${YELLOW}  Warning: Only ${available_gb}GB free disk space${NC}"
        echo -e "  ${label} recommends ${required_gb}GB free"
        # Try to prompt user in interactive mode
        if read -p "  Continue anyway? (y/N) " -r < /dev/tty 2>/dev/null; then
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                return 0  # User wants to proceed
            fi
            return 1  # User declined
        fi
        # Non-interactive mode - skip by default
        echo "  [Non-interactive mode - skipping due to low disk space]"
        return 1
    fi
    return 0  # Enough disk space
}

if [ "$SERVICE_MODE" = "gguf" ]; then
    if [ $VRAM_GB -lt 10 ]; then
        QUANT="Q4_0"
    else
        QUANT="Q8_0"
    fi

    download_file \
        "$HF_BASE/leejet/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-${QUANT}.gguf" \
        "$INSTALL_DIR/models/diffusion/z_image_turbo-${QUANT}.gguf" \
        "Diffusion model (${QUANT})"

    download_file \
        "$HF_BASE/ffxvs/vae-flux/resolve/main/ae.safetensors" \
        "$INSTALL_DIR/models/vae/ae.safetensors" \
        "VAE"

    download_file \
        "$HF_BASE/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
        "$INSTALL_DIR/models/text_encoder/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
        "Text encoder"

    DIFFUSION_PATH="$INSTALL_DIR/models/diffusion/z_image_turbo-${QUANT}.gguf"
    VAE_PATH="$INSTALL_DIR/models/vae/ae.safetensors"
    TEXT_ENCODER_PATH="$INSTALL_DIR/models/text_encoder/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
else
    # PyTorch mode - download full model via huggingface-cli
    echo "  Downloading Z-Image-Turbo model (PyTorch, ~31GB)..."
    echo "  This may take a while on slower connections..."

    MODEL_DIR="$INSTALL_DIR/models/z-image-turbo"
    if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/model_index.json" ]; then
        echo -e "  ${GREEN}✓${NC} Z-Image-Turbo (cached)"
    else
        source "$INSTALL_DIR/venv/bin/activate"
        if ! pip install huggingface_hub --quiet; then
            echo -e "${RED}ERROR: Failed to install huggingface_hub${NC}"
            exit 1
        fi
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Tongyi-MAI/Z-Image-Turbo',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False
)
"
        if [ $? -ne 0 ]; then
            echo -e "${RED}ERROR: Failed to download Z-Image-Turbo model${NC}"
            exit 1
        fi
        deactivate
    fi

    # Also need VAE for PyTorch (from public Comfy-Org repo)
    download_file \
        "$HF_BASE/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors" \
        "$INSTALL_DIR/models/vae/ae.safetensors" \
        "VAE"

    # =============================================================================
    # Video Generation Support (Auto-detected based on GPU capability)
    # =============================================================================
    WAN_MODEL_DIR="$INSTALL_DIR/models/Wan2.1-T2V-1.3B-Diffusers"

    # Check if system is video-capable: PyTorch mode (already true here) + 12GB+ VRAM
    VIDEO_CAPABLE=false
    if [ "$VRAM_GB" -ge 12 ]; then
        VIDEO_CAPABLE=true
    fi

    if [ "$VIDEO_CAPABLE" = true ] && [ "$SKIP_VIDEO" != true ]; then
        echo ""
        echo -e "${CYAN}Video Generation Support:${NC}"
        echo "  Your GPU supports video generation (${VRAM_GB}GB VRAM, PyTorch mode)"

        # Check if already cached
        if [ -d "$WAN_MODEL_DIR" ] && [ -f "$WAN_MODEL_DIR/model_index.json" ]; then
            echo -e "  ${GREEN}✓${NC} Wan2.1-T2V-1.3B-Diffusers (cached)"
        else
            # Check disk space (recommend 50GB free for video model)
            if check_disk_space_warning 50 "Video model (~29GB)"; then
                echo "  Auto-downloading Wan2.1 video model (~29GB)..."
                echo "  This may take a while depending on your internet connection."
                source "$INSTALL_DIR/venv/bin/activate"
                if ! pip install huggingface_hub --quiet 2>/dev/null; then
                    echo -e "${YELLOW}  Warning: Failed to install huggingface_hub${NC}"
                fi
                python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
    local_dir='$WAN_MODEL_DIR',
    local_dir_use_symlinks=False
)
"
                DOWNLOAD_EXIT_CODE=$?

                # Validate the download was successful
                if [ $DOWNLOAD_EXIT_CODE -eq 0 ] && [ -f "$WAN_MODEL_DIR/model_index.json" ]; then
                    echo -e "  ${GREEN}✓${NC} Wan2.1-Diffusers video model installed successfully"
                else
                    echo -e "${RED}  ✗ Wan2.1 model download incomplete or failed${NC}"
                    if [ ! -f "$WAN_MODEL_DIR/model_index.json" ]; then
                        echo -e "${YELLOW}    Missing: model_index.json${NC}"
                    fi
                    echo "  Video generation will not be available."
                    echo "  You can try downloading manually:"
                    echo "    huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir $WAN_MODEL_DIR"
                fi
                deactivate
            else
                echo "  Skipping video model (disk space check)"
            fi
        fi
    elif [ "$SKIP_VIDEO" = true ]; then
        echo ""
        echo -e "  Video: ${YELLOW}Skipped${NC} (--no-video flag)"
    elif [ "$VRAM_GB" -lt 12 ]; then
        echo ""
        echo -e "  Video: ${YELLOW}Not available${NC} (${VRAM_GB}GB VRAM < 12GB required)"
    fi
fi

echo -e "${GREEN}  ✓ Models ready${NC}"

# =============================================================================
# Create Inference Script
# =============================================================================
if [ "$SERVICE_MODE" = "gguf" ]; then
cat > "$INSTALL_DIR/scripts/inference.py" << 'INFERENCE_EOF'
#!/usr/bin/env python3
"""
Power Node Inference Service (GGUF/stable-diffusion.cpp)
"""

import sys
import json
import base64
import io
import os
import argparse
import random
import time
import threading
from urllib.parse import urlparse

# Security constants - allowed hosts for face-swap image downloads
# storage.picshapes.com: S3 storage where user uploads go
# api.gelotto.io: Primary worker API domain
# api.picshapes.com: Legacy API domain (for backwards compatibility)
ALLOWED_HOSTS = ['storage.picshapes.com', 'api.gelotto.io', 'api.picshapes.com']
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
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split('\n')[0]) // 1024
    except:
        pass
    return 8


class InferenceService:
    def __init__(self, diffusion_path, vae_path, text_encoder_path, vram_gb=None):
        self.diffusion_path = diffusion_path
        self.vae_path = vae_path
        self.text_encoder_path = text_encoder_path
        self.vram_gb = vram_gb or detect_vram_gb()
        self.sd = None
        # Face-swap model lifecycle tracking
        self._face_swapper = None
        self._faceswap_funcs = None
        self._last_faceswap_time = 0
        self._faceswap_lock = threading.Lock()  # Protects face-swap model access
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Start background thread to unload idle face-swap models."""
        def cleanup_loop():
            while True:
                time.sleep(60)  # Check every minute
                with self._faceswap_lock:
                    if self._face_swapper and time.time() - self._last_faceswap_time > FACESWAP_IDLE_TIMEOUT:
                        sys.stderr.write("Unloading idle face-swap models (5 min timeout)...\n")
                        sys.stderr.flush()
                        self._face_swapper = None
                        self._faceswap_funcs = None
                        import gc
                        gc.collect()

        t = threading.Thread(target=cleanup_loop, daemon=True)
        t.start()

    def initialize(self):
        from stable_diffusion_cpp import StableDiffusion

        sys.stderr.write("=== Power Node Inference Service (GGUF) ===\n")
        sys.stderr.write(f"VRAM: {self.vram_gb}GB\n")
        sys.stderr.flush()

        for path, name in [(self.diffusion_path, "Diffusion"),
                           (self.vae_path, "VAE"),
                           (self.text_encoder_path, "Text encoder")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found: {path}")

        offload = self.vram_gb < 10
        keep_vae_cpu = self.vram_gb < 12
        use_flash = self.vram_gb >= 8

        sys.stderr.write("Loading model...\n")
        sys.stderr.flush()

        self.sd = StableDiffusion(
            diffusion_model_path=self.diffusion_path,
            vae_path=self.vae_path,
            llm_path=self.text_encoder_path,
            vae_decode_only=True,
            n_threads=-1,
            wtype="default",
            rng_type="cuda",
            offload_params_to_cpu=offload,
            keep_clip_on_cpu=offload,
            keep_control_net_on_cpu=offload,
            keep_vae_on_cpu=keep_vae_cpu,
            diffusion_flash_attn=use_flash,
            verbose=True,
        )

        sys.stderr.write("Ready.\n")
        sys.stderr.flush()

    def generate(self, prompt, width=1024, height=1024, steps=8, seed=-1):
        sys.stderr.write(f"Generating: {prompt[:50]}...\n")
        sys.stderr.flush()

        if seed == -1:
            seed = random.randint(0, 2**31 - 1)

        # Z-Image-Turbo with stable-diffusion.cpp uses cfg_scale=1.0
        # (Note: HuggingFace diffusers uses guidance_scale=0.0, but sd.cpp is different!)
        images = self.sd.generate_image(
            prompt=prompt,
            negative_prompt="",  # Z-Image-Turbo doesn't use negative prompts
            width=width,
            height=height,
            cfg_scale=1.0,       # stable-diffusion.cpp uses 1.0 for Z-Image-Turbo
            guidance=1.0,        # stable-diffusion.cpp uses 1.0
            sample_steps=steps,
            seed=seed,
            batch_count=1,
            vae_tiling=True,  # Always enable - Z-Image VAE needs ~27GB for 1024x1024 without tiling
        )

        if not images:
            raise RuntimeError("No image generated")

        buf = io.BytesIO()
        images[0].save(buf, format='PNG')
        return {"image_data": base64.b64encode(buf.getvalue()).decode(), "format": "png"}

    def generate_video(self, prompt, width=832, height=480, duration_seconds=5, fps=24, total_frames=120, seed=-1, negative_prompt=None):
        """Generate a video using Wan2GP (if available)"""
        # Video generation requires Wan2GP model which is not included in GGUF mode
        # Workers need to install video support separately
        raise NotImplementedError(
            "Video generation not available in GGUF mode. "
            "Video requires Wan2GP model installation. "
            "Please use PyTorch mode with video support enabled."
        )

    def face_swap(self, source_image_url, target_image_url, is_gif=False, swap_all_faces=True, enhance=True, max_frames=100):
        """Perform face swap on images or GIFs with security validation."""
        # Validate URLs (SSRF protection) - outside lock
        source_image_url = validate_image_url(source_image_url)
        target_image_url = validate_image_url(target_image_url)

        # Acquire lock for model check/load and timestamp update
        with self._faceswap_lock:
            self._last_faceswap_time = time.time()

            # Lazy load face swapper
            if self._face_swapper is None:
                model_path = os.environ.get("FACESWAP_MODEL_PATH", "")
                if not model_path:
                    raise ValueError("Face swap not available: FACESWAP_MODEL_PATH not set")

                sys.stderr.write(f"Loading face swap models from {model_path}...\n")
                sys.stderr.flush()

                # Add scripts directory to path for faceswap module
                scripts_dir = os.path.dirname(os.path.abspath(__file__))
                if scripts_dir not in sys.path:
                    sys.path.insert(0, scripts_dir)

                from faceswap import FaceSwapper, load_image_from_bytes, image_to_bytes, GifProcessor
                self._face_swapper = FaceSwapper(model_dir=model_path, enable_enhancement=enhance)
                self._faceswap_funcs = {
                    'load_image_from_bytes': load_image_from_bytes,
                    'image_to_bytes': image_to_bytes,
                    'GifProcessor': GifProcessor,
                }
                sys.stderr.write("Face swap models loaded.\n")
                sys.stderr.flush()

            # Get references while holding lock (prevents cleanup during processing)
            swapper = self._face_swapper
            funcs = self._faceswap_funcs

        # Processing happens outside lock to avoid blocking other requests
        sys.stderr.write(f"Face swap: is_gif={is_gif}, swap_all={swap_all_faces}, enhance={enhance}\n")
        sys.stderr.flush()

        # Download source image with size limit (DoS protection)
        source_data = download_with_limit(source_image_url)
        source_img = funcs['load_image_from_bytes'](source_data)

        # Download target with size limit
        target_data = download_with_limit(target_image_url)

        if is_gif:
            # Process GIF frame by frame
            processor = funcs['GifProcessor'](
                swapper,
                enhancer=swapper.enhancer if enhance else None
            )
            result_bytes = processor.process_gif(source_img, target_data, max_frames=max_frames, enhance=enhance)
            return {
                "image_data": base64.b64encode(result_bytes).decode(),
                "format": "gif",
                "frames_swapped": min(max_frames, 100)  # Approximate
            }
        else:
            # Process single image
            target_img = funcs['load_image_from_bytes'](target_data)
            result_img = swapper.swap(source_img, target_img, swap_all=swap_all_faces, enhance=enhance)
            result_bytes = funcs['image_to_bytes'](result_img, ".jpg", quality=95)
            return {
                "image_data": base64.b64encode(result_bytes).decode(),
                "format": "jpeg",
                "frames_swapped": 1
            }

    def handle_request(self, req):
        try:
            method = req.get("method")
            if method == "generate":
                return {"id": req.get("id", 0), "result": self.generate(**req.get("params", {})), "error": None}
            elif method == "generate_video":
                return {"id": req.get("id", 0), "result": self.generate_video(**req.get("params", {})), "error": None}
            elif method == "face_swap":
                return {"id": req.get("id", 0), "result": self.face_swap(**req.get("params", {})), "error": None}
            return {"id": req.get("id", 0), "result": None, "error": f"Unknown method: {method}"}
        except Exception as e:
            return {"id": req.get("id", 0), "result": None, "error": str(e)}

    def run(self):
        sys.stderr.write("Waiting for requests...\n")
        sys.stderr.flush()
        for line in sys.stdin:
            if line.strip():
                print(json.dumps(self.handle_request(json.loads(line))), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diffusion", "-d", required=True)
    p.add_argument("--vae", "-v", required=True)
    p.add_argument("--text-encoder", "-t", required=True)
    p.add_argument("--vram", "-m", type=int)
    args = p.parse_args()

    svc = InferenceService(args.diffusion, args.vae, args.text_encoder, args.vram)
    try:
        svc.initialize()
        svc.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.stderr.write(f"FATAL: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
INFERENCE_EOF
else
# PyTorch inference script
cat > "$INSTALL_DIR/scripts/inference.py" << 'INFERENCE_EOF'
#!/usr/bin/env python3
"""
Power Node Inference Service (PyTorch/Z-Image-Turbo)
For Blackwell GPUs (RTX 50-series) with 14GB+ VRAM
"""

import sys
import json
import base64
import io
import os
import argparse
import random
import gc
import time
import threading
from urllib.parse import urlparse

# Security constants - allowed hosts for face-swap image downloads
# storage.picshapes.com: S3 storage where user uploads go
# api.gelotto.io: Primary worker API domain
# api.picshapes.com: Legacy API domain (for backwards compatibility)
ALLOWED_HOSTS = ['storage.picshapes.com', 'api.gelotto.io', 'api.picshapes.com']
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


class InferenceService:
    def __init__(self, model_path, vae_path=None):
        self.model_path = model_path
        self.vae_path = vae_path
        self.pipe = None
        # Face-swap model lifecycle tracking
        self._face_swapper = None
        self._faceswap_funcs = None
        self._last_faceswap_time = 0
        self._faceswap_lock = threading.Lock()  # Protects face-swap model access
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Start background thread to unload idle face-swap models."""
        def cleanup_loop():
            import torch
            while True:
                time.sleep(60)  # Check every minute
                with self._faceswap_lock:
                    if self._face_swapper and time.time() - self._last_faceswap_time > FACESWAP_IDLE_TIMEOUT:
                        sys.stderr.write("Unloading idle face-swap models (5 min timeout)...\n")
                        sys.stderr.flush()
                        self._face_swapper = None
                        self._faceswap_funcs = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        t = threading.Thread(target=cleanup_loop, daemon=True)
        t.start()

    def initialize(self):
        import torch
        from diffusers import ZImagePipeline

        sys.stderr.write("=== Power Node Inference Service (PyTorch) ===\n")
        sys.stderr.write(f"Model: {self.model_path}\n")
        sys.stderr.flush()

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        sys.stderr.write("Loading model (this may take a minute)...\n")
        sys.stderr.flush()

        # Load Z-Image-Turbo pipeline with optimizations for 16GB VRAM
        self.pipe = ZImagePipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        )

        # Enable memory optimizations
        self.pipe.enable_model_cpu_offload()

        sys.stderr.write("Ready.\n")
        sys.stderr.flush()

    def generate(self, prompt, width=1024, height=1024, steps=8, seed=-1):
        import torch

        # Reload Z-Image model if it was unloaded for video generation
        if self.pipe is None:
            sys.stderr.write("Z-Image model not loaded, reloading...\n")
            sys.stderr.flush()
            self.initialize()

        sys.stderr.write(f"Generating: {prompt[:50]}...\n")
        sys.stderr.flush()

        if seed == -1:
            seed = random.randint(0, 2**31 - 1)

        generator = torch.Generator("cuda").manual_seed(seed)

        # Z-Image-Turbo is a distilled model - CFG is baked in during training
        # guidance_scale MUST be 0.0, negative prompts are not supported
        image = self.pipe(
            prompt=prompt,
            # DO NOT use negative_prompt - Z-Image-Turbo ignores it
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=0.0,  # MUST be 0.0 for turbo models
            generator=generator,
        ).images[0]

        buf = io.BytesIO()
        image.save(buf, format='PNG')

        # Clear CUDA cache
        gc.collect()
        torch.cuda.empty_cache()

        return {"image_data": base64.b64encode(buf.getvalue()).decode(), "format": "png"}

    def generate_video(self, prompt, width=832, height=480, duration_seconds=5, fps=24, total_frames=120, seed=-1, negative_prompt=None, steps=None, guidance_scale=None):
        """Generate a video using Wan2.1 model"""
        import torch
        import tempfile
        import subprocess

        # Default inference steps based on tier (can be overridden by caller)
        # Basic=20, Pro=35, Premium=50 (set by backend based on tier)
        num_inference_steps = steps if steps is not None else 25
        cfg_scale = guidance_scale if guidance_scale is not None else 5.0

        # Default negative prompt for Wan2.1 (improves quality significantly)
        DEFAULT_NEGATIVE_PROMPT = (
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
            "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
            "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
            "misshapen limbs, fused fingers, still picture, messy background, walking backwards"
        )
        video_negative_prompt = negative_prompt if negative_prompt else DEFAULT_NEGATIVE_PROMPT

        sys.stderr.write(f"Generating video: {prompt[:50]}... ({duration_seconds}s @ {fps}fps, {num_inference_steps} steps, cfg={cfg_scale})\n")
        sys.stderr.flush()

        if seed == -1:
            seed = random.randint(0, 2**31 - 1)

        # Check if Wan model is available (must be Diffusers-compatible version)
        wan_model_path = os.environ.get("WAN_MODEL_PATH", os.path.expanduser("~/.power-node/models/Wan2.1-T2V-1.3B-Diffusers"))
        if not os.path.exists(wan_model_path):
            raise FileNotFoundError(
                f"Video model not found at {wan_model_path}. "
                "Please install the Wan2.1-Diffusers model for video generation."
            )

        # Validate model_index.json exists (required for Diffusers pipeline)
        model_index_path = os.path.join(wan_model_path, "model_index.json")
        if not os.path.exists(model_index_path):
            raise FileNotFoundError(
                f"Invalid model: model_index.json not found in {wan_model_path}. "
                "Please download the Diffusers-compatible version: Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
            )

        try:
            from diffusers import WanPipeline, AutoencoderKLWan
            from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        except ImportError:
            raise ImportError("Wan video generation requires diffusers>=0.32.0 with Wan support")

        # CRITICAL: Unload Z-Image model to free VRAM before loading Wan
        # Without this, 16GB GPUs will OOM trying to load both models
        if self.pipe is not None:
            sys.stderr.write("Unloading Z-Image model to free VRAM...\n")
            sys.stderr.flush()
            del self.pipe
            self.pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            import time
            time.sleep(1)  # Allow GPU to fully release memory
            sys.stderr.write("Z-Image model unloaded.\n")
            sys.stderr.flush()

        sys.stderr.write(f"Loading Wan model from {wan_model_path}...\n")
        sys.stderr.flush()

        # Load VAE separately with float32 for better decoding quality (per HuggingFace docs)
        vae = AutoencoderKLWan.from_pretrained(
            wan_model_path,
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        # Load Wan pipeline with custom VAE
        video_pipe = WanPipeline.from_pretrained(
            wan_model_path,
            vae=vae,
            torch_dtype=torch.bfloat16,
        )

        # Configure scheduler with flow_shift (3.0 for 480p, 5.0 for 720p)
        flow_shift = 3.0  # Using 480p resolution
        video_pipe.scheduler = UniPCMultistepScheduler.from_config(
            video_pipe.scheduler.config,
            flow_shift=flow_shift
        )

        video_pipe.enable_model_cpu_offload()

        generator = torch.Generator("cuda").manual_seed(seed)

        # Generate video frames with progress callback
        sys.stderr.write("Generating video frames...\n")
        sys.stderr.flush()

        # Progress callback that emits JSON to stdout for Go to read
        def progress_callback(pipeline, step, timestep, callback_kwargs):
            # Calculate progress percentage based on denoising step (0-indexed)
            progress_percent = (step + 1) / num_inference_steps * 100
            # Estimate frames completed based on step progress
            frames_estimated = int((step + 1) / num_inference_steps * total_frames)

            # Emit progress message to stdout (Go reads this)
            progress_msg = {
                "type": "progress",
                "step": step + 1,
                "total_steps": num_inference_steps,
                "progress_percent": progress_percent,
                "frames_completed": frames_estimated
            }
            print(json.dumps(progress_msg), flush=True)

            return callback_kwargs

        output = video_pipe(
            prompt=prompt,
            negative_prompt=video_negative_prompt,
            width=width,
            height=height,
            num_frames=total_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg_scale,
            generator=generator,
            callback_on_step_end=progress_callback,
        )

        frames = output.frames[0]  # Get first batch
        sys.stderr.write(f"Generated {len(frames)} frames\n")
        sys.stderr.flush()

        # Encode to MP4 using ffmpeg
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
            tmp_path = tmp_video.name

        # Export frames to video using export_to_video helper or manual ffmpeg
        try:
            from diffusers.utils import export_to_video
            export_to_video(frames, tmp_path, fps=fps)
        except Exception as e:
            # Fallback to manual ffmpeg encoding if export_to_video fails
            sys.stderr.write(f"export_to_video failed ({e}), using ffmpeg fallback\\n")
            sys.stderr.flush()
            from PIL import Image
            import numpy as np
            with tempfile.TemporaryDirectory() as tmp_dir:
                for i, frame in enumerate(frames):
                    # Convert numpy array to PIL Image if needed
                    if isinstance(frame, np.ndarray):
                        frame = Image.fromarray(frame)
                    frame.save(os.path.join(tmp_dir, f"frame_{i:04d}.png"))

                subprocess.run([
                    'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', os.path.join(tmp_dir, 'frame_%04d.png'),
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-crf', '18',  # Higher quality encoding (was 23)
                    tmp_path
                ], check=True, capture_output=True)

        # Read video file and encode to base64
        with open(tmp_path, 'rb') as f:
            video_data = base64.b64encode(f.read()).decode()

        # Cleanup
        os.unlink(tmp_path)
        del video_pipe
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "video_data": video_data,
            "format": "mp4",
            "frames_generated": len(frames),
            "duration_seconds": duration_seconds
        }

    def face_swap(self, source_image_url, target_image_url, is_gif=False, swap_all_faces=True, enhance=True, max_frames=100):
        """Perform face swap on images or GIFs with security validation."""
        # Validate URLs (SSRF protection) - outside lock
        source_image_url = validate_image_url(source_image_url)
        target_image_url = validate_image_url(target_image_url)

        # Acquire lock for model check/load and timestamp update
        with self._faceswap_lock:
            self._last_faceswap_time = time.time()

            # Lazy load face swapper
            if self._face_swapper is None:
                model_path = os.environ.get("FACESWAP_MODEL_PATH", "")
                if not model_path:
                    raise ValueError("Face swap not available: FACESWAP_MODEL_PATH not set")

                sys.stderr.write(f"Loading face swap models from {model_path}...\n")
                sys.stderr.flush()

                # Add scripts directory to path for faceswap module
                scripts_dir = os.path.dirname(os.path.abspath(__file__))
                if scripts_dir not in sys.path:
                    sys.path.insert(0, scripts_dir)

                from faceswap import FaceSwapper, load_image_from_bytes, image_to_bytes, GifProcessor
                self._face_swapper = FaceSwapper(model_dir=model_path, enable_enhancement=enhance)
                self._faceswap_funcs = {
                    'load_image_from_bytes': load_image_from_bytes,
                    'image_to_bytes': image_to_bytes,
                    'GifProcessor': GifProcessor,
                }
                sys.stderr.write("Face swap models loaded.\n")
                sys.stderr.flush()

            # Get references while holding lock (prevents cleanup during processing)
            swapper = self._face_swapper
            funcs = self._faceswap_funcs

        # Processing happens outside lock to avoid blocking other requests
        sys.stderr.write(f"Face swap: is_gif={is_gif}, swap_all={swap_all_faces}, enhance={enhance}\n")
        sys.stderr.flush()

        # Download source image with size limit (DoS protection)
        source_data = download_with_limit(source_image_url)
        source_img = funcs['load_image_from_bytes'](source_data)

        # Download target with size limit
        target_data = download_with_limit(target_image_url)

        if is_gif:
            # Process GIF frame by frame
            processor = funcs['GifProcessor'](
                swapper,
                enhancer=swapper.enhancer if enhance else None
            )
            result_bytes = processor.process_gif(source_img, target_data, max_frames=max_frames, enhance=enhance)
            return {
                "image_data": base64.b64encode(result_bytes).decode(),
                "format": "gif",
                "frames_swapped": min(max_frames, 100)  # Approximate
            }
        else:
            # Process single image
            target_img = funcs['load_image_from_bytes'](target_data)
            result_img = swapper.swap(source_img, target_img, swap_all=swap_all_faces, enhance=enhance)
            result_bytes = funcs['image_to_bytes'](result_img, ".jpg", quality=95)
            return {
                "image_data": base64.b64encode(result_bytes).decode(),
                "format": "jpeg",
                "frames_swapped": 1
            }

    def handle_request(self, req):
        try:
            method = req.get("method")
            if method == "generate":
                return {"id": req.get("id", 0), "result": self.generate(**req.get("params", {})), "error": None}
            elif method == "generate_video":
                return {"id": req.get("id", 0), "result": self.generate_video(**req.get("params", {})), "error": None}
            elif method == "face_swap":
                return {"id": req.get("id", 0), "result": self.face_swap(**req.get("params", {})), "error": None}
            return {"id": req.get("id", 0), "result": None, "error": f"Unknown method: {method}"}
        except Exception as e:
            return {"id": req.get("id", 0), "result": None, "error": str(e)}

    def run(self):
        sys.stderr.write("Waiting for requests...\n")
        sys.stderr.flush()
        for line in sys.stdin:
            if line.strip():
                print(json.dumps(self.handle_request(json.loads(line))), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", required=True, help="Path to Z-Image-Turbo model directory")
    p.add_argument("--vae", "-v", help="Path to VAE (optional)")
    args = p.parse_args()

    svc = InferenceService(args.model, args.vae)
    try:
        svc.initialize()
        svc.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.stderr.write(f"FATAL: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
INFERENCE_EOF
fi

chmod +x "$INSTALL_DIR/scripts/inference.py"

# =============================================================================
# Install Face-Swap Module
# =============================================================================
SCRIPT_DIR="$(dirname "$(readlink -f "$0")" 2>/dev/null || echo "")"
GITHUB_RAW="https://raw.githubusercontent.com/Gelotto/power-node/main"

# Create faceswap directory
mkdir -p "$INSTALL_DIR/scripts/faceswap"

# Try local copy first (when running ./install.sh from repo)
if [ -d "$SCRIPT_DIR/scripts/faceswap" ]; then
    cp -r "$SCRIPT_DIR/scripts/faceswap"/* "$INSTALL_DIR/scripts/faceswap/"
    echo -e "  ${GREEN}✓${NC} Installed faceswap module (local)"
else
    # Download from GitHub (when running via curl | bash)
    echo "  Downloading faceswap module from GitHub..."
    FACESWAP_FILES="__init__.py face_swap.py face_enhancer.py gif_processor.py torchvision_compat.py"
    DOWNLOAD_OK=true
    for file in $FACESWAP_FILES; do
        if ! curl -sSL "$GITHUB_RAW/scripts/faceswap/$file" -o "$INSTALL_DIR/scripts/faceswap/$file" 2>/dev/null; then
            DOWNLOAD_OK=false
            break
        fi
    done
    if [ "$DOWNLOAD_OK" = true ]; then
        echo -e "  ${GREEN}✓${NC} Installed faceswap module (downloaded)"
    else
        echo -e "  ${YELLOW}!${NC} Failed to download faceswap module"
    fi
fi

# Install download script
if [ -f "$SCRIPT_DIR/scripts/download_faceswap_models.py" ]; then
    cp "$SCRIPT_DIR/scripts/download_faceswap_models.py" "$INSTALL_DIR/scripts/"
    echo -e "  ${GREEN}✓${NC} Installed faceswap model download script (local)"
else
    if curl -sSL "$GITHUB_RAW/scripts/download_faceswap_models.py" -o "$INSTALL_DIR/scripts/download_faceswap_models.py" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Installed faceswap model download script (downloaded)"
    else
        echo -e "  ${YELLOW}!${NC} Failed to download faceswap model script"
    fi
fi

# =============================================================================
# Download Face-Swap Models (Auto-detected based on GPU capability)
# =============================================================================
# Check if system is face-swap capable: 6GB+ VRAM
FACESWAP_CAPABLE=false
if [ "$VRAM_GB" -ge 6 ]; then
    FACESWAP_CAPABLE=true
fi

if [ "$FACESWAP_CAPABLE" = true ] && [ "$SKIP_FACESWAP" != true ] && [ -f "$INSTALL_DIR/scripts/download_faceswap_models.py" ]; then
    echo -e "\n${YELLOW}[7.5/8] Downloading face-swap models (~860MB)...${NC}"
    echo -e "  ${YELLOW}Note: This may take several minutes${NC}"
    if "$INSTALL_DIR/venv/bin/python3" "$INSTALL_DIR/scripts/download_faceswap_models.py" "$INSTALL_DIR/models/faceswap" 2>&1; then
        echo -e "  ${GREEN}✓${NC} Face-swap models downloaded"
    else
        echo -e "  ${YELLOW}!${NC} Face-swap model download failed (non-critical)"
        echo -e "  ${YELLOW}!${NC} Run manually: python3 $INSTALL_DIR/scripts/download_faceswap_models.py $INSTALL_DIR/models/faceswap"
    fi
elif [ "$SKIP_FACESWAP" = true ]; then
    echo -e "\n  Face-swap: ${YELLOW}Skipped${NC} (--no-faceswap flag)"
elif [ "$VRAM_GB" -lt 6 ]; then
    echo -e "\n  Face-swap: ${YELLOW}Not available${NC} (${VRAM_GB}GB VRAM < 6GB required)"
fi

# =============================================================================
# Create Configuration
# =============================================================================
# Preserve existing API key and worker ID if config exists
EXISTING_API_KEY=""
EXISTING_WORKER_ID=""
if [ -f "$INSTALL_DIR/config/config.yaml" ]; then
    EXISTING_API_KEY=$(grep -E '^\s*key:' "$INSTALL_DIR/config/config.yaml" | head -1 | sed 's/.*key:\s*"\?\([^"]*\)"\?.*/\1/' | tr -d ' ')
    EXISTING_WORKER_ID=$(grep -E '^\s*id:' "$INSTALL_DIR/config/config.yaml" | head -1 | sed 's/.*id:\s*"\?\([^"]*\)"\?.*/\1/' | tr -d ' ')

    # Filter out placeholder values
    if [ "$EXISTING_API_KEY" = "YOUR_API_KEY_HERE" ] || [ "$EXISTING_API_KEY" = "" ]; then
        EXISTING_API_KEY=""
    fi
    if [ "$EXISTING_WORKER_ID" = "YOUR_WORKER_ID_HERE" ] || [ "$EXISTING_WORKER_ID" = "" ]; then
        EXISTING_WORKER_ID=""
    fi

    if [ -n "$EXISTING_API_KEY" ] || [ -n "$EXISTING_WORKER_ID" ]; then
        echo -e "  ${GREEN}✓${NC} Preserving existing credentials"
    fi
fi

if [ "$SERVICE_MODE" = "gguf" ]; then
cat > "$INSTALL_DIR/config/config.yaml" << EOF
# Power Node Configuration
# Register at https://gelotto.io/workers to get your credentials

api:
  url: $API_URL
  key: "${EXISTING_API_KEY}"  # Your API key here

model:
  service_mode: gguf
  vram_gb: $VRAM_GB

worker:
  id: "${EXISTING_WORKER_ID}"  # Your worker ID here
  hostname: "$(hostname)"
  gpu_info: "$GPU_NAME"
  poll_interval: 5s
  heartbeat_interval: 30s

python:
  executable: $INSTALL_DIR/venv/bin/python3
  script_path: $INSTALL_DIR/scripts/inference.py
  script_args:
    - "--diffusion"
    - "$DIFFUSION_PATH"
    - "--vae"
    - "$VAE_PATH"
    - "--text-encoder"
    - "$TEXT_ENCODER_PATH"
    - "--vram"
    - "$VRAM_GB"
  env:
    PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
    FACESWAP_MODEL_PATH: "$INSTALL_DIR/models/faceswap"
EOF
else
# PyTorch configuration
cat > "$INSTALL_DIR/config/config.yaml" << EOF
# Power Node Configuration (PyTorch Mode)
# Register at https://gelotto.io/workers to get your credentials

api:
  url: $API_URL
  key: "${EXISTING_API_KEY}"  # Your API key here

model:
  service_mode: pytorch
  vram_gb: $VRAM_GB

worker:
  id: "${EXISTING_WORKER_ID}"  # Your worker ID here
  hostname: "$(hostname)"
  gpu_info: "$GPU_NAME"
  poll_interval: 5s
  heartbeat_interval: 30s

python:
  executable: $INSTALL_DIR/venv/bin/python3
  script_path: $INSTALL_DIR/scripts/inference.py
  script_args:
    - "--model"
    - "$INSTALL_DIR/models/z-image-turbo"
  env:
    PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
    HF_HOME: "$INSTALL_DIR/models/.cache"
    FACESWAP_MODEL_PATH: "$INSTALL_DIR/models/faceswap"
EOF

# Only add video config if the video model was downloaded (check for model_index.json)
if [ -f "$INSTALL_DIR/models/Wan2.1-T2V-1.3B-Diffusers/model_index.json" ]; then
    # Add WAN_MODEL_PATH to python.env for Python script
    echo "    WAN_MODEL_PATH: \"$INSTALL_DIR/models/Wan2.1-T2V-1.3B-Diffusers\"" >> "$INSTALL_DIR/config/config.yaml"
    # Add video.model_path section for Go worker to detect video capability
    cat >> "$INSTALL_DIR/config/config.yaml" << VIDEOEOF

video:
  model_path: "$INSTALL_DIR/models/Wan2.1-T2V-1.3B-Diffusers"
VIDEOEOF
fi
fi

# Add faceswap config if models were downloaded
if [ -f "$INSTALL_DIR/models/faceswap/inswapper_128.onnx" ]; then
    cat >> "$INSTALL_DIR/config/config.yaml" << FACESWAPEOF

faceswap:
  enabled: true
  model_path: "$INSTALL_DIR/models/faceswap"
FACESWAPEOF
fi

# =============================================================================
# Create Start Script
# =============================================================================
cat > "$INSTALL_DIR/start.sh" << EOF
#!/bin/bash
set -e
cd "$INSTALL_DIR" || { echo "ERROR: Cannot cd to $INSTALL_DIR"; exit 1; }
source "$INSTALL_DIR/venv/bin/activate" || { echo "ERROR: Cannot activate venv"; exit 1; }
exec "$INSTALL_DIR/bin/power-node" -config "$INSTALL_DIR/config/config.yaml"
EOF
chmod +x "$INSTALL_DIR/start.sh"

# =============================================================================
# Create Update Script
# =============================================================================
cat > "$INSTALL_DIR/update.sh" << 'UPDATEEOF'
#!/bin/bash
set -e

INSTALL_DIR="${POWER_NODE_DIR:-$HOME/.power-node}"
GITHUB_REPO="Gelotto/power-node"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Power Node Updater${NC}"
echo ""

# Check current version
if [ -f "$INSTALL_DIR/bin/power-node" ]; then
    CURRENT_VERSION=$("$INSTALL_DIR/bin/power-node" --version 2>/dev/null || echo "unknown")
    echo -e "Current version: ${GREEN}$CURRENT_VERSION${NC}"
else
    echo -e "${RED}Power Node not installed. Run the installer first.${NC}"
    exit 1
fi

# Stop service if running
RESTART_SERVICE=false
if systemctl is-active --quiet power-node 2>/dev/null; then
    echo "Stopping power-node service..."
    sudo systemctl stop power-node
    RESTART_SERVICE=true
fi

# Download latest binary
ARCH=$(uname -m)
OS=$(uname -s | tr '[:upper:]' '[:lower:]')

# Normalize architecture name
if [ "$ARCH" = "x86_64" ]; then
    ARCH="x86_64"
elif [ "$ARCH" = "aarch64" ]; then
    ARCH="arm64"
fi

BINARY_URL="https://github.com/$GITHUB_REPO/releases/latest/download/power-node-${OS}-${ARCH}"

echo "Downloading latest binary..."
if ! curl -sSL "$BINARY_URL" -o "$INSTALL_DIR/bin/power-node.new" --fail; then
    echo -e "${RED}Failed to download update${NC}"
    if [ "$RESTART_SERVICE" = true ]; then
        sudo systemctl start power-node
    fi
    exit 1
fi

chmod +x "$INSTALL_DIR/bin/power-node.new"

# Backup and swap binaries
mv "$INSTALL_DIR/bin/power-node" "$INSTALL_DIR/bin/power-node.old"
mv "$INSTALL_DIR/bin/power-node.new" "$INSTALL_DIR/bin/power-node"

# Show new version
NEW_VERSION=$("$INSTALL_DIR/bin/power-node" --version 2>/dev/null || echo "unknown")
echo -e "Updated to: ${GREEN}$NEW_VERSION${NC}"

# Restart service if it was running
if [ "$RESTART_SERVICE" = true ]; then
    echo "Restarting power-node service..."
    sudo systemctl start power-node
    echo -e "${GREEN}Service restarted!${NC}"
fi

echo ""
echo -e "${GREEN}Update complete!${NC}"
UPDATEEOF
chmod +x "$INSTALL_DIR/update.sh"

# =============================================================================
# Create Systemd Service
# =============================================================================
cat > "$INSTALL_DIR/power-node.service" << EOF
[Unit]
Description=Power Node - Gelotto GPU Compute
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/start.sh
Restart=always
RestartSec=10
Environment=HOME=$HOME

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=power-node

# Timeouts (model loading can take a while)
TimeoutStartSec=300
TimeoutStopSec=120

# Resource limits
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF

# =============================================================================
# Done
# =============================================================================
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                   INSTALLATION COMPLETE                       ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BLUE}Installation:${NC}  $INSTALL_DIR"
echo -e "  ${BLUE}GPU:${NC}           $GPU_NAME ($VRAM_GB GB)"
echo -e "  ${BLUE}Mode:${NC}          $SERVICE_MODE"

# Check face-swap availability
FACESWAP_STATUS="Not installed"
if [ -f "$INSTALL_DIR/models/faceswap/inswapper_128.onnx" ]; then
    FACESWAP_STATUS="Enabled"
fi
echo -e "  ${BLUE}Face-Swap:${NC}     $FACESWAP_STATUS"

# Check video availability
VIDEO_STATUS="Not available"
if [ -f "$INSTALL_DIR/models/Wan2.1-T2V-1.3B-Diffusers/model_index.json" ]; then
    VIDEO_STATUS="Enabled"
elif [ "$SERVICE_MODE" = "gguf" ]; then
    VIDEO_STATUS="Not available (GGUF mode)"
elif [ "$VRAM_GB" -lt 12 ]; then
    VIDEO_STATUS="Not available (${VRAM_GB}GB VRAM)"
fi
echo -e "  ${BLUE}Video:${NC}         $VIDEO_STATUS"
echo ""

# Offer to install systemd service
echo -e "${CYAN}Service Installation:${NC}"
echo ""
SERVICE_INSTALLED=false

# Check if we can read from terminal (won't work with curl | bash)
if read -p "  Enable Power Node to run at system startup? (Y/n) " -r < /dev/tty 2>/dev/null; then
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo "  Installing systemd service..."
        if sudo cp "$INSTALL_DIR/power-node.service" /etc/systemd/system/ && \
           sudo systemctl daemon-reload && \
           sudo systemctl enable power-node; then
            echo -e "  ${GREEN}✓ Service installed and enabled${NC}"
            SERVICE_INSTALLED=true
        else
            echo -e "  ${YELLOW}Warning: Could not install service automatically${NC}"
            echo "  You can install it manually later (see commands below)"
        fi
    else
        echo ""
        echo "  To install manually later:"
        echo "    sudo cp $INSTALL_DIR/power-node.service /etc/systemd/system/"
        echo "    sudo systemctl daemon-reload"
        echo "    sudo systemctl enable power-node"
    fi
else
    # Non-interactive mode (curl | bash) - show manual instructions
    echo "  [Non-interactive mode detected]"
    echo ""
    echo "  To enable the service, run:"
    echo "    sudo cp $INSTALL_DIR/power-node.service /etc/systemd/system/"
    echo "    sudo systemctl daemon-reload"
    echo "    sudo systemctl enable power-node"
    echo "    sudo systemctl start power-node"
fi
echo ""

echo -e "${CYAN}Next steps:${NC}"
echo ""
echo "  1. Register at https://gelotto.io/workers"
echo ""
echo "  2. Add your credentials to $INSTALL_DIR/config/config.yaml:"
echo "     api:"
echo "       key: \"YOUR_API_KEY\""
echo "     worker:"
echo "       id: \"YOUR_WORKER_ID\""
echo ""
if [ "$SERVICE_INSTALLED" = true ]; then
    echo "  3. Start the service:"
    echo "     sudo systemctl start power-node"
else
    echo "  3. Start the node:"
    echo "     $INSTALL_DIR/start.sh"
fi
echo ""
echo -e "${CYAN}Useful commands:${NC}"
echo ""
echo "  Check status:   $INSTALL_DIR/bin/power-node --status"
echo "  Validate setup: $INSTALL_DIR/bin/power-node --check"
echo "  Update:         $INSTALL_DIR/update.sh"
echo "  Show version:   $INSTALL_DIR/bin/power-node --version"
echo ""
echo -e "${CYAN}Service commands:${NC}"
echo ""
echo "  Start service:  sudo systemctl start power-node"
echo "  Stop service:   sudo systemctl stop power-node"
echo "  View logs:      sudo journalctl -u power-node -f"
echo "  Service status: sudo systemctl status power-node"
echo ""
echo -e "${GREEN}Happy computing!${NC}"
