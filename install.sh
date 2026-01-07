#!/bin/bash
# =============================================================================
# Power Node Installation Script (Bootstrap)
# https://github.com/Gelotto/power-node
#
# This bootstrap script downloads and runs the modular installer.
# Supports: curl -sSL https://raw.githubusercontent.com/.../install.sh | bash
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/Gelotto/power-node/main/install.sh | bash
#   ./install.sh [--no-video] [--no-faceswap] [--minimal] [--help]
#
# For local development, you can also run:
#   ./scripts/install/install.sh
# =============================================================================

set -e

REPO="Gelotto/power-node"
BRANCH="main"

# Handle --help before downloading (for faster response)
for arg in "$@"; do
    if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
        echo "Power Node Installation Script"
        echo ""
        echo "Usage: install.sh [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --no-video      Skip video model (auto-downloaded for 12GB+ VRAM PyTorch GPUs)"
        echo "  --no-faceswap   Skip face-swap model (auto-downloaded for 6GB+ VRAM)"
        echo "  --minimal       Skip all optional models (video + face-swap)"
        echo "  --help, -h      Show this help message"
        echo ""
        echo "Model selection is fully automatic based on VRAM:"
        echo "  - 12GB+ VRAM: Both z-image-turbo + flux-schnell (multi-model mode)"
        echo "  - 8-11GB VRAM: z-image-turbo only"
        echo ""
        echo "Environment variables:"
        echo "  HF_TOKEN        HuggingFace token for FLUX model (required for PyTorch mode)"
        echo "                  Get one at: https://huggingface.co/settings/tokens"
        echo "  POWER_NODE_DIR  Installation directory (default: ~/.power-node)"
        exit 0
    fi
done

# Check for required commands
if ! command -v curl &> /dev/null; then
    echo "ERROR: curl is required but not found."
    exit 1
fi

if ! command -v tar &> /dev/null; then
    echo "ERROR: tar is required but not found."
    exit 1
fi

# Create temporary directory for installer
INSTALL_TMP=$(mktemp -d)
trap 'rm -rf "$INSTALL_TMP"' EXIT

echo "Downloading Power Node installer..."

# Download the repository as a tarball
if ! curl -fsSL "https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz" | \
     tar -xzf - -C "$INSTALL_TMP" --strip-components=1; then
    echo "ERROR: Failed to download installer from GitHub."
    echo "Check your internet connection and try again."
    exit 1
fi

# Verify the modular installer exists
if [ ! -f "$INSTALL_TMP/scripts/install/install.sh" ]; then
    echo "ERROR: Installer files not found in downloaded package."
    echo "The repository structure may have changed."
    exit 1
fi

# Run the modular installer
cd "$INSTALL_TMP/scripts/install"
exec bash install.sh "$@"
