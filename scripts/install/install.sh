#!/bin/bash
# =============================================================================
# Power Node Installation Script
# https://github.com/Gelotto/power-node
#
# This script automatically installs the Power Node worker software.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/Gelotto/power-node/main/scripts/install/install.sh | bash
#   ./install.sh [--no-video] [--no-faceswap] [--minimal]
#
# Requirements:
#   - NVIDIA GPU with 8GB+ VRAM
#   - NVIDIA drivers with nvidia-smi
#   - Python 3.10+
#   - ~50GB free disk space (depends on models)
#
# Environment Variables:
#   POWER_NODE_DIR    Installation directory (default: ~/.power-node)
#   API_URL           Backend API URL (default: https://api.gelotto.io)
#   HF_TOKEN          HuggingFace token for FLUX model (required for FLUX)
# =============================================================================

set -e

# Determine script directory for sourcing library modules
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source library modules
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/system.sh"
source "$SCRIPT_DIR/lib/gpu.sh"
source "$SCRIPT_DIR/lib/config.sh"
source "$SCRIPT_DIR/lib/python.sh"
source "$SCRIPT_DIR/lib/models.sh"

# =============================================================================
# Parse Command Line Arguments
# =============================================================================
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
            echo ""
            echo "Model selection is fully automatic based on VRAM:"
            echo "  - 12GB+ VRAM: Both z-image-turbo + flux-schnell (multi-model mode)"
            echo "  - 8-11GB VRAM: z-image-turbo only"
            echo ""
            echo "Environment variables:"
            echo "  HF_TOKEN        HuggingFace token for FLUX model (required for PyTorch mode)"
            echo "                  Get one at: https://huggingface.co/settings/tokens"
            exit 0
            ;;
    esac
    shift
done

# =============================================================================
# Main Installation Flow
# =============================================================================
main() {
    print_banner

    # Initialize configuration
    init_config

    # Step 1: Check system requirements
    check_system_requirements

    # Step 2: Detect GPU
    detect_gpu

    # Step 3: Validate GPU and determine service mode
    validate_gpu

    # Step 3.5: Detect model support based on VRAM
    detect_model_support
    export_gpu_info

    # Step 4: Create directory structure
    create_directories

    # Step 5: Install binary
    install_binary

    # Step 6: Setup Python environment and dependencies
    install_python_dependencies

    # Step 7: Download models
    download_models

    # Step 8: Copy inference scripts from repo
    copy_inference_scripts

    # Step 9: Generate configuration
    generate_configuration

    # Step 10: Generate helper scripts
    generate_helper_scripts

    # Print completion summary
    print_summary
}

# =============================================================================
# Binary Installation
# =============================================================================
install_binary() {
    log_step 5 "Installing Power Node binary..."

    local arch=$(uname -m)
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')

    # Normalize architecture names
    if [ "$arch" = "aarch64" ]; then
        arch="arm64"
    fi

    local binary_url="https://github.com/$GITHUB_REPO/releases/latest/download/power-node-${os}-${arch}"
    log_info "Binary URL: $binary_url"

    log_info "Downloading pre-built binary..."
    if curl -fsSL "$binary_url" -o "$INSTALL_DIR/bin/power-node"; then
        chmod +x "$INSTALL_DIR/bin/power-node"
        log_success "Binary downloaded"
    else
        log_warning "Download failed, building from source..."
        rm -f "$INSTALL_DIR/bin/power-node"
        build_from_source
    fi

    # Verify binary installation
    if [ ! -f "$INSTALL_DIR/bin/power-node" ] || [ ! -x "$INSTALL_DIR/bin/power-node" ]; then
        log_fatal "Binary not found or not executable"
    fi

    log_success "Binary installed"
}

# Build from source if pre-built binary unavailable
build_from_source() {
    log_info "Pre-built binary not available, building from source..."

    require_command go "Go not found. Please install Go 1.21+ from https://go.dev/doc/install"
    require_command git "git not found. Required for building from source."

    local temp_dir=$(mktemp -d)
    trap "rm -rf '$temp_dir'" EXIT

    git clone --quiet "https://github.com/$GITHUB_REPO.git" "$temp_dir/power-node"
    (cd "$temp_dir/power-node" && go build -o "$INSTALL_DIR/bin/power-node" ./cmd/power-node)

    rm -rf "$temp_dir"
    trap - EXIT
}

# =============================================================================
# Copy Inference Scripts
# =============================================================================
copy_inference_scripts() {
    log_info "Installing inference scripts..."

    local scripts_src="$SCRIPT_DIR/../inference"

    # Copy shared utilities
    cp "$scripts_src/shared_utils.py" "$INSTALL_DIR/scripts/"

    # Copy appropriate inference script based on service mode and model selection
    if [ "$SERVICE_MODE" = "gguf" ]; then
        if [ "$IMAGE_MODEL" = "flux-schnell" ]; then
            cp "$scripts_src/inference_gguf_flux.py" "$INSTALL_DIR/scripts/inference.py"
        else
            cp "$scripts_src/inference_gguf_zimage.py" "$INSTALL_DIR/scripts/inference.py"
        fi
    else
        # PyTorch mode
        if [ "$INSTALL_BOTH_MODELS" = true ]; then
            cp "$scripts_src/inference_pytorch_zimage.py" "$INSTALL_DIR/scripts/inference_zimage.py"
            cp "$scripts_src/inference_pytorch_flux.py" "$INSTALL_DIR/scripts/inference_flux.py"
            ln -sf "$INSTALL_DIR/scripts/inference_zimage.py" "$INSTALL_DIR/scripts/inference.py"
            log_info "Multi-model scripts installed (inference_zimage.py, inference_flux.py)"
            log_info "${CYAN}  Default: inference.py -> inference_zimage.py${NC}"
        elif [ "$IMAGE_MODEL" = "flux-schnell" ]; then
            cp "$scripts_src/inference_pytorch_flux.py" "$INSTALL_DIR/scripts/inference.py"
        else
            cp "$scripts_src/inference_pytorch_zimage.py" "$INSTALL_DIR/scripts/inference.py"
        fi
    fi

    # Copy face-swap module if available
    if [ -d "$SCRIPT_DIR/../faceswap" ]; then
        cp -r "$SCRIPT_DIR/../faceswap" "$INSTALL_DIR/scripts/"
    fi

    chmod +x "$INSTALL_DIR/scripts/inference.py" 2>/dev/null || true

    log_success "Inference scripts installed"
}

# =============================================================================
# Generate Configuration
# =============================================================================
generate_configuration() {
    log_info "Generating configuration..."

    if [ "$SERVICE_MODE" = "gguf" ]; then
        if [ "$IMAGE_MODEL" = "flux-schnell" ]; then
            generate_gguf_flux_config "${FLUX_QUANT:-Q8_0}"
        else
            generate_gguf_zimage_config "${ZIMAGE_QUANT:-Q8_0}"
        fi
    else
        if [ "$IMAGE_MODEL" = "flux-schnell" ]; then
            generate_pytorch_flux_config
        else
            generate_pytorch_zimage_config
        fi

        # Add multi-model configuration if both models installed
        if [ "$INSTALL_BOTH_MODELS" = true ]; then
            append_multimodel_config \
                "$INSTALL_DIR/models/Z-Image-Turbo" \
                "$INSTALL_DIR/models/FLUX.1-schnell"
        fi

        # Add video configuration if available
        if is_video_capable && [ "$SKIP_VIDEO" != true ]; then
            if [ -d "$INSTALL_DIR/models/Wan2.1-T2V-1.3B-Diffusers" ]; then
                append_video_config "$INSTALL_DIR/models/Wan2.1-T2V-1.3B-Diffusers"
            fi
        fi
    fi

    # Add face-swap configuration if available
    if [ "$SKIP_FACESWAP" != true ] && [ -d "$INSTALL_DIR/models/faceswap" ]; then
        append_faceswap_config "$INSTALL_DIR/models/faceswap"
    fi

    log_success "Configuration generated"
}

# =============================================================================
# Generate Helper Scripts
# =============================================================================
generate_helper_scripts() {
    generate_start_script
    generate_systemd_service
    generate_update_script

    log_success "Helper scripts generated"
}

# =============================================================================
# Print Summary
# =============================================================================
print_summary() {
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}              Installation Complete!                            ${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "Install directory: ${CYAN}$INSTALL_DIR${NC}"
    echo -e "Service mode:      ${CYAN}$SERVICE_MODE${NC}"
    echo -e "Primary model:     ${CYAN}$IMAGE_MODEL${NC}"

    if [ "$INSTALL_BOTH_MODELS" = true ]; then
        echo -e "Secondary model:   ${CYAN}flux-schnell${NC}"
    fi

    if is_video_capable && [ "$SKIP_VIDEO" != true ]; then
        echo -e "Video support:     ${GREEN}Enabled${NC}"
    fi

    if [ "$SKIP_FACESWAP" != true ]; then
        echo -e "Face-swap:         ${GREEN}Enabled${NC}"
    fi

    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  1. Register your worker at https://picshapes.com/workers/register"
    echo -e "  2. Add your API key to: ${CYAN}$INSTALL_DIR/config/config.yaml${NC}"
    echo -e "  3. Start the worker: ${CYAN}$INSTALL_DIR/start.sh${NC}"
    echo ""
    echo -e "${YELLOW}To install as systemd service:${NC}"
    echo -e "  sudo cp $INSTALL_DIR/power-node.service /etc/systemd/system/"
    echo -e "  sudo systemctl daemon-reload"
    echo -e "  sudo systemctl enable --now power-node"
    echo ""
}

# Run main installation
main
