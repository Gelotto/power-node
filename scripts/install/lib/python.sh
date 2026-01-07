#!/bin/bash
# =============================================================================
# python.sh - Python virtual environment and dependency installation
# =============================================================================

# Create or validate Python virtual environment
setup_venv() {
    log_step 6 "Setting up Python environment..."

    # Validate venv completeness (not just existence)
    local venv_valid=true
    if [ ! -f "$INSTALL_DIR/venv/bin/activate" ] || \
       [ ! -f "$INSTALL_DIR/venv/bin/pip" ] || \
       [ ! -f "$INSTALL_DIR/venv/bin/python3" ]; then
        venv_valid=false
    fi

    if [ "$venv_valid" = false ]; then
        log_info "Creating Python virtual environment..."
        rm -rf "$INSTALL_DIR/venv"  # Remove incomplete venv

        if ! python3 -m venv "$INSTALL_DIR/venv"; then
            log_fatal "Failed to create Python virtual environment. Try: sudo apt install python3-venv"
        fi

        # Verify creation was successful
        if [ ! -f "$INSTALL_DIR/venv/bin/activate" ]; then
            log_fatal "Virtual environment creation incomplete. The venv module may be broken."
        fi
    else
        log_info "Using existing Python virtual environment"
    fi
}

# Activate virtual environment
activate_venv() {
    source "$INSTALL_DIR/venv/bin/activate"
}

# Deactivate virtual environment
deactivate_venv() {
    deactivate 2>/dev/null || true
}

# Upgrade pip in virtual environment
upgrade_pip() {
    activate_venv
    if ! pip install --upgrade pip --quiet; then
        deactivate_venv
        log_fatal "Failed to upgrade pip"
    fi
}

# Install GGUF mode dependencies (stable-diffusion-cpp-python)
install_gguf_dependencies() {
    log_info "Installing stable-diffusion-cpp-python..."
    log_info "(This may take several minutes for CUDA compilation)"

    ensure_build_tools

    activate_venv

    if command_exists nvcc; then
        log_info "CUDA detected: $(nvcc --version | grep release | awk '{print $6}')"
        if ! CMAKE_ARGS="-DSD_CUDA=ON" pip install 'stable-diffusion-cpp-python>=0.4.0'; then
            deactivate_venv
            log_fatal "Failed to compile stable-diffusion-cpp-python. CUDA headers may be missing or incompatible."
        fi
    else
        ensure_cuda_toolkit
        if command_exists nvcc; then
            if ! CMAKE_ARGS="-DSD_CUDA=ON" pip install 'stable-diffusion-cpp-python>=0.4.0'; then
                deactivate_venv
                log_fatal "Failed to compile stable-diffusion-cpp-python"
            fi
        else
            log_warning "nvcc not found, installing CPU-only (slower inference)"
            if ! pip install 'stable-diffusion-cpp-python>=0.4.0'; then
                deactivate_venv
                log_fatal "Failed to install stable-diffusion-cpp-python"
            fi
        fi
    fi

    pip install pillow --quiet || { deactivate_venv; log_fatal "Failed to install pillow"; }

    # Verify installation
    if ! python3 -c "from stable_diffusion_cpp import StableDiffusion" 2>/dev/null; then
        deactivate_venv
        log_fatal "stable-diffusion-cpp-python installed but cannot be imported."
    fi

    deactivate_venv
    log_success "ML packages installed (GGUF mode)"
}

# Install PyTorch mode dependencies
install_pytorch_dependencies() {
    log_info "Installing PyTorch and dependencies for Blackwell GPU..."

    activate_venv

    # Blackwell GPUs (sm_120) require PyTorch nightly with CUDA 12.8+
    if ! pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --quiet; then
        deactivate_venv
        log_fatal "Failed to install PyTorch"
    fi

    # Pin diffusers>=0.32.0 for Wan video model support
    if ! pip install transformers 'diffusers>=0.32.0' safetensors accelerate tqdm pillow ftfy imageio imageio-ffmpeg sentencepiece protobuf --quiet; then
        deactivate_venv
        log_fatal "Failed to install ML dependencies"
    fi

    deactivate_venv
    log_success "ML packages installed (PyTorch mode)"
}

# Install face-swap dependencies
install_faceswap_dependencies() {
    log_info "Installing face-swap dependencies..."

    activate_venv

    # Get Python minor version for compatibility checks
    local python_minor=$(python3 -c 'import sys; print(sys.version_info.minor)')

    # Install insightface and dependencies
    pip install insightface opencv-python-headless requests imageio imageio-ffmpeg --quiet

    # Install ONNX runtime (GPU preferred, CPU fallback)
    if ! pip install onnxruntime-gpu --quiet 2>/dev/null; then
        log_warning "GPU ONNX runtime not available, using CPU version"
        pip install onnxruntime --quiet
    fi

    # Install GFPGAN for face enhancement
    # Python 3.13+ needs special handling for basicsr compatibility
    if [ "$python_minor" -ge 13 ]; then
        log_info "Python 3.13+ detected, installing compatible basicsr..."
        pip install 'git+https://github.com/Disty0/basicsr.git' --quiet 2>/dev/null || true
    fi

    if ! pip install gfpgan --quiet 2>/dev/null; then
        log_warning "GFPGAN not available, face enhancement will be disabled"
    fi

    deactivate_venv
    log_success "Face-swap dependencies installed"
}

# Install all Python dependencies based on service mode
install_python_dependencies() {
    setup_venv
    upgrade_pip

    if [ "$SERVICE_MODE" = "gguf" ]; then
        install_gguf_dependencies
    else
        install_pytorch_dependencies
    fi

    # Install face-swap dependencies if not skipped
    if [ "$SKIP_FACESWAP" != true ]; then
        install_faceswap_dependencies
    fi
}
