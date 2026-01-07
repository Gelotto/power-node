#!/bin/bash
# =============================================================================
# models.sh - Model download functions with validation
# =============================================================================

# Download a file with progress indicator
download_file() {
    local url="$1"
    local dest="$2"
    local name="$3"

    # Check if already downloaded
    if [ -f "$dest" ]; then
        log_info "$name: Already exists, skipping download"
        return 0
    fi

    log_info "Downloading $name..."
    if curl -fSL "$url" -o "$dest" --progress-bar; then
        log_success "$name downloaded"
        return 0
    else
        log_error "Failed to download $name"
        rm -f "$dest"
        return 1
    fi
}

# Check available disk space
check_disk_space() {
    local required_gb="$1"
    local path="${2:-$INSTALL_DIR}"

    local available_kb=$(df -k "$path" | tail -1 | awk '{print $4}')
    local available_gb=$((available_kb / 1024 / 1024))

    if [ "$available_gb" -lt "$required_gb" ]; then
        return 1
    fi
    return 0
}

# Check disk space with warning prompt
check_disk_space_warning() {
    local required_gb="$1"
    local description="$2"

    if ! check_disk_space "$required_gb"; then
        local available_kb=$(df -k "$INSTALL_DIR" | tail -1 | awk '{print $4}')
        local available_gb=$((available_kb / 1024 / 1024))
        log_warning "Low disk space for $description"
        log_info "Required: ${required_gb}GB, Available: ${available_gb}GB"
        return 1
    fi
    return 0
}

# Download Z-Image GGUF models
download_zimage_gguf() {
    log_info "Downloading Z-Image-Turbo GGUF models..."

    # Select quantization based on VRAM
    local quant="Q8_0"
    if [ "$VRAM_GB" -lt "$ZIMAGE_Q4_THRESHOLD" ]; then
        quant="Q4_0"
        log_info "Using Q4_0 quantization (${VRAM_GB}GB VRAM detected)"
    else
        log_info "Using Q8_0 quantization (${VRAM_GB}GB+ VRAM detected)"
    fi

    local base_url="$HF_BASE/city96/Z-Image-Turbo-GGUF/resolve/main"

    download_file "$base_url/z-image-turbo-${quant}.gguf" \
        "$INSTALL_DIR/models/diffusion/z-image-turbo-${quant}.gguf" \
        "Z-Image-Turbo diffusion model"

    download_file "$base_url/sdxl_vae.safetensors" \
        "$INSTALL_DIR/models/vae/sdxl_vae.safetensors" \
        "SDXL VAE"

    download_file "$base_url/Qwen2.5-0.5B-${quant}.gguf" \
        "$INSTALL_DIR/models/text_encoder/Qwen2.5-0.5B-${quant}.gguf" \
        "Qwen2.5 text encoder"

    log_success "Z-Image-Turbo GGUF models installed"

    # Return quantization used for config generation
    echo "$quant"
}

# Download FLUX GGUF models
download_flux_gguf() {
    log_info "Downloading FLUX.1-schnell GGUF models..."

    # Select quantization based on VRAM
    local quant="Q8_0"
    if [ "$VRAM_GB" -lt "$FLUX_Q4_THRESHOLD" ]; then
        quant="Q4_K_S"
        log_info "Using Q4_K_S quantization (${VRAM_GB}GB VRAM detected)"
    else
        log_info "Using Q8_0 quantization (${VRAM_GB}GB+ VRAM detected)"
    fi

    local base_url="$HF_BASE/city96/FLUX.1-schnell-gguf/resolve/main"

    download_file "$base_url/flux1-schnell-${quant}.gguf" \
        "$INSTALL_DIR/models/diffusion/flux1-schnell-${quant}.gguf" \
        "FLUX.1-schnell diffusion model"

    download_file "$base_url/clip_l.safetensors" \
        "$INSTALL_DIR/models/text_encoder/clip_l.safetensors" \
        "CLIP-L encoder"

    download_file "$base_url/t5xxl_fp16.safetensors" \
        "$INSTALL_DIR/models/text_encoder/t5xxl_fp16.safetensors" \
        "T5-XXL encoder"

    download_file "$base_url/ae.safetensors" \
        "$INSTALL_DIR/models/vae/ae.safetensors" \
        "FLUX VAE"

    log_success "FLUX.1-schnell GGUF models installed"

    # Return quantization used for config generation
    echo "$quant"
}

# Download Z-Image PyTorch model (via HuggingFace Hub)
download_zimage_pytorch() {
    log_info "Downloading Z-Image-Turbo PyTorch model..."

    local model_dir="$INSTALL_DIR/models/Z-Image-Turbo"

    # Check if already downloaded
    if [ -d "$model_dir" ] && [ -f "$model_dir/model_index.json" ]; then
        log_info "Z-Image-Turbo: Already exists, skipping download"
        return 0
    fi

    activate_venv

    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Tongyi-MAI/Z-Image-Turbo',
    local_dir='$model_dir',
    local_dir_use_symlinks=False
)
"
    local exit_code=$?

    deactivate_venv

    if [ $exit_code -ne 0 ]; then
        log_fatal "Failed to download Z-Image-Turbo model"
    fi

    log_success "Z-Image-Turbo PyTorch model installed"
}

# Download FLUX PyTorch model (requires HuggingFace token)
download_flux_pytorch() {
    log_info "Downloading FLUX.1-schnell PyTorch model..."

    local model_dir="$INSTALL_DIR/models/FLUX.1-schnell"

    # Check if already downloaded
    if [ -d "$model_dir" ] && [ -f "$model_dir/model_index.json" ]; then
        log_info "FLUX.1-schnell: Already exists, skipping download"
        return 0
    fi

    # Check for HuggingFace token
    if [ -z "$HF_TOKEN" ]; then
        log_error "FLUX.1-schnell requires a HuggingFace token."
        log_info "1. Create account at https://huggingface.co/"
        log_info "2. Accept license at https://huggingface.co/black-forest-labs/FLUX.1-schnell"
        log_info "3. Get token from https://huggingface.co/settings/tokens"
        log_info "4. Re-run: HF_TOKEN=hf_your_token_here ./install.sh"
        return 1
    fi

    activate_venv

    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'black-forest-labs/FLUX.1-schnell',
    local_dir='$model_dir',
    local_dir_use_symlinks=False,
    token='$HF_TOKEN'
)
"
    local exit_code=$?

    deactivate_venv

    if [ $exit_code -ne 0 ]; then
        log_error "Failed to download FLUX.1-schnell model"
        return 1
    fi

    log_success "FLUX.1-schnell PyTorch model installed"
}

# Download Wan2.1 video model
download_video_model() {
    log_info "Downloading Wan2.1 video model..."

    local model_dir="$INSTALL_DIR/models/Wan2.1-T2V-1.3B-Diffusers"

    # Check if already downloaded
    if [ -d "$model_dir" ] && [ -f "$model_dir/model_index.json" ]; then
        log_info "Wan2.1: Already exists, skipping download"
        return 0
    fi

    # Check disk space
    if ! check_disk_space_warning 50 "Video model (~29GB)"; then
        log_warning "Skipping video model due to disk space"
        return 1
    fi

    activate_venv

    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
    local_dir='$model_dir',
    local_dir_use_symlinks=False
)
"
    local exit_code=$?

    deactivate_venv

    if [ $exit_code -eq 0 ] && [ -f "$model_dir/model_index.json" ]; then
        log_success "Wan2.1 video model installed"
        return 0
    else
        log_error "Wan2.1 model download incomplete or failed"
        if [ ! -f "$model_dir/model_index.json" ]; then
            log_info "Missing: model_index.json"
        fi
        return 1
    fi
}

# Download face-swap models
download_faceswap_models() {
    log_info "Downloading face-swap models..."

    local model_dir="$INSTALL_DIR/models/faceswap"
    mkdir -p "$model_dir"

    # Check if models already exist
    if [ -f "$model_dir/inswapper_128.onnx" ] && [ -f "$model_dir/GFPGANv1.4.pth" ]; then
        log_info "Face-swap models: Already exist, skipping download"
        return 0
    fi

    # Use the download script if available
    if [ -f "$INSTALL_DIR/scripts/download_faceswap_models.py" ]; then
        activate_venv
        python3 "$INSTALL_DIR/scripts/download_faceswap_models.py" --output-dir "$model_dir"
        local exit_code=$?
        deactivate_venv

        if [ $exit_code -eq 0 ]; then
            log_success "Face-swap models installed"
            return 0
        fi
    fi

    log_warning "Face-swap models download failed or script not found"
    return 1
}

# Download all models based on service mode and flags
download_models() {
    log_step 7 "Downloading AI models..."

    if [ "$SERVICE_MODE" = "gguf" ]; then
        if [ "$INSTALL_ZIMAGE" = true ]; then
            ZIMAGE_QUANT=$(download_zimage_gguf)
        fi
        if [ "$INSTALL_FLUX" = true ]; then
            FLUX_QUANT=$(download_flux_gguf)
        fi
    else
        if [ "$INSTALL_ZIMAGE" = true ]; then
            download_zimage_pytorch
        fi
        if [ "$INSTALL_FLUX" = true ]; then
            download_flux_pytorch
        fi

        # Video model only for PyTorch mode with 12GB+ VRAM
        if is_video_capable && [ "$SKIP_VIDEO" != true ]; then
            download_video_model
        fi
    fi

    # Face-swap models
    if [ "$SKIP_FACESWAP" != true ]; then
        download_faceswap_models
    fi

    log_success "Model downloads complete"
}
