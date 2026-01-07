#!/bin/bash
# =============================================================================
# gpu.sh - GPU detection and validation
# =============================================================================

# GPU requirements constants
MIN_VRAM_REQUIRED=8
BLACKWELL_MIN_VRAM=14
MULTI_MODEL_THRESHOLD=12

# GPU info (populated by detect_gpu)
GPU_NAME=""
VRAM_MB=0
VRAM_GB=0
COMPUTE_CAP=""
IS_BLACKWELL=false
SERVICE_MODE=""

# Detect GPU information
detect_gpu() {
    log_step 2 "Detecting GPU..."

    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    VRAM_GB=$(( (VRAM_MB + 512) / 1024 ))
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')

    log_info "GPU: ${GREEN}$GPU_NAME${NC}"
    log_info "VRAM: ${GREEN}${VRAM_GB}GB${NC}"
    log_info "Compute Capability: ${GREEN}${COMPUTE_CAP:0:1}.${COMPUTE_CAP:1}${NC}"
}

# Validate GPU meets requirements
validate_gpu() {
    log_step 3 "Validating GPU requirements..."

    if [ "$VRAM_GB" -lt "$MIN_VRAM_REQUIRED" ]; then
        log_fatal "Minimum ${MIN_VRAM_REQUIRED}GB VRAM required. Your GPU has ${VRAM_GB}GB."
    fi

    # Check for Blackwell architecture (sm_120+)
    IS_BLACKWELL=false
    if [ "$COMPUTE_CAP" -ge 120 ] 2>/dev/null; then
        IS_BLACKWELL=true
    fi

    # Blackwell GPUs require PyTorch mode due to GGUF kernel incompatibility
    if [ "$IS_BLACKWELL" = true ] && [ "$VRAM_GB" -lt "$BLACKWELL_MIN_VRAM" ]; then
        log_fatal "Blackwell GPUs require ${BLACKWELL_MIN_VRAM}GB+ VRAM for PyTorch backend."
    fi

    # Set service mode based on GPU architecture
    if [ "$IS_BLACKWELL" = true ]; then
        SERVICE_MODE="pytorch"
        log_info "Mode: ${BLUE}PyTorch${NC} (Blackwell GPU)"
    else
        SERVICE_MODE="gguf"
        log_info "Mode: ${BLUE}GGUF${NC} (stable-diffusion.cpp)"
    fi

    log_success "GPU validated"
}

# Determine which models to install based on VRAM
detect_model_support() {
    log_step 3.5 "Auto-detecting supported models based on VRAM..."

    # Multi-model support flags (exported for use by other modules)
    INSTALL_BOTH_MODELS=false
    INSTALL_ZIMAGE=false
    INSTALL_FLUX=false

    if [ "$VRAM_GB" -ge "$MULTI_MODEL_THRESHOLD" ]; then
        log_info "${GREEN}${VRAM_GB}GB+ VRAM detected - enabling both models (multi-model mode)${NC}"
        IMAGE_MODEL="z-image-turbo"  # Primary model (loaded first)
        INSTALL_BOTH_MODELS=true
        INSTALL_ZIMAGE=true
        INSTALL_FLUX=true
    elif [ "$VRAM_GB" -ge "$MIN_VRAM_REQUIRED" ]; then
        log_info "${YELLOW}${VRAM_GB}GB VRAM detected - enabling Z-Image-Turbo only${NC}"
        IMAGE_MODEL="z-image-turbo"
        INSTALL_ZIMAGE=true
    else
        log_fatal "Minimum ${MIN_VRAM_REQUIRED}GB VRAM required. Your GPU has ${VRAM_GB}GB."
    fi

    if [ "$INSTALL_BOTH_MODELS" = true ]; then
        log_info "Selected models: ${GREEN}z-image-turbo${NC} + ${BLUE}flux-schnell${NC} (multi-model mode)"
        log_info "${CYAN}  Worker will dynamically switch between models based on job requirements${NC}"
    else
        log_info "Selected model: ${GREEN}$IMAGE_MODEL${NC}"
    fi
}

# Check if video generation is supported (12GB+ VRAM with PyTorch)
is_video_capable() {
    [ "$VRAM_GB" -ge "$MULTI_MODEL_THRESHOLD" ] && [ "$SERVICE_MODE" = "pytorch" ]
}

# Export GPU info for use by other scripts
export_gpu_info() {
    export GPU_NAME VRAM_MB VRAM_GB COMPUTE_CAP IS_BLACKWELL SERVICE_MODE
    export INSTALL_BOTH_MODELS INSTALL_ZIMAGE INSTALL_FLUX IMAGE_MODEL
}
