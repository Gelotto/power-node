#!/bin/bash
# =============================================================================
# system.sh - Package manager detection and system dependency installation
# =============================================================================

# Detect package manager
detect_package_manager() {
    if command_exists apt-get; then
        echo "apt"
    elif command_exists dnf; then
        echo "dnf"
    elif command_exists pacman; then
        echo "pacman"
    elif command_exists zypper; then
        echo "zypper"
    else
        echo "unknown"
    fi
}

# Install a package using the appropriate package manager
install_package() {
    local pkg="$1"
    local pkg_manager=$(detect_package_manager)

    case "$pkg_manager" in
        apt)
            sudo apt-get update -qq && sudo apt-get install -y "$pkg"
            ;;
        dnf)
            sudo dnf install -y "$pkg"
            ;;
        pacman)
            sudo pacman -S --noconfirm "$pkg"
            ;;
        zypper)
            sudo zypper install -y "$pkg"
            ;;
        *)
            log_error "No supported package manager found"
            return 1
            ;;
    esac
}

# Install multiple packages
install_packages() {
    local pkg_manager=$(detect_package_manager)

    case "$pkg_manager" in
        apt)
            sudo apt-get update -qq && sudo apt-get install -y "$@"
            ;;
        dnf)
            sudo dnf install -y "$@"
            ;;
        pacman)
            sudo pacman -S --noconfirm "$@"
            ;;
        zypper)
            sudo zypper install -y "$@"
            ;;
        *)
            log_error "No supported package manager found"
            return 1
            ;;
    esac
}

# Check and install ffmpeg if missing
ensure_ffmpeg() {
    if ! command_exists ffmpeg; then
        log_warning "ffmpeg not found. Installing..."
        if install_package ffmpeg; then
            log_success "ffmpeg installed"
        else
            log_warning "Could not install ffmpeg automatically."
            log_info "Video generation will require ffmpeg. Install manually."
        fi
    fi
}

# Check and install python3-venv if missing
ensure_python_venv() {
    if ! python3 -c "import venv" 2>/dev/null; then
        log_warning "Python venv module not found. Installing..."

        local pkg_manager=$(detect_package_manager)
        local pkg_name="python3-venv"

        # pacman uses different package name
        if [ "$pkg_manager" = "pacman" ]; then
            pkg_name="python"
        fi

        if install_package "$pkg_name"; then
            # Verify it worked
            if python3 -c "import venv" 2>/dev/null; then
                log_success "python3-venv installed"
                return 0
            fi
        fi

        log_fatal "Failed to install python3-venv. Please install it manually and re-run this script."
    fi
}

# Check and install build tools for GGUF mode
ensure_build_tools() {
    if ! command_exists gcc || ! command_exists g++; then
        log_warning "Build tools (gcc/g++) not found. Installing..."

        local pkg_manager=$(detect_package_manager)
        case "$pkg_manager" in
            apt)
                install_packages build-essential cmake
                ;;
            dnf)
                install_packages gcc gcc-c++ cmake
                ;;
            pacman)
                install_packages base-devel cmake
                ;;
            *)
                log_fatal "Could not auto-install build tools. Please install gcc, g++, and cmake manually."
                ;;
        esac
        log_success "Build tools installed"
    fi
}

# Check and install CUDA toolkit if nvcc missing
ensure_cuda_toolkit() {
    if ! command_exists nvcc; then
        log_warning "CUDA toolkit (nvcc) not found. Installing..."
        local cuda_installed=false

        local pkg_manager=$(detect_package_manager)
        case "$pkg_manager" in
            apt)
                if sudo apt-get update -qq && sudo apt-get install -y nvidia-cuda-toolkit; then
                    cuda_installed=true
                fi
                ;;
            dnf)
                if sudo dnf install -y cuda-toolkit; then
                    cuda_installed=true
                fi
                ;;
        esac

        # Reload PATH to pick up nvcc
        export PATH="/usr/local/cuda/bin:$PATH"

        if command_exists nvcc; then
            log_success "CUDA toolkit installed"
        elif [ "$cuda_installed" = true ]; then
            log_warning "CUDA installed but nvcc not in PATH"
            log_info "You may need to add /usr/local/cuda/bin to your PATH"
        else
            log_warning "Could not install CUDA toolkit automatically"
            log_info "Install manually: sudo apt install nvidia-cuda-toolkit"
            log_info "Continuing without CUDA (will use CPU-only mode)"
        fi
    fi
}

# Verify Python version
check_python_version() {
    local python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    local python_major=$(echo "$python_version" | cut -d. -f1)
    local python_minor=$(echo "$python_version" | cut -d. -f2)

    if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 10 ]); then
        log_fatal "Python 3.10+ required. Found: $python_version"
    fi

    echo "$python_version"
}

# Check all system requirements
check_system_requirements() {
    log_step 1 "Checking system requirements..."

    require_command nvidia-smi "nvidia-smi not found. Please install NVIDIA drivers first."
    require_command python3 "Python 3 not found."
    require_command curl "curl not found."

    local python_version=$(check_python_version)

    ensure_ffmpeg
    ensure_python_venv

    log_success "All system requirements met (Python $python_version)"
}
