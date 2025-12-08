#!/bin/bash
#
# Power Node Installation Script
# https://github.com/Gelotto/power-node
#
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

INSTALL_DIR="${POWER_NODE_DIR:-$HOME/.power-node}"
API_URL="${API_URL:-https://api.gen.gelotto.io}"
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

# Try to download pre-built binary
BINARY_URL="https://github.com/$GITHUB_REPO/releases/latest/download/power-node-${OS}-${ARCH}"
if curl -sSL --head "$BINARY_URL" 2>/dev/null | grep -q "200"; then
    echo "  Downloading pre-built binary..."
    curl -sSL "$BINARY_URL" -o "$INSTALL_DIR/bin/power-node"
    chmod +x "$INSTALL_DIR/bin/power-node"
else
    echo "  Pre-built binary not available, building from source..."

    if ! command -v go &> /dev/null; then
        echo -e "${RED}ERROR: Go not found. Please install Go 1.21+${NC}"
        echo "  https://go.dev/doc/install"
        exit 1
    fi

    TEMP_DIR=$(mktemp -d)
    git clone --quiet "https://github.com/$GITHUB_REPO.git" "$TEMP_DIR/power-node"
    cd "$TEMP_DIR/power-node"
    go build -o "$INSTALL_DIR/bin/power-node" ./cmd/power-node
    rm -rf "$TEMP_DIR"
fi

echo -e "${GREEN}  ✓ Binary installed${NC}"

# =============================================================================
# Setup Python Environment
# =============================================================================
echo -e "${YELLOW}[6/7] Setting up Python environment...${NC}"

if [ ! -d "$INSTALL_DIR/venv" ]; then
    python3 -m venv "$INSTALL_DIR/venv"
fi

source "$INSTALL_DIR/venv/bin/activate"
pip install --upgrade pip --quiet

if [ "$SERVICE_MODE" = "gguf" ]; then
    echo "  Installing stable-diffusion-cpp-python..."
    echo "  (This may take several minutes for compilation)"

    if command -v nvcc &> /dev/null; then
        CMAKE_ARGS="-DSD_CUDA=ON" pip install 'stable-diffusion-cpp-python>=0.4.0' --quiet 2>&1 | grep -E "(error|Error)" || true
    else
        pip install 'stable-diffusion-cpp-python>=0.4.0' --quiet
    fi
    pip install pillow --quiet
else
    echo "  Installing PyTorch and dependencies..."
    pip install torch torchvision --quiet
    pip install transformers safetensors accelerate tqdm pillow --quiet
fi

deactivate
echo -e "${GREEN}  ✓ Python environment ready${NC}"

# =============================================================================
# Download Models
# =============================================================================
echo -e "${YELLOW}[7/7] Downloading models...${NC}"

download_file() {
    local url="$1"
    local dest="$2"
    local name="$3"

    if [ -f "$dest" ]; then
        echo -e "  ${GREEN}✓${NC} $name (cached)"
        return 0
    fi

    echo "  Downloading $name..."
    curl -L "$url" -o "$dest" --progress-bar
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

        images = self.sd.generate_image(
            prompt=prompt,
            negative_prompt="",
            width=width,
            height=height,
            cfg_scale=1.0,
            guidance=3.5,
            sample_steps=steps,
            seed=seed,
            batch_count=1,
            vae_tiling=self.vram_gb < 12,
        )

        if not images:
            raise RuntimeError("No image generated")

        buf = io.BytesIO()
        images[0].save(buf, format='PNG')
        return {"image_data": base64.b64encode(buf.getvalue()).decode(), "format": "png"}

    def handle_request(self, req):
        try:
            if req.get("method") == "generate":
                return {"id": req.get("id", 0), "result": self.generate(**req.get("params", {})), "error": None}
            return {"id": req.get("id", 0), "result": None, "error": f"Unknown method: {req.get('method')}"}
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
fi

chmod +x "$INSTALL_DIR/scripts/inference.py"

# =============================================================================
# Create Configuration
# =============================================================================
if [ "$SERVICE_MODE" = "gguf" ]; then
cat > "$INSTALL_DIR/config/config.yaml" << EOF
# Power Node Configuration
# Register at https://gen.gelotto.io/workers/register to get your credentials

api:
  url: $API_URL
  key: ""  # Your API key here

model:
  service_mode: gguf
  vram_gb: $VRAM_GB

worker:
  id: ""  # Your worker ID here
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
EOF
fi

# =============================================================================
# Create Start Script
# =============================================================================
cat > "$INSTALL_DIR/start.sh" << EOF
#!/bin/bash
cd "$INSTALL_DIR"
source venv/bin/activate
./bin/power-node -config "$INSTALL_DIR/config/config.yaml"
EOF
chmod +x "$INSTALL_DIR/start.sh"

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
echo ""
echo -e "${CYAN}Next steps:${NC}"
echo ""
echo "  1. Register at https://gen.gelotto.io/workers/register"
echo ""
echo "  2. Add your credentials to $INSTALL_DIR/config/config.yaml:"
echo "     api:"
echo "       key: \"YOUR_API_KEY\""
echo "     worker:"
echo "       id: \"YOUR_WORKER_ID\""
echo ""
echo "  3. Start the node:"
echo "     $INSTALL_DIR/start.sh"
echo ""
echo -e "${GREEN}Happy computing!${NC}"
