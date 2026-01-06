# Power Node

Distributed GPU compute node for the Gelotto AI image generation network.

Power Node lets you contribute GPU compute power to generate AI images and earn rewards. It connects to the Gelotto network, claims generation jobs, and uses your NVIDIA GPU to create images.

## Requirements

- **NVIDIA GPU** with 8GB+ VRAM (RTX 3060, RTX 4060, or newer)
- **Linux** with NVIDIA drivers installed
- **Python 3.10+**
- **CUDA toolkit** (for GGUF mode, to build stable-diffusion.cpp)
- **Disk space:**
  - GGUF mode: ~15GB (base) + ~500MB (face-swap)
  - PyTorch mode: ~40GB (base) + ~29GB (video, optional) + ~500MB (face-swap)

### Supported GPUs

| GPU | VRAM | Mode | Video | Disk Space | Performance |
|-----|------|------|-------|------------|-------------|
| RTX 3060 | 12GB | GGUF | No | ~15GB | Good |
| RTX 4060 | 8GB | GGUF | No | ~10GB | Good |
| RTX 4070/Ti | 12GB | GGUF | No | ~15GB | Better |
| RTX 4080 | 16GB | GGUF | No | ~15GB | Better |
| RTX 4090 | 24GB | GGUF | No | ~15GB | Best |
| RTX 5070 Ti | 16GB | PyTorch | Yes | ~70GB | Best |
| RTX 5080 | 16GB | PyTorch | Yes | ~70GB | Best |
| RTX 5090 | 32GB | PyTorch | Yes | ~70GB | Best |

**Notes:**
- RTX 50-series (Blackwell) GPUs require PyTorch mode due to CUDA compute capability 12.0
- Video generation requires PyTorch mode with 12GB+ VRAM (automatically detected)

## Quick Start

### 1. Register Your Worker

Visit [https://gelotto.io/workers](https://gelotto.io/workers) and register with a unique hostname.

You'll receive:
- **Worker ID** - Your unique identifier
- **API Key** - Your authentication token (save this securely!)

### 2. Install Power Node

**Option A: One-line installer** (recommended)
```bash
curl -sSL https://raw.githubusercontent.com/Gelotto/power-node/main/install.sh | bash
```

**Option B: Clone and install**
```bash
git clone https://github.com/Gelotto/power-node.git
cd power-node
./install.sh
```

The installer will:
1. Detect your GPU and VRAM
2. Download the appropriate model (~9GB for GGUF, ~31GB for PyTorch)
3. **Auto-download optional models** based on GPU capability:
   - **Video generation** (Wan2.1, ~29GB) - PyTorch mode with 12GB+ VRAM
   - **Face-swap** (~500MB) - Any mode with 6GB+ VRAM
4. Set up Python environment and dependencies
5. Create configuration files

**Installation time:** 10-30 minutes depending on internet speed (longer if video model is downloaded)

#### Installation Options

Skip optional models with these flags:

```bash
# Skip video model only
curl -sSL ... | bash -s -- --no-video

# Skip face-swap only
curl -sSL ... | bash -s -- --no-faceswap

# Skip all optional models (minimal install)
curl -sSL ... | bash -s -- --minimal
```

The installer will warn if disk space is below 50GB when downloading the video model.

### 3. Configure

Edit `~/.power-node/config/config.yaml` and add your credentials:

```yaml
api:
  key: "YOUR_API_KEY_HERE"

worker:
  id: "YOUR_WORKER_ID_HERE"
```

### 4. Start

```bash
~/.power-node/start.sh
```

You should see:
```
Starting worker...
Using credentials - Worker ID: abc123...
API Key: wk_xxxxxxxxxx...
Starting Python inference service...
Worker fully initialized!
Starting job loop (polling every 5s)...
```

### 5. Run as System Service (Optional)

To start automatically on boot:

```bash
sudo cp ~/.power-node/power-node.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable power-node
sudo systemctl start power-node
```

Check status:
```bash
sudo systemctl status power-node
sudo journalctl -u power-node -f
```

## How It Works

1. **Register** - Create an account at gelotto.io/workers
2. **Connect** - Power Node connects to the Gelotto API
3. **Claim Jobs** - The node polls for available generation jobs
4. **Generate** - Your GPU generates images using the Z-Image model
5. **Upload** - Results are uploaded back to the network
6. **Earn** - Receive rewards for completed jobs

## Configuration Reference

```yaml
api:
  url: https://api.gelotto.io      # API endpoint
  key: ""                            # Your API key

model:
  service_mode: gguf                 # gguf or pytorch (auto-detected)
  vram_gb: 8                         # Your GPU VRAM

worker:
  id: ""                             # Your worker ID
  hostname: ""                       # Auto-detected if empty
  gpu_info: ""                       # Auto-detected if empty
  poll_interval: 5s                  # Job polling interval
  heartbeat_interval: 30s            # Heartbeat interval

python:
  executable: ~/.power-node/venv/bin/python3
  script_path: ~/.power-node/scripts/inference.py
  script_args: []                    # Model paths (auto-configured)
  env:
    PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
```

## Troubleshooting

### GPU not detected

Ensure NVIDIA drivers are installed:
```bash
nvidia-smi
```

### Out of memory errors

The node automatically configures for your GPU's VRAM. For 8GB GPUs:
- Uses quantized Q4_0 models
- Enables VAE tiling
- Offloads components to CPU

If you still get OOM errors, try reducing resolution or closing other GPU applications.

### Connection errors

1. Check your API key and worker ID are correct in config.yaml
2. Verify network connectivity: `curl https://api.gelotto.io/health`
3. Check if the API is reachable from your network

### Model download failed

If model download was interrupted:
```bash
rm -rf ~/.power-node/models
~/.power-node/install.sh
```

### Python errors

Check Python environment:
```bash
source ~/.power-node/venv/bin/activate
python3 -c "import torch; print(torch.cuda.is_available())"
```

## Building from Source

Requires Go 1.21+:

```bash
git clone https://github.com/Gelotto/power-node.git
cd power-node
go build -o bin/power-node ./cmd/power-node
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POWER_NODE_DIR` | Installation directory | `~/.power-node` |
| `API_URL` | Backend API URL | `https://api.gelotto.io` |
| `HF_TOKEN` | HuggingFace token for FLUX model (PyTorch mode only) | - |

## Multi-Model Support

With 12GB+ VRAM, the installer automatically configures **multi-model mode** with both:
- **Z-Image-Turbo** - Fast generation, good quality
- **FLUX.1-schnell** - Excellent prompt understanding, ultra-fast

The worker dynamically switches between models based on job requirements.

### FLUX.1-schnell (PyTorch/Blackwell GPUs)

FLUX.1-schnell is a **gated model** on HuggingFace requiring authentication. This only affects PyTorch mode (RTX 50-series Blackwell GPUs). GGUF mode uses an ungated source.

**To enable FLUX on Blackwell GPUs:**

1. **Accept the license**: https://huggingface.co/black-forest-labs/FLUX.1-schnell
2. **Create an access token**: https://huggingface.co/settings/tokens
   - Token type: **Read** (simplest option)
   - Or Fine-grained with: "Read access to contents of all public gated repos you can access"
3. **Run installer with token**:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxx curl -sSL https://raw.githubusercontent.com/Gelotto/power-node/main/install.sh | bash
```

Or set it in your environment first:
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx
curl -sSL https://raw.githubusercontent.com/Gelotto/power-node/main/install.sh | bash
```

**Without HF_TOKEN**: The installer will skip FLUX and install Z-Image-Turbo only. You can re-run with the token later to add FLUX.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [Gelotto Network](https://gelotto.io)
- [PicShapes AI Generator](https://picshapes.com)
- [Worker Registration](https://gelotto.io/workers)
- [Discord](https://discord.gg/gelotto)
