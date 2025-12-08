# Power Node

Distributed GPU compute node for the Gelotto AI image generation network.

Power Node lets you contribute GPU compute power to generate AI images and earn rewards. It connects to the Gelotto network, claims generation jobs, and uses your NVIDIA GPU to create images.

## Requirements

- **NVIDIA GPU** with 8GB+ VRAM (RTX 3060, RTX 4060, or newer)
- **Linux** with NVIDIA drivers installed
- **Python 3.10+**
- **CUDA toolkit** (for building stable-diffusion.cpp)

### Supported GPUs

| GPU | VRAM | Mode | Performance |
|-----|------|------|-------------|
| RTX 3060 | 12GB | GGUF | Good |
| RTX 4060 | 8GB | GGUF | Good |
| RTX 4070/Ti | 12GB | GGUF | Better |
| RTX 4080 | 16GB | GGUF | Better |
| RTX 4090 | 24GB | GGUF | Best |
| RTX 5070 Ti+ | 16GB+ | PyTorch | Best |

## Quick Start

### 1. Register Your Worker

Visit [gen.gelotto.io/workers/register](https://gen.gelotto.io/workers/register) and register with a unique hostname.

You'll receive:
- **Worker ID** - Your unique identifier
- **API Key** - Your authentication token (save this!)

### 2. Install Power Node

```bash
curl -sSL https://get.power.gelotto.io | bash
```

Or install manually:

```bash
git clone https://github.com/Gelotto/power-node.git
cd power-node
./install.sh
```

### 3. Configure

Edit `~/.power-node/config/config.yaml`:

```yaml
api:
  url: https://api.gen.gelotto.io
  key: "YOUR_API_KEY_HERE"

worker:
  id: "YOUR_WORKER_ID_HERE"
  hostname: "your-hostname"
```

### 4. Start

```bash
~/.power-node/start.sh
```

Or run as a system service:

```bash
sudo cp ~/.power-node/power-node.service /etc/systemd/system/
sudo systemctl enable power-node
sudo systemctl start power-node
```

## How It Works

1. **Register** - Create an account at gen.gelotto.io
2. **Connect** - Power Node connects to the Gelotto API
3. **Claim Jobs** - The node polls for available generation jobs
4. **Generate** - Your GPU generates images using the Z-Image model
5. **Upload** - Results are uploaded back to the network
6. **Earn** - Receive rewards for completed jobs

## Configuration Reference

```yaml
api:
  url: https://api.gen.gelotto.io   # API endpoint
  key: ""                            # Your API key

model:
  service_mode: gguf                 # gguf or pytorch
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
  script_args: []                    # Model paths, etc.
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

### Connection errors

Check your API key and worker ID are correct in config.yaml.

## Building from Source

```bash
git clone https://github.com/Gelotto/power-node.git
cd power-node
go build -o bin/power-node ./cmd/power-node
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [Gelotto AI Generator](https://gen.gelotto.io)
- [Worker Dashboard](https://gen.gelotto.io/workers)
- [Discord](https://discord.gg/gelotto)
