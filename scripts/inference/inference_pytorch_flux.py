#!/usr/bin/env python3
"""
Power Node Inference Service (FLUX PyTorch)
FLUX.1-schnell for Blackwell GPUs (RTX 50-series) with 16GB+ VRAM
"""

import sys
import json
import base64
import io
import os
import argparse
import gc

from shared_utils import detect_vram_gb


class FLUXInferenceService:
    """FLUX.1-schnell inference via diffusers FluxPipeline"""

    def __init__(self, model_path, vram_gb=None):
        self.model_path = model_path
        self.vram_gb = vram_gb or detect_vram_gb()
        self.pipe = None

    def initialize(self):
        import torch
        from diffusers import FluxPipeline

        sys.stderr.write("=== Power Node FLUX Inference (PyTorch) ===\n")
        sys.stderr.write(f"VRAM: {self.vram_gb}GB\n")
        sys.stderr.flush()

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        sys.stderr.write("Loading FLUX.1-schnell...\n")
        sys.stderr.flush()

        self.pipe = FluxPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        )

        # Memory optimization based on VRAM
        if self.vram_gb < 20:
            # Sequential offload moves individual layers to GPU one at a time
            # This uses much less VRAM than model_cpu_offload (which keeps whole model on GPU)
            sys.stderr.write("Enabling sequential CPU offload for low VRAM (<20GB)\n")
            self.pipe.enable_sequential_cpu_offload()
            # VAE optimizations further reduce memory during decode
            self.pipe.vae.enable_tiling()
            self.pipe.vae.enable_slicing()
            sys.stderr.write("VAE tiling and slicing enabled\n")
        else:
            self.pipe = self.pipe.to("cuda")

        sys.stderr.write("FLUX.1-schnell ready.\n")
        sys.stderr.flush()

    def generate(self, prompt, width, height, steps, seed=None, negative_prompt=None):
        """Generate image with FLUX.1-schnell"""
        import torch

        generator = None
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(seed)
        else:
            import random
            seed = random.randint(0, 2**31 - 1)
            generator = torch.Generator("cpu").manual_seed(seed)

        # FLUX.1-schnell optimal: 1-4 steps
        steps = min(steps, 4)

        sys.stderr.write(f"Generating: {width}x{height}, {steps} steps, seed={seed}\n")
        sys.stderr.flush()

        # FLUX.1-schnell specific parameters
        result = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=0.0,  # CRITICAL: schnell is distilled, no CFG
            max_sequence_length=256,
            generator=generator,
        )

        image = result.images[0]

        # Convert to base64 PNG
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8"), "png"

    def handle_request(self, req):
        method = req.get("method")
        params = req.get("params", {})

        if method == "generate":
            img_data, fmt = self.generate(
                prompt=params.get("prompt", ""),
                width=params.get("width", 1024),
                height=params.get("height", 1024),
                steps=params.get("steps", 4),
                seed=params.get("seed"),
                negative_prompt=params.get("negative_prompt"),
            )
            return {"image_data": img_data, "format": fmt}
        elif method == "generate_video":
            raise ValueError("Video generation not supported with FLUX.1-schnell")
        elif method == "face_swap":
            raise ValueError("Face swap not supported with FLUX.1-schnell")
        else:
            raise ValueError(f"Unknown method: {method}")

    def run(self):
        """Main JSON-RPC loop"""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                req = json.loads(line)
                req_id = req.get("id")
                result = self.handle_request(req)
                response = {"id": req_id, "result": result, "error": None}
            except Exception as e:
                sys.stderr.write(f"Error: {e}\n")
                sys.stderr.flush()
                response = {"id": req.get("id"), "result": None, "error": str(e)}

            # Clean up VRAM after each request
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass

            print(json.dumps(response), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to FLUX model")
    parser.add_argument("--vram", type=int, help="VRAM in GB")
    args = parser.parse_args()

    # Fallback to environment variable or default path
    model_path = args.model or os.environ.get("FLUX_MODEL_PATH")
    if not model_path:
        model_path = os.path.expanduser("~/.power-node/models/FLUX.1-schnell")

    svc = FLUXInferenceService(model_path, args.vram)
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
