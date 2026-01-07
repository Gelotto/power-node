#!/usr/bin/env python3
"""
Power Node Inference Service (FLUX GGUF/stable-diffusion.cpp)
FLUX.1-schnell model with GGUF quantization for 10-16GB GPUs
"""

import sys
import json
import base64
import io
import os
import argparse

from shared_utils import detect_vram_gb


class FLUXInferenceService:
    """FLUX.1-schnell inference via stable-diffusion.cpp GGUF"""

    def __init__(self, diffusion_path, clip_path, t5_path, vae_path, vram_gb=None):
        self.diffusion_path = diffusion_path
        self.clip_path = clip_path
        self.t5_path = t5_path
        self.vae_path = vae_path
        self.vram_gb = vram_gb or detect_vram_gb()
        self.sd = None

    def initialize(self):
        from stable_diffusion_cpp import StableDiffusion

        sys.stderr.write("=== Power Node FLUX Inference (GGUF) ===\n")
        sys.stderr.write(f"VRAM: {self.vram_gb}GB\n")
        sys.stderr.flush()

        for path, name in [
            (self.diffusion_path, "Diffusion"),
            (self.clip_path, "CLIP-L"),
            (self.t5_path, "T5-XXL"),
            (self.vae_path, "VAE")
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found: {path}")

        # Memory optimization based on VRAM
        offload = self.vram_gb < 14
        keep_vae_cpu = self.vram_gb < 16
        use_flash = self.vram_gb >= 10

        sys.stderr.write("Loading FLUX model...\n")
        sys.stderr.flush()

        self.sd = StableDiffusion(
            diffusion_model_path=self.diffusion_path,
            clip_l_path=self.clip_path,
            t5xxl_path=self.t5_path,
            vae_path=self.vae_path,
            vae_decode_only=True,
            n_threads=-1,
            wtype="default",
            rng_type="cuda",
            offload_params_to_cpu=offload,
            keep_clip_on_cpu=offload,
            keep_vae_on_cpu=keep_vae_cpu,
            diffusion_flash_attn=use_flash,
            verbose=True,
        )

        sys.stderr.write("FLUX model ready.\n")
        sys.stderr.flush()

    def generate(self, prompt, width, height, steps, seed=None, negative_prompt=None):
        """Generate image with FLUX.1-schnell"""
        if seed is None:
            import random
            seed = random.randint(0, 2**31 - 1)

        # FLUX.1-schnell uses 1-4 steps max
        steps = min(steps, 4)

        sys.stderr.write(f"Generating: {width}x{height}, {steps} steps, seed={seed}\n")
        sys.stderr.flush()

        images = self.sd.txt_to_img(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            width=width,
            height=height,
            sample_steps=steps,
            cfg_scale=1.0,  # FLUX uses minimal CFG
            seed=seed,
        )

        if not images:
            raise RuntimeError("No image generated")

        img = images[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
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

            print(json.dumps(response), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion", required=True, help="Path to FLUX GGUF diffusion model")
    parser.add_argument("--clip", required=True, help="Path to CLIP-L encoder")
    parser.add_argument("--t5", required=True, help="Path to T5-XXL encoder")
    parser.add_argument("--vae", required=True, help="Path to VAE")
    parser.add_argument("--vram", type=int, help="VRAM in GB")
    args = parser.parse_args()

    svc = FLUXInferenceService(args.diffusion, args.clip, args.t5, args.vae, args.vram)
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
