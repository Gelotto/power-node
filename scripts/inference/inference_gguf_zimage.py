#!/usr/bin/env python3
"""
Power Node Inference Service (GGUF/stable-diffusion.cpp)
Z-Image-Turbo model with GGUF quantization for 8-16GB GPUs
"""

import sys
import json
import base64
import io
import os
import argparse
import random
import time
import threading

# Import shared utilities (security constants and common functions)
from shared_utils import (
    ALLOWED_HOSTS, MAX_DOWNLOAD_SIZE, FACESWAP_IDLE_TIMEOUT,
    validate_image_url, download_with_limit, detect_vram_gb
)


class InferenceService:
    def __init__(self, diffusion_path, vae_path, text_encoder_path, vram_gb=None):
        self.diffusion_path = diffusion_path
        self.vae_path = vae_path
        self.text_encoder_path = text_encoder_path
        self.vram_gb = vram_gb or detect_vram_gb()
        self.sd = None
        # Face-swap model lifecycle tracking
        self._face_swapper = None
        self._faceswap_funcs = None
        self._last_faceswap_time = 0
        self._faceswap_lock = threading.Lock()  # Protects face-swap model access
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Start background thread to unload idle face-swap models."""
        def cleanup_loop():
            while True:
                time.sleep(60)  # Check every minute
                with self._faceswap_lock:
                    if self._face_swapper and time.time() - self._last_faceswap_time > FACESWAP_IDLE_TIMEOUT:
                        sys.stderr.write("Unloading idle face-swap models (5 min timeout)...\n")
                        sys.stderr.flush()
                        self._face_swapper = None
                        self._faceswap_funcs = None
                        import gc
                        gc.collect()

        t = threading.Thread(target=cleanup_loop, daemon=True)
        t.start()

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

    def generate(self, prompt, width=1024, height=1024, steps=8, seed=-1, negative_prompt=None):
        # negative_prompt accepted but ignored (Z-Image-Turbo is a distilled model)
        sys.stderr.write(f"Generating: {prompt[:50]}...\n")
        sys.stderr.flush()

        if seed == -1:
            seed = random.randint(0, 2**31 - 1)

        # Z-Image-Turbo with stable-diffusion.cpp uses cfg_scale=1.0
        # (Note: HuggingFace diffusers uses guidance_scale=0.0, but sd.cpp is different!)
        images = self.sd.generate_image(
            prompt=prompt,
            negative_prompt="",  # Z-Image-Turbo doesn't use negative prompts (ignored)
            width=width,
            height=height,
            cfg_scale=1.0,       # stable-diffusion.cpp uses 1.0 for Z-Image-Turbo
            guidance=1.0,        # stable-diffusion.cpp uses 1.0
            sample_steps=steps,
            seed=seed,
            batch_count=1,
            vae_tiling=True,  # Always enable - Z-Image VAE needs ~27GB for 1024x1024 without tiling
        )

        if not images:
            raise RuntimeError("No image generated")

        buf = io.BytesIO()
        images[0].save(buf, format='PNG')
        return {"image_data": base64.b64encode(buf.getvalue()).decode(), "format": "png"}

    def generate_video(self, prompt, width=832, height=480, duration_seconds=5, fps=24, total_frames=120, seed=-1, negative_prompt=None):
        """Generate a video using Wan2GP (if available)"""
        # Video generation requires Wan2GP model which is not included in GGUF mode
        # Workers need to install video support separately
        raise NotImplementedError(
            "Video generation not available in GGUF mode. "
            "Video requires Wan2GP model installation. "
            "Please use PyTorch mode with video support enabled."
        )

    def face_swap(self, source_image_url, target_image_url, is_gif=False, swap_all_faces=True, enhance=True, max_frames=100):
        """Perform face swap on images or GIFs with security validation."""
        # Validate URLs (SSRF protection) - outside lock
        source_image_url = validate_image_url(source_image_url)
        target_image_url = validate_image_url(target_image_url)

        # Acquire lock for model check/load and timestamp update
        with self._faceswap_lock:
            self._last_faceswap_time = time.time()

            # Lazy load face swapper
            if self._face_swapper is None:
                model_path = os.environ.get("FACESWAP_MODEL_PATH", "")
                if not model_path:
                    raise ValueError("Face swap not available: FACESWAP_MODEL_PATH not set")

                sys.stderr.write(f"Loading face swap models from {model_path}...\n")
                sys.stderr.flush()

                # Add scripts directory to path for faceswap module
                scripts_dir = os.path.dirname(os.path.abspath(__file__))
                if scripts_dir not in sys.path:
                    sys.path.insert(0, scripts_dir)

                from faceswap import FaceSwapper, load_image_from_bytes, image_to_bytes, GifProcessor
                self._face_swapper = FaceSwapper(model_dir=model_path, enable_enhancement=enhance)
                self._faceswap_funcs = {
                    'load_image_from_bytes': load_image_from_bytes,
                    'image_to_bytes': image_to_bytes,
                    'GifProcessor': GifProcessor,
                }
                sys.stderr.write("Face swap models loaded.\n")
                sys.stderr.flush()

            # Get references while holding lock (prevents cleanup during processing)
            swapper = self._face_swapper
            funcs = self._faceswap_funcs

        # Processing happens outside lock to avoid blocking other requests
        sys.stderr.write(f"Face swap: is_gif={is_gif}, swap_all={swap_all_faces}, enhance={enhance}\n")
        sys.stderr.flush()

        # Download source image with size limit (DoS protection)
        source_data = download_with_limit(source_image_url)
        source_img = funcs['load_image_from_bytes'](source_data)

        # Download target with size limit
        target_data = download_with_limit(target_image_url)

        if is_gif:
            # Process GIF frame by frame
            processor = funcs['GifProcessor'](
                swapper,
                enhancer=swapper.enhancer if enhance else None
            )
            result_bytes = processor.process_gif(source_img, target_data, max_frames=max_frames, enhance=enhance)
            return {
                "image_data": base64.b64encode(result_bytes).decode(),
                "format": "gif",
                "frames_swapped": min(max_frames, 100)  # Approximate
            }
        else:
            # Process single image
            target_img = funcs['load_image_from_bytes'](target_data)
            result_img = swapper.swap(source_img, target_img, swap_all=swap_all_faces, enhance=enhance)
            result_bytes = funcs['image_to_bytes'](result_img, ".jpg", quality=95)
            return {
                "image_data": base64.b64encode(result_bytes).decode(),
                "format": "jpeg",
                "frames_swapped": 1
            }

    def handle_request(self, req):
        try:
            method = req.get("method")
            if method == "generate":
                return {"id": req.get("id", 0), "result": self.generate(**req.get("params", {})), "error": None}
            elif method == "generate_video":
                return {"id": req.get("id", 0), "result": self.generate_video(**req.get("params", {})), "error": None}
            elif method == "face_swap":
                return {"id": req.get("id", 0), "result": self.face_swap(**req.get("params", {})), "error": None}
            return {"id": req.get("id", 0), "result": None, "error": f"Unknown method: {method}"}
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
