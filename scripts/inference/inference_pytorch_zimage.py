#!/usr/bin/env python3
"""
Power Node Inference Service (PyTorch/Z-Image-Turbo)
For Blackwell GPUs (RTX 50-series) with 14GB+ VRAM
Supports: Image generation, Video generation (Wan2.1), Face-swap
"""

import sys
import json
import base64
import io
import os
import argparse
import random
import gc
import time
import threading

# Import shared utilities (security constants and common functions)
from shared_utils import (
    ALLOWED_HOSTS, MAX_DOWNLOAD_SIZE, FACESWAP_IDLE_TIMEOUT,
    validate_image_url, download_with_limit, detect_vram_gb
)


class InferenceService:
    def __init__(self, model_path, vae_path=None):
        self.model_path = model_path
        self.vae_path = vae_path
        self.pipe = None
        # Face-swap model lifecycle tracking
        self._face_swapper = None
        self._faceswap_funcs = None
        self._last_faceswap_time = 0
        self._faceswap_lock = threading.Lock()  # Protects face-swap model access
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Start background thread to unload idle face-swap models."""
        def cleanup_loop():
            import torch
            while True:
                time.sleep(60)  # Check every minute
                with self._faceswap_lock:
                    if self._face_swapper and time.time() - self._last_faceswap_time > FACESWAP_IDLE_TIMEOUT:
                        sys.stderr.write("Unloading idle face-swap models (5 min timeout)...\n")
                        sys.stderr.flush()
                        self._face_swapper = None
                        self._faceswap_funcs = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        t = threading.Thread(target=cleanup_loop, daemon=True)
        t.start()

    def initialize(self):
        import torch
        from diffusers import ZImagePipeline

        sys.stderr.write("=== Power Node Inference Service (PyTorch) ===\n")
        sys.stderr.write(f"Model: {self.model_path}\n")
        sys.stderr.flush()

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        sys.stderr.write("Loading model (this may take a minute)...\n")
        sys.stderr.flush()

        # Load Z-Image-Turbo pipeline with optimizations for 16GB VRAM
        self.pipe = ZImagePipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        )

        # Enable memory optimizations
        self.pipe.enable_model_cpu_offload()

        sys.stderr.write("Ready.\n")
        sys.stderr.flush()

    def generate(self, prompt, width=1024, height=1024, steps=8, seed=-1, negative_prompt=None):
        # negative_prompt accepted but ignored (Z-Image-Turbo is a distilled model)
        import torch

        # Reload Z-Image model if it was unloaded for video generation
        if self.pipe is None:
            sys.stderr.write("Z-Image model not loaded, reloading...\n")
            sys.stderr.flush()
            self.initialize()

        sys.stderr.write(f"Generating: {prompt[:50]}...\n")
        sys.stderr.flush()

        if seed == -1:
            seed = random.randint(0, 2**31 - 1)

        generator = torch.Generator("cuda").manual_seed(seed)

        # Z-Image-Turbo is a distilled model - CFG is baked in during training
        # guidance_scale MUST be 0.0, negative prompts are not supported
        image = self.pipe(
            prompt=prompt,
            # negative_prompt ignored - Z-Image-Turbo doesn't use it
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=0.0,  # MUST be 0.0 for turbo models
            generator=generator,
        ).images[0]

        buf = io.BytesIO()
        image.save(buf, format='PNG')

        # Clear CUDA cache
        gc.collect()
        torch.cuda.empty_cache()

        return {"image_data": base64.b64encode(buf.getvalue()).decode(), "format": "png"}

    def generate_video(self, prompt, width=832, height=480, duration_seconds=5, fps=24, total_frames=120, seed=-1, negative_prompt=None, steps=None, guidance_scale=None):
        """Generate a video using Wan2.1 model"""
        import torch
        import tempfile
        import subprocess

        # Default inference steps based on tier (can be overridden by caller)
        # Basic=20, Pro=35, Premium=50 (set by backend based on tier)
        num_inference_steps = steps if steps is not None else 25
        cfg_scale = guidance_scale if guidance_scale is not None else 5.0

        # Default negative prompt for Wan2.1 (improves quality significantly)
        DEFAULT_NEGATIVE_PROMPT = (
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
            "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
            "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
            "misshapen limbs, fused fingers, still picture, messy background, walking backwards"
        )
        video_negative_prompt = negative_prompt if negative_prompt else DEFAULT_NEGATIVE_PROMPT

        sys.stderr.write(f"Generating video: {prompt[:50]}... ({duration_seconds}s @ {fps}fps, {num_inference_steps} steps, cfg={cfg_scale})\n")
        sys.stderr.flush()

        if seed == -1:
            seed = random.randint(0, 2**31 - 1)

        # Check if Wan model is available (must be Diffusers-compatible version)
        wan_model_path = os.environ.get("WAN_MODEL_PATH", os.path.expanduser("~/.power-node/models/Wan2.1-T2V-1.3B-Diffusers"))
        if not os.path.exists(wan_model_path):
            raise FileNotFoundError(
                f"Video model not found at {wan_model_path}. "
                "Please install the Wan2.1-Diffusers model for video generation."
            )

        # Validate model_index.json exists (required for Diffusers pipeline)
        model_index_path = os.path.join(wan_model_path, "model_index.json")
        if not os.path.exists(model_index_path):
            raise FileNotFoundError(
                f"Invalid model: model_index.json not found in {wan_model_path}. "
                "Please download the Diffusers-compatible version: Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
            )

        try:
            from diffusers import WanPipeline, AutoencoderKLWan
            from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        except ImportError:
            raise ImportError("Wan video generation requires diffusers>=0.32.0 with Wan support")

        # CRITICAL: Unload Z-Image model to free VRAM before loading Wan
        # Without this, 16GB GPUs will OOM trying to load both models
        if self.pipe is not None:
            sys.stderr.write("Unloading Z-Image model to free VRAM...\n")
            sys.stderr.flush()
            del self.pipe
            self.pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)  # Allow GPU to fully release memory
            sys.stderr.write("Z-Image model unloaded.\n")
            sys.stderr.flush()

        sys.stderr.write(f"Loading Wan model from {wan_model_path}...\n")
        sys.stderr.flush()

        # Load VAE separately with float32 for better decoding quality (per HuggingFace docs)
        vae = AutoencoderKLWan.from_pretrained(
            wan_model_path,
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        # Load Wan pipeline with custom VAE
        video_pipe = WanPipeline.from_pretrained(
            wan_model_path,
            vae=vae,
            torch_dtype=torch.bfloat16,
        )

        # Configure scheduler with flow_shift (3.0 for 480p, 5.0 for 720p)
        flow_shift = 3.0  # Using 480p resolution
        video_pipe.scheduler = UniPCMultistepScheduler.from_config(
            video_pipe.scheduler.config,
            flow_shift=flow_shift
        )

        video_pipe.enable_model_cpu_offload()

        generator = torch.Generator("cuda").manual_seed(seed)

        # Generate video frames with progress callback
        sys.stderr.write("Generating video frames...\n")
        sys.stderr.flush()

        # Progress callback that emits JSON to stdout for Go to read
        def progress_callback(pipeline, step, timestep, callback_kwargs):
            # Calculate progress percentage based on denoising step (0-indexed)
            progress_percent = (step + 1) / num_inference_steps * 100
            # Estimate frames completed based on step progress
            frames_estimated = int((step + 1) / num_inference_steps * total_frames)

            # Emit progress message to stdout (Go reads this)
            progress_msg = {
                "type": "progress",
                "step": step + 1,
                "total_steps": num_inference_steps,
                "progress_percent": progress_percent,
                "frames_completed": frames_estimated
            }
            print(json.dumps(progress_msg), flush=True)

            return callback_kwargs

        output = video_pipe(
            prompt=prompt,
            negative_prompt=video_negative_prompt,
            width=width,
            height=height,
            num_frames=total_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg_scale,
            generator=generator,
            callback_on_step_end=progress_callback,
        )

        frames = output.frames[0]  # Get first batch
        sys.stderr.write(f"Generated {len(frames)} frames\n")
        sys.stderr.flush()

        # Encode to MP4 using ffmpeg
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
            tmp_path = tmp_video.name

        # Export frames to video using export_to_video helper or manual ffmpeg
        try:
            from diffusers.utils import export_to_video
            export_to_video(frames, tmp_path, fps=fps)
        except Exception as e:
            # Fallback to manual ffmpeg encoding if export_to_video fails
            sys.stderr.write(f"export_to_video failed ({e}), using ffmpeg fallback\n")
            sys.stderr.flush()
            from PIL import Image
            import numpy as np
            with tempfile.TemporaryDirectory() as tmp_dir:
                for i, frame in enumerate(frames):
                    # Convert numpy array to PIL Image if needed
                    if isinstance(frame, np.ndarray):
                        frame = Image.fromarray(frame)
                    frame.save(os.path.join(tmp_dir, f"frame_{i:04d}.png"))

                subprocess.run([
                    'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', os.path.join(tmp_dir, 'frame_%04d.png'),
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-crf', '18',  # Higher quality encoding (was 23)
                    tmp_path
                ], check=True, capture_output=True)

        # Read video file and encode to base64
        with open(tmp_path, 'rb') as f:
            video_data = base64.b64encode(f.read()).decode()

        # Cleanup
        os.unlink(tmp_path)
        del video_pipe
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "video_data": video_data,
            "format": "mp4",
            "frames_generated": len(frames),
            "duration_seconds": duration_seconds
        }

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
    p.add_argument("--model", "-m", required=True, help="Path to Z-Image-Turbo model directory")
    p.add_argument("--vae", "-v", help="Path to VAE (optional)")
    args = p.parse_args()

    svc = InferenceService(args.model, args.vae)
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
