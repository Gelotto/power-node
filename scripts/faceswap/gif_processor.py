"""GIF processing for frame-by-frame face swapping with enhancement."""

import io
import cv2
import numpy as np
import imageio.v3 as iio
from PIL import Image


class GifProcessor:
    """Process GIFs for face swapping with optional enhancement."""

    def __init__(self, swapper, enhancer=None):
        """
        Initialize with a FaceSwapper instance and optional enhancer.

        Args:
            swapper: FaceSwapper instance for performing face swaps
            enhancer: Optional FaceEnhancer instance for post-processing
        """
        self.swapper = swapper
        self.enhancer = enhancer

    def process_gif(
        self,
        source_img: np.ndarray,
        gif_data: bytes,
        max_frames: int = 100,
        enhance: bool = True
    ) -> bytes:
        """
        Process a GIF by swapping faces in each frame with optional enhancement.

        Args:
            source_img: The source face image (numpy array, BGR)
            gif_data: The GIF file data as bytes
            max_frames: Maximum number of frames to process
            enhance: Whether to apply face enhancement after swap

        Returns:
            The processed GIF as bytes
        """
        # Read GIF frames
        frames = iio.imread(gif_data, index=None)

        if len(frames) == 0:
            raise ValueError("No frames found in GIF")

        # Get metadata for duration
        metadata = iio.immeta(gif_data)
        duration = metadata.get("duration", 100)  # Default 100ms per frame

        # Limit frames
        if len(frames) > max_frames:
            # Sample frames evenly
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = [frames[i] for i in indices]

        # Capture target dimensions from first frame for consistency
        # All output frames must have identical dimensions for GIF writing
        target_h, target_w = frames[0].shape[:2]

        processed_frames = []
        frames_swapped = 0  # Track successful swaps

        for i, frame in enumerate(frames):
            try:
                # Convert RGBA to RGB if needed
                if frame.shape[-1] == 4:
                    # Create white background
                    bg = np.ones((frame.shape[0], frame.shape[1], 3), dtype=np.uint8) * 255
                    # Alpha blend
                    alpha = frame[:, :, 3:4] / 255.0
                    frame_rgb = frame[:, :, :3]
                    frame = (frame_rgb * alpha + bg * (1 - alpha)).astype(np.uint8)

                # Convert RGB to BGR for OpenCV
                frame_bgr = frame[:, :, ::-1]

                # Swap face (without enhancement - we'll do it separately for GIFs)
                swapped = self.swapper.swap(source_img, frame_bgr, enhance=False)

                # Apply enhancement if enabled and enhancer is available
                if enhance and self.enhancer is not None:
                    swapped = self.enhancer.enhance(swapped)

                # Convert back to RGB
                swapped_rgb = swapped[:, :, ::-1]

                # Safety: ensure frame matches target dimensions
                # (handles edge cases where enhancement changed size)
                if swapped_rgb.shape[:2] != (target_h, target_w):
                    swapped_rgb = cv2.resize(
                        swapped_rgb,
                        (target_w, target_h),
                        interpolation=cv2.INTER_LANCZOS4
                    )

                processed_frames.append(swapped_rgb)
                frames_swapped += 1

            except Exception as e:
                # If face swap fails on a frame, use original
                print(f"Frame {i} swap failed: {e}")
                if frame.shape[-1] == 4:
                    frame = frame[:, :, :3]

                # Safety: ensure fallback frame matches target dimensions
                if frame.shape[:2] != (target_h, target_w):
                    frame = cv2.resize(
                        frame,
                        (target_w, target_h),
                        interpolation=cv2.INTER_LANCZOS4
                    )

                processed_frames.append(frame)

        # Fail if no frames were successfully swapped
        # This prevents silent failures where output looks identical to input
        if frames_swapped == 0:
            raise ValueError(
                f"No faces detected in any of the {len(frames)} frames. "
                "Face swap only works on human faces, not cartoons or illustrations."
            )

        # Write output GIF
        output = io.BytesIO()
        iio.imwrite(
            output,
            processed_frames,
            format="GIF",
            duration=duration,
            loop=0
        )

        return output.getvalue()

    def extract_frames(self, gif_data: bytes) -> list[np.ndarray]:
        """Extract all frames from a GIF."""
        return list(iio.imread(gif_data, index=None))

    def create_gif(
        self,
        frames: list[np.ndarray],
        duration: int = 100
    ) -> bytes:
        """Create a GIF from frames."""
        output = io.BytesIO()
        iio.imwrite(
            output,
            frames,
            format="GIF",
            duration=duration,
            loop=0
        )
        return output.getvalue()
