"""Face enhancement using GFPGAN for post-swap quality improvement."""

import os
import cv2
import numpy as np
from pathlib import Path

# Apply torchvision compatibility patch before importing GFPGAN
# This fixes the functional_tensor -> _functional_tensor rename in torchvision 0.17+
try:
    from . import torchvision_compat
except ImportError:
    pass

# Try to import GFPGAN - may fail due to basicsr/torchvision compatibility
try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GFPGAN not available - {e}")
    print("Face enhancement will be disabled. To enable, install compatible torch/torchvision versions.")
    GFPGAN_AVAILABLE = False
    GFPGANer = None


class FaceEnhancer:
    """Face enhancement using GFPGAN 1.4 model."""

    def __init__(self, model_dir: str = "./models"):
        """
        Initialize the face enhancer.

        Args:
            model_dir: Directory containing the GFPGAN model
        """
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "GFPGANv1.4.pth"
        self.enhancer = None

        # Check if GFPGAN is available (import succeeded)
        if not GFPGAN_AVAILABLE:
            print("GFPGAN not available due to import error. Enhancement disabled.")
            return

        # Only initialize if model exists
        if self.model_path.exists():
            self._init_enhancer()
        else:
            print(f"GFPGAN model not found at {self.model_path}")
            print("Face enhancement will be disabled until model is downloaded.")

    def _init_enhancer(self):
        """Initialize the GFPGAN enhancer."""
        print("Loading GFPGAN face enhancer...")
        self.enhancer = GFPGANer(
            model_path=str(self.model_path),
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None  # Skip background upscaling for speed
        )
        print("Face enhancer initialized!")

    def is_available(self) -> bool:
        """Check if the enhancer is available."""
        return self.enhancer is not None

    def enhance(
        self,
        img: np.ndarray,
        only_center_face: bool = False,
        weight: float = 0.5
    ) -> np.ndarray:
        """
        Enhance faces in an image.

        Args:
            img: Input image (BGR format from OpenCV)
            only_center_face: Only process the center face
            weight: Weight for restoration (0-1, higher = more restoration)

        Returns:
            Enhanced image with improved face quality (same dimensions as input)
        """
        if not self.is_available():
            # Return original if enhancer not available
            return img

        # Capture original dimensions BEFORE enhancement
        # GFPGAN with upscale=2 outputs 2X larger images
        original_h, original_w = img.shape[:2]

        try:
            # GFPGAN expects BGR input and returns BGR output
            _, _, output = self.enhancer.enhance(
                img,
                has_aligned=False,
                only_center_face=only_center_face,
                paste_back=True,
                weight=weight
            )

            # Resize back to original dimensions if GFPGAN upscaled
            # This ensures consistent output size for GIF frame processing
            if output.shape[:2] != (original_h, original_w):
                output = cv2.resize(
                    output,
                    (original_w, original_h),  # cv2.resize uses (width, height)
                    interpolation=cv2.INTER_LANCZOS4
                )

            return output
        except Exception as e:
            print(f"Face enhancement failed: {e}")
            # Return original on failure
            return img

    def enhance_face_region(
        self,
        img: np.ndarray,
        bbox: tuple,
        padding: int = 20
    ) -> np.ndarray:
        """
        Enhance only a specific face region for better performance.

        Args:
            img: Full image (BGR format)
            bbox: Face bounding box (x1, y1, x2, y2)
            padding: Padding around the face region

        Returns:
            Image with enhanced face region
        """
        if not self.is_available():
            return img

        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Add padding
        h, w = img.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        # Extract face region
        face_region = img[y1:y2, x1:x2].copy()

        # Enhance the region
        enhanced_region = self.enhance(face_region, only_center_face=True)

        # Resize if dimensions changed due to upscaling
        if enhanced_region.shape[:2] != face_region.shape[:2]:
            enhanced_region = cv2.resize(
                enhanced_region,
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_LANCZOS4
            )

        # Paste back
        result = img.copy()
        result[y1:y2, x1:x2] = enhanced_region

        return result


def download_gfpgan_model(model_dir: str = "./models") -> str:
    """
    Download the GFPGAN v1.4 model if not present.

    Args:
        model_dir: Directory to save the model

    Returns:
        Path to the downloaded model
    """
    import urllib.request

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "GFPGANv1.4.pth"

    if model_path.exists():
        print(f"GFPGAN model already exists at {model_path}")
        return str(model_path)

    url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    print(f"Downloading GFPGAN model from {url}...")
    print("This may take a few minutes (~350MB)...")

    try:
        urllib.request.urlretrieve(url, str(model_path))
        print(f"Model downloaded to {model_path}")
        return str(model_path)
    except Exception as e:
        print(f"Failed to download model: {e}")
        raise
