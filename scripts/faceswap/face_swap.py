"""Face swap implementation using InsightFace with GFPGAN enhancement."""

import os
import cv2
import numpy as np
from pathlib import Path

# InsightFace imports
import insightface
from insightface.app import FaceAnalysis

# Face enhancement
from .face_enhancer import FaceEnhancer

# Minimum confidence score for face detection (0-1)
FACE_CONFIDENCE_THRESHOLD = 0.5


class FaceSwapper:
    """Face swap class using InsightFace's inswapper model with GFPGAN enhancement."""

    def __init__(self, model_dir: str = "./models", enable_enhancement: bool = True):
        """Initialize the face swapper with models."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.enable_enhancement = enable_enhancement

        # Initialize face analyzer
        print("Loading face analysis model...")
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=str(self.model_dir),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # Load inswapper model
        print("Loading face swap model...")
        model_path = self.model_dir / "inswapper_128.onnx"

        if not model_path.exists():
            print(f"Model not found at {model_path}")
            print("Downloading inswapper model...")
            # The model will be downloaded automatically by insightface
            self.swapper = insightface.model_zoo.get_model(
                "inswapper_128.onnx",
                download=True,
                download_zip=True,
                root=str(self.model_dir)
            )
        else:
            self.swapper = insightface.model_zoo.get_model(
                str(model_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

        # Initialize face enhancer (GFPGAN)
        self.enhancer = None
        if self.enable_enhancement:
            self.enhancer = FaceEnhancer(model_dir=str(self.model_dir))
            if not self.enhancer.is_available():
                print("Warning: Face enhancement disabled (model not found)")
                self.enhancer = None

        print("Face swapper initialized!")

    def detect_faces(self, img: np.ndarray, filter_low_confidence: bool = True) -> list:
        """
        Detect faces in an image.

        Args:
            img: Image to detect faces in
            filter_low_confidence: Filter out low-confidence detections

        Returns:
            List of detected faces
        """
        faces = self.app.get(img)

        if filter_low_confidence:
            # Filter faces below confidence threshold
            faces = [f for f in faces if f.det_score >= FACE_CONFIDENCE_THRESHOLD]

        return faces

    def swap(
        self,
        source_img: np.ndarray,
        target_img: np.ndarray,
        swap_all: bool = True,
        enhance: bool = True
    ) -> np.ndarray:
        """
        Swap face from source onto target image with optional enhancement.

        Args:
            source_img: Image containing the face to use
            target_img: Image where faces will be replaced
            swap_all: If True, replace all faces in target
            enhance: If True, apply GFPGAN enhancement after swap

        Returns:
            Result image with swapped and enhanced face(s)
        """
        # Detect faces
        source_faces = self.detect_faces(source_img)
        target_faces = self.detect_faces(target_img)

        if not source_faces:
            raise ValueError("No face detected in source image")
        if not target_faces:
            raise ValueError("No face detected in target image")

        # Use the first detected face from source
        source_face = source_faces[0]

        # Swap faces
        result = target_img.copy()

        if swap_all:
            # Replace all faces in target
            for target_face in target_faces:
                result = self.swapper.get(
                    result,
                    target_face,
                    source_face,
                    paste_back=True
                )
        else:
            # Only replace the first/largest face
            target_face = max(target_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            result = self.swapper.get(
                result,
                target_face,
                source_face,
                paste_back=True
            )

        # Apply face enhancement (GFPGAN) to improve quality
        if enhance and self.enhancer is not None:
            result = self.enhancer.enhance(result)

        return result


def load_image(path: str) -> np.ndarray:
    """Load an image from file path."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


from .constants import MAX_IMAGE_DIMENSION


def load_image_from_bytes(data: bytes, max_dim: int = MAX_IMAGE_DIMENSION) -> np.ndarray:
    """
    Load an image from bytes with dimension validation.

    Args:
        data: Raw image bytes
        max_dim: Maximum allowed dimension (width or height)

    Returns:
        Image as numpy array (BGR format)

    Raises:
        ValueError: If image fails to decode or exceeds max dimensions
    """
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from bytes")

    # Validate dimensions to prevent OOM attacks
    h, w = img.shape[:2]
    if h > max_dim or w > max_dim:
        raise ValueError(
            f"Image dimensions too large: {w}x{h} (max: {max_dim}x{max_dim}). "
            "Please resize the image before processing."
        )

    return img


def save_image(img: np.ndarray, path: str) -> None:
    """Save an image to file path."""
    cv2.imwrite(path, img)


def image_to_bytes(img: np.ndarray, format: str = ".jpg", quality: int = 95) -> bytes:
    """
    Convert image to bytes with configurable quality.

    Args:
        img: Image to encode
        format: Output format (.jpg, .png, etc.)
        quality: JPEG quality (0-100, higher = better quality)

    Returns:
        Encoded image bytes
    """
    if format.lower() in [".jpg", ".jpeg"]:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif format.lower() == ".png":
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]  # 0-9, lower = faster
    else:
        encode_params = []

    success, encoded = cv2.imencode(format, img, encode_params)
    if not success:
        raise ValueError("Failed to encode image")
    return encoded.tobytes()
