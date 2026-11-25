"""Image Preprocessing Module for Egg Candling Fertility Classification.

Author: Anas (Image Processing Engineer - Track 1)
Role & Responsibilities:
- Resize images (configurable, e.g., 640x640 or 256x256)
- Grayscale conversion
- Green (G) channel extraction from RGB to enhance internal vascular/embryonic structures
- Contrast Limited Adaptive Histogram Equalization (CLAHE) for illumination normalization
- Integrated pipeline delivering modular, testable image processing transforms for both
  PyTorch / Albumentations / OpenCV pipelines and batch file preparation.
"""

from typing import Tuple, Union, Optional
import cv2
import numpy as np


class CandlingPreprocessor:
    """Preprocessor specialized for candling egg image enhancement and normalization.

    Attributes:
        target_size (Tuple[int, int]): (width, height) to resize images. Default is (640, 640).
        extract_green (bool): Whether to isolate the G channel from RGB.
        apply_clahe (bool): Whether to apply Contrast Limited Adaptive Histogram Equalization.
        clahe_clip_limit (float): Threshold for contrast limiting in CLAHE. Default: 2.0.
        clahe_tile_grid_size (Tuple[int, int]): Size of grid for histogram equalization. Default: (8, 8).
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        extract_green: bool = True,
        apply_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    ):
        self.target_size = target_size
        self.extract_green = extract_green
        self.apply_clahe = apply_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size

        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size
        )

    def resize(self, image: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """Resizes the input image to target_size (width, height)."""
        return cv2.resize(image, self.target_size, interpolation=interpolation)

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Converts an RGB or BGR image to single-channel grayscale."""
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def extract_g_channel(self, image: np.ndarray, is_bgr: bool = True) -> np.ndarray:
        """Extracts the Green (G) component from RGB/BGR image.

        Under candling (transillumination), hemoglobin and embryonic blood vessels
        absorb green wavelengths strongly, providing superior contrast of internal structures
        relative to the red-dominated background.

        Args:
            image (np.ndarray): Input 3-channel image.
            is_bgr (bool): True if loaded via OpenCV (BGR format), False if RGB.

        Returns:
            np.ndarray: Single-channel Green component.
        """
        if len(image.shape) == 2:
            return image
        # In BGR: index 1 is Green. In RGB: index 1 is also Green!
        return image[:, :, 1]

    def apply_clahe_enhancement(self, single_channel_img: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).

        Normalizes non-uniform illumination across the eggshell caused by focal LED hotspots.
        """
        if len(single_channel_img.shape) == 3:
            # If 3-channel, apply to L channel in LAB color space
            lab = cv2.cvtColor(single_channel_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = self._clahe.apply(l)
            merged = cv2.merge((l_enhanced, a, b))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        return self._clahe.apply(single_channel_img)

    def process(
        self,
        image: np.ndarray,
        return_3channel: bool = True,
        is_bgr: bool = True
    ) -> np.ndarray:
        """Full preprocessing pipeline execution for a single image.

        Steps:
        1. Resize to target dimension (e.g. 640x640 or 256x256).
        2. Extract G channel (if enabled) or grayscale to isolate vascular contrast.
        3. Apply CLAHE illumination normalization.
        4. Optionally duplicate back into 3 channels for standard CNN / YOLO input shapes.

        Args:
            image (np.ndarray): Raw input image array.
            return_3channel (bool): If True, replicates single-channel result into (H, W, 3).
            is_bgr (bool): True if image format is BGR.

        Returns:
            np.ndarray: Preprocessed and enhanced image.
        """
        # Step 1: Resize
        resized = self.resize(image)

        # Step 2: G-channel extraction or standard grayscale
        if self.extract_green:
            channel = self.extract_g_channel(resized, is_bgr=is_bgr)
        else:
            channel = self.to_grayscale(resized)

        # Step 3: CLAHE illumination normalization
        if self.apply_clahe:
            enhanced = self.apply_clahe_enhancement(channel)
        else:
            enhanced = channel

        # Step 4: Format output channels
        if return_3channel and len(enhanced.shape) == 2:
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return enhanced


def preprocess_image_file(
    input_path: str,
    output_path: Optional[str] = None,
    preprocessor: Optional[CandlingPreprocessor] = None,
    target_size: Tuple[int, int] = (640, 640),
    extract_green: bool = True,
    apply_clahe: bool = True,
    return_3channel: bool = True,
) -> np.ndarray:
    """Convenience helper to read an image file, preprocess, and optionally save to disk."""
    if preprocessor is None:
        preprocessor = CandlingPreprocessor(
            target_size=target_size,
            extract_green=extract_green,
            apply_clahe=apply_clahe,
        )

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at path: {input_path}")

    processed = preprocessor.process(img, return_3channel=return_3channel)

    if output_path:
        cv2.imwrite(output_path, processed)

    return processed
