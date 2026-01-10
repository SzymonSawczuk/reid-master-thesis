"""
Detection utilities.
"""

import cv2
import numpy as np
from typing import List, Tuple


def crop_bbox(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """
    Crop bounding box from image.

    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]

    # Clip to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return image[y1:y2, x1:x2]
