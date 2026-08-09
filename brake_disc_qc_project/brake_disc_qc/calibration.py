from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class CameraCalibration:
    pixels_per_mm: float | None = None
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None
    source: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)

    def undistort(self, image: np.ndarray) -> np.ndarray:
        if self.camera_matrix is None or self.dist_coeffs is None:
            return image

        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def set_scale(self, measured_pixels: float, actual_mm: float, source: str = "manual") -> None:
        if measured_pixels <= 0:
            raise ValueError("measured_pixels must be greater than zero")
        if actual_mm <= 0:
            raise ValueError("actual_mm must be greater than zero")

        self.pixels_per_mm = float(measured_pixels) / float(actual_mm)
        self.source = source

    def get_status(self) -> dict[str, Any]:
        return {
            "pixels_per_mm": self.pixels_per_mm,
            "has_intrinsics": self.camera_matrix is not None and self.dist_coeffs is not None,
            "source": self.source,
            "metadata": self.metadata,
        }
