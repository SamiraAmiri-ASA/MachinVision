from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")

    return image


def to_gray(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None")

    if image.ndim == 2:
        return image

    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")


def normalize_illumination(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def circle_circularity(contour_or_area, perimeter: float | None = None) -> float:
    if perimeter is None:
        contour = contour_or_area
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
    else:
        area = contour_or_area

    if perimeter <= 0:
        return 0.0

    return float(4.0 * np.pi * area / (perimeter ** 2))

def angle_between_points(
    center: tuple[float, float],
    point: tuple[float, float],
) -> float:
    cx, cy = center
    px, py = point
    angle = np.degrees(np.arctan2(py - cy, px - cx)) % 360
    return float(angle)

class ImageQualityResult(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        
def check_image_quality(
    image: np.ndarray,
    thresholds: dict[str, Any] | None = None,
    min_sharpness: float | None = None,
    min_disc_area_fraction: float | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or {}

    if min_sharpness is None:
        min_sharpness = thresholds.get("min_sharpness")
    if min_sharpness is None:
        min_sharpness = thresholds.get("min_image_sharpness")
    if min_sharpness is None:
        min_sharpness = thresholds.get("min_blur_score")

    if min_disc_area_fraction is None:
        min_disc_area_fraction = thresholds.get("min_disc_area_fraction")

    gray = to_gray(image)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    result = ImageQualityResult({
        "sharpness": sharpness,
        "blur_score": sharpness,
        "brightness": brightness,
        "contrast": contrast,
        "warnings": [],
        "acceptable": True,
        "accepted": True,
        "ok": True,
        "passed": True,
        "rejection_reason": None,
    })

    if min_sharpness is not None and sharpness < min_sharpness:
        result["warnings"].append(
            f"Image sharpness too low: {sharpness:.2f} < {min_sharpness}"
        )
        result["acceptable"] = False
        result["accepted"] = False
        result["ok"] = False
        result["passed"] = False
        result["rejection_reason"] = result["warnings"][0]

    min_brightness = thresholds.get("min_brightness")
    max_brightness = thresholds.get("max_brightness")
    min_contrast = thresholds.get("min_contrast")

    if min_brightness is not None and brightness < min_brightness:
        result["warnings"].append(
            f"Brightness too low: {brightness:.2f} < {min_brightness}"
        )
        result["acceptable"] = False
        result["accepted"] = False
        result["ok"] = False
        result["passed"] = False
        result["rejection_reason"] = result["warnings"][0]

    if max_brightness is not None and brightness > max_brightness:
        result["warnings"].append(
            f"Brightness too high: {brightness:.2f} > {max_brightness}"
        )
        result["acceptable"] = False
        result["accepted"] = False
        result["ok"] = False
        result["passed"] = False
        result["rejection_reason"] = result["warnings"][0]

    if min_contrast is not None and contrast < min_contrast:
        result["warnings"].append(
            f"Contrast too low: {contrast:.2f} < {min_contrast}"
        )
        result["acceptable"] = False
        result["accepted"] = False
        result["ok"] = False
        result["passed"] = False
        result["rejection_reason"] = result["warnings"][0]

    if min_disc_area_fraction is not None:
        result["min_disc_area_fraction"] = float(min_disc_area_fraction)

    return result
