"""Surface defect detection on the annular friction ring.

Approach:
  1. Build an annular ROI mask (outer boundary minus hub, holes, and center).
  2. Normalize illumination to suppress shading and glare gradients.
  3. If a registered healthy reference is supplied, use difference analysis;
     otherwise fall back to local statistical anomaly detection.
  4. Threshold, clean with morphology, extract connected components.
  5. Filter by configurable area/shape and classify severity.
"""
from __future__ import annotations

import logging
import numpy as np
import cv2
from typing import Optional

from .models import DiscGeometry, SurfaceQCResult, DefectResult, Status
from .utils import to_gray, normalize_illumination

logger = logging.getLogger(__name__)


def _build_ring_mask(shape: tuple[int, int], geo: DiscGeometry,
                     inner_margin: float, outer_margin: float) -> np.ndarray:
    """Annular polished-surface mask, excluding hub, holes, and borders."""
    mask = np.zeros(shape, dtype=np.uint8)
    ocx, ocy, oradius = geo.outer_circle
    ch_x, ch_y, ch_r = geo.center_hole

    # Outer ring: shrink outer boundary to drop the edge chamfer/border.
    r_out = int(oradius * outer_margin)
    cv2.circle(mask, (int(ocx), int(ocy)), r_out, 255, -1)

    # Exclude the central hub region. We approximate the hub radius from the
    # bolt circle so cast/rough material near the center is not inspected.
    if geo.mounting_holes:
        bolt_r = np.mean([np.hypot(mx - ch_x, my - ch_y) for mx, my, _ in geo.mounting_holes])
        hub_r = int(bolt_r * inner_margin)
    else:
        hub_r = int(ch_r * 3)
    cv2.circle(mask, (int(ch_x), int(ch_y)), hub_r, 0, -1)

    # Punch out each mounting hole with a safety margin.
    for mx, my, mr in geo.mounting_holes:
        cv2.circle(mask, (int(mx), int(my)), int(mr * 1.4), 0, -1)

    return mask


def _defect_mask_reference(roi: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Difference-based defect mask against a registered healthy reference."""
    ref_gray = to_gray(reference)
    if ref_gray.shape != roi.shape:
        ref_gray = cv2.resize(ref_gray, (roi.shape[1], roi.shape[0]))
    diff = cv2.absdiff(normalize_illumination(roi), normalize_illumination(ref_gray))
    _, defects = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_and(defects, defects, mask=mask)


def _defect_mask_statistical(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Self-referential anomaly detection when no golden image is available.

    A defect deviates locally from the surrounding polished texture. We compare
    each pixel to a large-kernel local mean; strong residuals are anomalies.
    """
    norm = normalize_illumination(roi)
    local_mean = cv2.blur(norm.astype(np.float32), (51, 51))
    residual = np.abs(norm.astype(np.float32) - local_mean)

    vals = residual[mask > 0]
    if vals.size == 0:
        return np.zeros_like(mask)
    thresh = float(vals.mean() + 3.0 * vals.std())  # 3-sigma anomaly gate
    defects = (residual > thresh).astype(np.uint8) * 255
    return cv2.bitwise_and(defects, defects, mask=mask)


def _classify_severity(area_mm2: float, cfg: dict) -> str:
    if area_mm2 >= cfg["max_single_defect_area_mm2"]:
        return "severe"
    if area_mm2 >= cfg["max_single_defect_area_mm2"] * 0.5:
        return "moderate"
    return "minor"


def detect_surface_defects(
    img: np.ndarray,
    config: dict,
    geo: DiscGeometry,
    calibration_valid: bool,
    reference: Optional[np.ndarray] = None,
    inner_margin: float = 1.25,
    outer_margin: float = 0.95,
) -> tuple[SurfaceQCResult, np.ndarray]:
    """Detect and score surface defects. Returns (result, defect_mask)."""
    gray = to_gray(img)
    cfg = config["surface_qc"]
    ppm = geo.pixels_per_mm

    mask = _build_ring_mask(gray.shape, geo, inner_margin, outer_margin)
    roi = cv2.bitwise_and(gray, gray, mask=mask)

    if reference is not None:
        defect_mask = _defect_mask_reference(roi, reference, mask)
    else:
        defect_mask = _defect_mask_statistical(roi, mask)

    # Clean noise; keep thin cracks by using a small kernel.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(defect_mask, 8)

    px2_per_mm2 = (ppm ** 2) if calibration_valid else float("nan")
    min_area_px = cfg["min_defect_area_mm2"] * (ppm ** 2) if calibration_valid else 15.0

    defects: list[DefectResult] = []
    total_area_mm2 = 0.0
    for i in range(1, n_labels):  # skip background label 0
        area_px = float(stats[i, cv2.CC_STAT_AREA])
        if area_px < min_area_px:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area_mm2 = area_px / px2_per_mm2 if calibration_valid else float("nan")
        if calibration_valid:
            total_area_mm2 += area_mm2

        # Confidence scales with how far the blob exceeds the noise floor.
        extent = area_px / (w * h + 1e-6)
        confidence = float(np.clip(0.6 + 0.4 * extent, 0.0, 1.0))

        defects.append(DefectResult(
            defect_id=len(defects) + 1,
            area_px=area_px,
            area_mm2=area_mm2 if calibration_valid else 0.0,
            centroid_px=(float(centroids[i][0]), float(centroids[i][1])),
            bbox=(x, y, w, h),
            severity=_classify_severity(area_mm2 if calibration_valid else 0.0, cfg),
            confidence=confidence,
        ))

    region_area_px = float(np.count_nonzero(mask))
    region_area_mm2 = region_area_px / px2_per_mm2 if calibration_valid else 0.0

    # Pass/fail decision.
    status = Status.PASS
    if not calibration_valid:
        status = Status.UNCERTAIN
    else:
        max_single = max((d.area_mm2 for d in defects), default=0.0)
        if (len(defects) > cfg["max_defect_count"]
                or total_area_mm2 > cfg["max_total_defect_area_mm2"]
                or max_single > cfg["max_single_defect_area_mm2"]):
            status = Status.FAIL

    result = SurfaceQCResult(
        status=status,
        total_defect_area_mm2=round(total_area_mm2, 4),
        defect_count=len(defects),
        defects=defects,
        inspection_region_area_mm2=round(region_area_mm2, 2),
        defect_density=round(len(defects) / region_area_mm2, 6) if region_area_mm2 > 0 else 0.0,
    )
    return result, defect_mask