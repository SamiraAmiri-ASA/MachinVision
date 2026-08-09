"""Geometric feature detection and fitting for brake disc inspection.

Detection strategy (classical CV, deterministic):
  1. Segment the disc from background -> outer boundary (ellipse/circle).
  2. Locate the central hole as the dark blob nearest the disc centroid.
  3. Locate mounting holes as circular dark blobs in the hub annulus,
     clustered by radial distance from the center hole.
  4. Select exactly N mounting holes and sort them by angle.

All detections carry a confidence derived from circularity and fit residual.
No physical (mm) value is emitted here; scaling happens in measurement.py.
"""
from __future__ import annotations

import logging
import numpy as np
import cv2
from typing import Optional

from .models import DiscGeometry
from .utils import to_gray, normalize_illumination, circle_circularity, angle_between_points

logger = logging.getLogger(__name__)


class FeatureDetectionError(Exception):
    """Raised when required geometric features cannot be located."""


def _fit_outer_boundary(gray: np.ndarray) -> tuple[tuple[float, float, float], Optional[np.ndarray]]:
    """Segment the disc and fit its outer boundary.

    Returns (cx, cy, radius) and the fitted ellipse (or None). The disc is the
    largest bright/foreground blob. Otsu handles most lighting; morphology
    closes internal holes so the disc is a solid mask.
    """
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # The disc may be darker or brighter than background; pick the mask whose
    # foreground is a compact central blob rather than the border.
    if mask[0, 0] == 255:  # corner is foreground -> background got labeled, invert
        mask = cv2.bitwise_not(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise FeatureDetectionError("No disc contour found during outer boundary fit.")

    disc = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(disc)

    ellipse = None
    if len(disc) >= 5:
        ellipse = cv2.fitEllipse(disc)  # ((cx,cy),(MA,ma),angle)

    return (float(cx), float(cy), float(radius)), ellipse


def _detect_dark_blobs(
    gray: np.ndarray,
    disc_mask: np.ndarray,
) -> list[dict]:
    """Detect dark circular blobs (holes) restricted to the disc mask.

    Returns a list of candidate dicts with center, radius, circularity, area.
    """
    norm = normalize_illumination(gray)
    # Holes are dark relative to the polished/cast surface -> inverse threshold.
    _, dark = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = cv2.bitwise_and(dark, dark, mask=disc_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 20:  # discard specks; real filtering happens by scale later
            continue
        (bx, by), br = cv2.minEnclosingCircle(c)
        circ = circle_circularity(c)
        # Fill ratio guards against ring/arc fragments that mimic circles.
        fill = area / (np.pi * br * br + 1e-6)
        candidates.append({
            "center": (float(bx), float(by)),
            "radius": float(br),
            "circularity": float(circ),
            "fill": float(fill),
            "area": float(area),
            "contour": c,
        })
    return candidates


def _select_center_hole(candidates: list[dict], disc_center: tuple[float, float]) -> dict:
    """Center hole = large, circular blob closest to the disc centroid."""
    circular = [c for c in candidates if c["circularity"] > 0.6 and c["fill"] > 0.6]
    if not circular:
        raise FeatureDetectionError("No circular candidate qualifies as center hole.")

    def score(c: dict) -> float:
        d = np.hypot(c["center"][0] - disc_center[0], c["center"][1] - disc_center[1])
        # Prefer near-center and large radius; penalize distance heavily.
        return c["radius"] * 2.0 - d
    return max(circular, key=score)


def _select_mounting_holes(
    candidates: list[dict],
    center_hole: dict,
    expected_count: int,
) -> list[dict]:
    """Select the mounting holes as blobs sharing a common radial distance.

    Mounting holes lie on a bolt circle: equal radius from the center hole and
    similar diameter. We cluster candidates by radial distance and pick the
    cluster of size >= expected_count with the most self-consistent geometry.
    """
    ch = center_hole["center"]
    ch_radius = center_hole["radius"]

    pool = []
    for c in candidates:
        if c is center_hole:
            continue
        if c["circularity"] < 0.55 or c["fill"] < 0.55:
            continue
        r = np.hypot(c["center"][0] - ch[0], c["center"][1] - ch[1])
        if r <= ch_radius * 1.1:  # too close -> part of the hub, not a bolt hole
            continue
        c = {**c, "radial_dist": float(r)}
        pool.append(c)

    if len(pool) < expected_count:
        raise FeatureDetectionError(
            f"Found {len(pool)} mounting-hole candidates, need {expected_count}."
        )

    # Cluster by radial distance with a tolerance proportional to median radius.
    radii = np.array([c["radial_dist"] for c in pool])
    order = np.argsort(radii)
    pool = [pool[i] for i in order]
    radii = radii[order]

    best_cluster: list[dict] = []
    best_cost = np.inf
    tol = max(radii.std() * 0.5, radii.mean() * 0.06)
    for i in range(len(pool)):
        cluster = [pool[i]]
        base = radii[i]
        for j in range(len(pool)):
            if j == i:
                continue
            if abs(radii[j] - base) <= tol:
                cluster.append(pool[j])
        if len(cluster) < expected_count:
            continue
        # Cost favors radial + diameter consistency.
        rd = np.array([c["radial_dist"] for c in cluster])
        dd = np.array([c["radius"] for c in cluster])
        cost = rd.std() + dd.std()
        if cost < best_cost:
            best_cost = cost
            best_cluster = cluster

    if len(best_cluster) < expected_count:
        raise FeatureDetectionError("Could not form a consistent bolt-circle cluster.")

    # If more than expected, keep the ones most consistent with the median.
    if len(best_cluster) > expected_count:
        med_r = np.median([c["radial_dist"] for c in best_cluster])
        med_d = np.median([c["radius"] for c in best_cluster])
        best_cluster.sort(
            key=lambda c: abs(c["radial_dist"] - med_r) + abs(c["radius"] - med_d)
        )
        best_cluster = best_cluster[:expected_count]

    # Deterministic sort by angle around the center hole.
    best_cluster.sort(key=lambda c: angle_between_points(ch, c["center"]))
    return best_cluster


def find_disc_features(
    img: np.ndarray,
    config: dict,
    pixels_per_mm: float,
) -> DiscGeometry:
    """Locate all geometric features required for dimensional inspection."""
    gray = to_gray(img)
    expected_holes = int(config["nominal_dimensions"]["mounting_hole_count"])

    outer, ellipse = _fit_outer_boundary(gray)
    ocx, ocy, oradius = outer

    # Build a filled disc mask to constrain hole search.
    disc_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(disc_mask, (int(ocx), int(ocy)), int(oradius), 255, -1)

    candidates = _detect_dark_blobs(gray, disc_mask)
    if not candidates:
        raise FeatureDetectionError("No hole candidates detected inside disc.")

    center_hole = _select_center_hole(candidates, (ocx, ocy))
    mounting = _select_mounting_holes(candidates, center_hole, expected_holes)

    geo = DiscGeometry(
        outer_circle=(ocx, ocy, oradius),
        inner_circle=None,
        inner_circle=(center_hole["center"][0], center_hole["center"][1], center_hole["radius"]),
        mounting_holes=[(m["center"][0], m["center"][1], m["radius"]) for m in mounting],
        disc_center_px=center_hole["center"],  # measurements reference the center hole
        pixels_per_mm=pixels_per_mm,
    )

    # Stash raw candidate metadata for confidence propagation.
    geo._center_meta = center_hole  # type: ignore[attr-defined]
    geo._mounting_meta = mounting   # type: ignore[attr-defined]
    if ellipse is not None:
        geo._outer_ellipse = ellipse  # type: ignore[attr-defined]

    logger.info(
        "Detected center hole r=%.1fpx and %d mounting holes.",
        center_hole["radius"], len(mounting),
    )
    return geo