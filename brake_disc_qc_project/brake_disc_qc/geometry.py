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

import cv2
import numpy as np

from .models import DiscGeometry
from .utils import angle_between_points, circle_circularity, to_gray

logger = logging.getLogger(__name__)
# ((cx, cy), (major_axis, minor_axis), angle_deg) as returned by cv2.fitEllipse
Ellipse = tuple[tuple[float, float], tuple[float, float], float]


class FeatureDetectionError(Exception):
    """Raised when required geometric features cannot be located."""


def _fit_outer_boundary(gray: np.ndarray) -> tuple[tuple[float, float, float], Ellipse | None]:
    """Segment the disc and fit its outer boundary.

    Returns (cx, cy, radius) and the fitted ellipse (or None). The disc is the
    largest bright/foreground blob. Otsu handles most lighting; morphology
    closes internal holes so the disc is a solid mask.
    """
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    border = np.concatenate((blurred[0], blurred[-1], blurred[:, 0], blurred[:, -1]))
    background_level = float(np.median(border))
    difference = cv2.absdiff(blurred, np.full_like(blurred, round(background_level)))
    # Use border noise as the foreground gate. Otsu can choose the gap between
    # the disc and its dark holes, returning only the center hole as foreground.
    foreground_gate = max(5.0, 3.0 * float(np.std(border)))
    mask = np.where(difference > foreground_gate, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    disc = max(contours, key=cv2.contourArea) if contours else None

    if disc is not None:
        (cx, cy), radius = cv2.minEnclosingCircle(disc)
        area = cv2.contourArea(disc)
        fill = area / (np.pi * radius * radius + 1e-6)
        # A non-uniform background (wood/concrete/etc. at similar brightness
        # to parts of the disc) can pass the brightness gate on only part of
        # the disc, producing a crescent instead of the full silhouette.
        # Reject that before trusting the fit.
        if circle_circularity(disc) >= 0.7 and fill >= 0.7:
            ellipse = cv2.fitEllipse(disc) if len(disc) >= 5 else None
            return (float(cx), float(cy), float(radius)), ellipse

    # Brightness segmentation didn't yield a plausible disc silhouette.
    # A Hough circle transform on the edge map tolerates cluttered/uneven
    # backgrounds since it only needs a strong circular edge, not a clean
    # interior/exterior brightness split.
    h, w = gray.shape
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=min(h, w) / 2.0,
        param1=120, param2=60,
        minRadius=int(min(h, w) * 0.25), maxRadius=int(min(h, w) * 0.55),
    )
    if circles is not None:
        # minRadius already excludes inner features (center bore); among
        # whatever qualifies, the disc's outer edge is the largest circle.
        cx, cy, radius = max(circles[0], key=lambda c: c[2])
        return (float(cx), float(cy), float(radius)), None

    if disc is None:
        raise FeatureDetectionError("No disc contour found during outer boundary fit.")

    # Last resort: neither check panned out cleanly; use the brightness-based
    # fit anyway rather than failing outright.
    (cx, cy), radius = cv2.minEnclosingCircle(disc)
    ellipse = cv2.fitEllipse(disc) if len(disc) >= 5 else None
    return (float(cx), float(cy), float(radius)), ellipse


def _detect_dark_blobs(
    gray: np.ndarray,
    disc_mask: np.ndarray,
    disc_radius: float,
) -> list[dict]:
    """Detect dark circular blobs (holes) restricted to the disc mask.

    Returns a list of candidate dicts with center, radius, circularity, area.
    """
    # Compute Otsu's threshold from disc pixels only. Including the brighter
    # background (or applying CLAHE across the disc edge) can merge the whole
    # disc body into one dark blob and hide every actual hole.
    if not np.any(disc_mask):
        return []

    # Adaptive thresholding: illumination is non-uniform across the disc
    # (bright machined friction ring vs. dark cast hub). A single global Otsu
    # threshold either swallows the entire hub or keeps only rim shadows.
    # Block size scales with the disc so it stays larger than a mounting hole.
    block = int(disc_radius * 0.35) | 1          # force odd
    block = int(np.clip(block, 31, 201)) | 1
    dark = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block, 7,
    )
    dark = cv2.bitwise_and(dark, disc_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)

    m = cv2.moments(disc_mask, binaryImage=True)
    if m["m00"] > 0:
        mcx, mcy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        hub_mask = np.zeros_like(disc_mask)
        # 0.85 (not e.g. 0.65) because the bolt circle can sit well out
        # toward the friction ring; this only needs to exclude candidates
        # right at the outer edge, not tightly hug the hub.
        cv2.circle(hub_mask, (int(round(mcx)), int(round(mcy))),
                   int(disc_radius * 0.85), 255, -1)
        dark = cv2.bitwise_and(dark, hub_mask)

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
            "contrast": _local_darkness_contrast(gray, c, disc_radius),
            "contour": c,
        })
    return candidates


def _local_darkness_contrast(gray: np.ndarray, contour: np.ndarray, disc_radius: float) -> float:
    """Mean intensity drop between a blob's interior and its immediate surroundings.

    Real through-holes are near-black against the surrounding metal; cast
    marks, pitting, and shadows are shallower and read as low contrast even
    when their shape passes the circularity/fill gates. The ring is sized off
    the disc scale (not the candidate's own radius) so it stays a thin,
    comparable band for both the large center bore and small mounting holes.
    """
    inside = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(inside, [contour], -1, 255, -1)
    ring_width = max(15, int(disc_radius * 0.08)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width, ring_width))
    ring = cv2.bitwise_and(cv2.dilate(inside, kernel), cv2.bitwise_not(inside))
    if not np.any(ring):
        return 0.0
    return float(gray[ring > 0].mean()) - float(gray[inside > 0].mean())


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
    # Real through-holes are as dark against their surroundings as the
    # (already-verified) center hole. Cast marks, pitting, and shadows pass
    # the shape gates below but are visibly shallower, so anchor the darkness
    # floor to the center hole's own contrast instead of a fixed gray value.
    min_contrast = center_hole.get("contrast", 0.0) * 0.35

    pool = []
    for c in candidates:
        if c is center_hole:
            continue
        if c["circularity"] < 0.55 or c["fill"] < 0.55:
            continue
        if c.get("contrast", 0.0) < min_contrast:
            continue
        if c["radius"] >= ch_radius:  # a mounting hole can't be as big as the hub bore
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

    candidates = _detect_dark_blobs(gray, disc_mask, oradius)
    if not candidates:
        raise FeatureDetectionError("No hole candidates detected inside disc.")

    center_hole = _select_center_hole(candidates, (ocx, ocy))
    mounting = _select_mounting_holes(candidates, center_hole, expected_holes)

    geo = DiscGeometry(
        outer_circle=(ocx, ocy, oradius),
        center_hole=(center_hole["center"][0], center_hole["center"][1], center_hole["radius"]),
        mounting_holes=[(m["center"][0], m["center"][1], m["radius"]) for m in mounting],
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
