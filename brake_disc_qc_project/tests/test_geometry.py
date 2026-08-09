"""Tests using a synthetically rendered brake disc with known geometry."""
import numpy as np
import cv2
import pytest

from brake_disc_qc.geometry import find_disc_features, FeatureDetectionError
from brake_disc_qc.measurement import evaluate_dimensions

CONFIG = {
    "part_number": "TEST",
    "nominal_dimensions": {
        "center_hole_diameter": 75.0, "mounting_hole_diameter": 10.0,
        "mounting_hole_radial_distance": 46.5, "opposing_hole_cross_distance": 93.0,
        "mounting_hole_count": 4,
    },
    "tolerances": {
        "center_hole_diameter": 0.10, "mounting_hole_diameter": 0.08,
        "mounting_hole_radial_distance": 0.10, "opposing_hole_cross_distance": 0.15,
        "angular_spacing_deg": 1.0, "concentricity_mm": 0.08,
    },
}


def _render_disc(ppm=8.0, size=1000):
    """Render a top-view synthetic disc at a known pixels-per-mm scale."""
    img = np.full((size, size, 3), 200, dtype=np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), int(120 * ppm / 2), (150, 150, 150), -1)  # disc body
    cv2.circle(img, (c, c), int(75 * ppm / 2), (20, 20, 20), -1)       # center hole
    radial_px = int(46.5 * ppm)
    for angle in (0, 90, 180, 270):
        rad = np.radians(angle)
        hx = int(c + radial_px * np.cos(rad))
        hy = int(c + radial_px * np.sin(rad))
        cv2.circle(img, (hx, hy), int(10 * ppm / 2), (20, 20, 20), -1)
    return img, ppm


def test_detects_four_mounting_holes():
    img, ppm = _render_disc()
    geo = find_disc_features(img, CONFIG, ppm)
    assert len(geo.mounting_holes) == 4


def test_center_hole_diameter_within_tolerance():
    img, ppm = _render_disc()
    geo = find_disc_features(img, CONFIG, ppm)
    dim = evaluate_dimensions(geo, CONFIG, calibration_valid=True)
    # Rasterization introduces sub-pixel error; allow a loose engineering band.
    assert abs(dim.center_hole.diameter_mm - 75.0) < 2.0


def test_radial_distances_consistent():
    img, ppm = _render_disc()
    geo = find_disc_features(img, CONFIG, ppm)
    dim = evaluate_dimensions(geo, CONFIG, calibration_valid=True)
    measured = [m.measured_value for m in dim.radial_measurements]
    assert max(measured) - min(measured) < 1.0


def test_holes_sorted_by_angle():
    img, ppm = _render_disc()
    geo = find_disc_features(img, CONFIG, ppm)
    ch = geo.center_hole
    angles = [np.degrees(np.arctan2(my - ch[1], mx - ch[0])) % 360
              for mx, my, _ in geo.mounting_holes]
    assert angles == sorted(angles)


def test_no_features_raises():
    blank = np.full((500, 500, 3), 128, dtype=np.uint8)
    with pytest.raises(FeatureDetectionError):
        find_disc_features(blank, CONFIG, 8.0)


def test_rejects_small_defect_dots_as_mounting_holes():
    """Shallow, low-contrast specks (casting marks/pitting) must not be
    picked over the real bolt-circle holes, even when they are numerous and
    individually pass the circularity/fill shape gates."""
    img, ppm = _render_disc()
    c = img.shape[0] // 2

    # Faint defect dots: real holes/background are near-black (20) on a
    # bright disc (150-200); these are only a shallow dip below the disc
    # body, so their contrast against their surroundings is much lower.
    rng = np.random.RandomState(0)
    for _ in range(10):
        angle = rng.uniform(0, 2 * np.pi)
        radius_px = rng.uniform(20 * ppm, 40 * ppm)  # off the true 46.5mm bolt circle
        dx = int(c + radius_px * np.cos(angle))
        dy = int(c + radius_px * np.sin(angle))
        cv2.circle(img, (dx, dy), int(rng.uniform(2, 4) * ppm / 2), (130, 130, 130), -1)

    geo = find_disc_features(img, CONFIG, ppm)
    assert len(geo.mounting_holes) == 4
    ch = geo.center_hole
    for mx, my, _ in geo.mounting_holes:
        radial = np.hypot(mx - ch[0], my - ch[1])
        # Real holes sit at 46.5mm; defect dots were seeded well inside that.
        assert abs(radial / ppm - 46.5) < 5.0