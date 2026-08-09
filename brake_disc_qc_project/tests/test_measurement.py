"""Tests for dimensional QC evaluation against a synthetically rendered disc."""
import numpy as np
import cv2
import pytest

from brake_disc_qc.geometry import find_disc_features
from brake_disc_qc.measurement import evaluate_dimensions
from brake_disc_qc.models import Status

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


def _render_disc(ppm=8.0, size=1000, hole_count=4):
    """Render a top-view synthetic disc with `hole_count` mounting holes."""
    img = np.full((size, size, 3), 200, dtype=np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), int(120 * ppm / 2), (150, 150, 150), -1)
    cv2.circle(img, (c, c), int(75 * ppm / 2), (20, 20, 20), -1)
    radial_px = int(46.5 * ppm)
    for i in range(hole_count):
        rad = np.radians(360.0 * i / hole_count)
        hx = int(c + radial_px * np.cos(rad))
        hy = int(c + radial_px * np.sin(rad))
        cv2.circle(img, (hx, hy), int(10 * ppm / 2), (20, 20, 20), -1)
    return img, ppm


def test_dimensional_qc_not_null_when_geometry_succeeds():
    img, ppm = _render_disc()
    geo = find_disc_features(img, CONFIG, ppm)
    dim = evaluate_dimensions(geo, CONFIG, calibration_valid=True)
    assert dim is not None
    assert dim.center_hole is not None
    assert len(dim.mounting_holes) == CONFIG["nominal_dimensions"]["mounting_hole_count"]


def test_supports_configured_mounting_hole_count():
    config = {
        **CONFIG,
        "nominal_dimensions": {**CONFIG["nominal_dimensions"], "mounting_hole_count": 6},
    }
    img, ppm = _render_disc(hole_count=6)
    geo = find_disc_features(img, config, ppm)
    dim = evaluate_dimensions(geo, config, calibration_valid=True)
    assert len(dim.mounting_holes) == 6
    # Opposite-pair cross distances only apply cleanly to an even hole count.
    assert len(dim.cross_measurements) == 3


def test_missing_scale_yields_uncertain_not_bad_numbers():
    img, ppm = _render_disc()
    geo = find_disc_features(img, CONFIG, ppm)
    dim = evaluate_dimensions(geo, CONFIG, calibration_valid=False)
    assert dim.status == Status.UNCERTAIN
    assert np.isnan(dim.center_hole.diameter_mm)


def test_pixels_per_mm_required_when_calibration_valid():
    img, ppm = _render_disc()
    geo = find_disc_features(img, CONFIG, ppm)
    geo.pixels_per_mm = None
    with pytest.raises(ValueError):
        evaluate_dimensions(geo, CONFIG, calibration_valid=True)
