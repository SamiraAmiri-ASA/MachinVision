"""Dimensional QC evaluation: converts geometry to mm and applies tolerances."""
from __future__ import annotations

import logging

import numpy as np

from .models import (
    DimensionalQCResult,
    DiscGeometry,
    HoleResult,
    MeasurementResult,
    Status,
)

logger = logging.getLogger(__name__)


def _evaluate(name: str, measured: float, nominal: float, tol: float,
              confidence: float, unit: str = "mm") -> MeasurementResult:
    dev = abs(measured - nominal)
    status = Status.PASS if dev <= tol else Status.FAIL
    if confidence < 0.5:
        status = Status.UNCERTAIN
    return MeasurementResult(
        name=name, measured_value=round(measured, 4), nominal_value=nominal,
        deviation=round(dev, 4), tolerance=tol, status=status,
        confidence=round(confidence, 3), unit=unit,
    )


def _confidence_from_meta(meta: dict) -> float:
    """Blend circularity and fill ratio into a bounded confidence score."""
    return float(np.clip(0.5 * meta["circularity"] + 0.5 * meta["fill"], 0.0, 1.0))


def evaluate_dimensions(geo: DiscGeometry, config: dict, calibration_valid: bool) -> DimensionalQCResult:
    """Produce a full dimensional QC result.

    Millimeter values are only emitted when calibration_valid is True, per the
    requirement that physical values require valid calibration.
    """
    nom = config["nominal_dimensions"]
    tol = config["tolerances"]
    ppm = geo.pixels_per_mm
    if calibration_valid and (ppm is None or ppm <= 0):
        raise ValueError("A positive pixels_per_mm value is required for calibrated measurements")
    scale = ppm if ppm is not None and ppm > 0 else 1.0

    def to_mm(px: float) -> float:
        return px / scale if calibration_valid else float("nan")

    ch_x, ch_y, ch_r = geo.center_hole
    ch_meta = getattr(geo, "_center_meta", {"circularity": 0.0, "fill": 0.0})
    ch_conf = _confidence_from_meta(ch_meta)

    center_hole = HoleResult(
        hole_id="CENTER",
        center_px=(ch_x, ch_y),
        center_mm=(0.0, 0.0),  # center hole is the coordinate origin
        diameter_px=2 * ch_r,
        diameter_mm=to_mm(2 * ch_r),
        circularity=ch_meta["circularity"],
        confidence=ch_conf,
        status=Status.PASS,
    )
    center_meas = _evaluate(
        "center_hole_diameter", to_mm(2 * ch_r),
        nom["center_hole_diameter"], tol["center_hole_diameter"], ch_conf,
    )
    center_hole.status = center_meas.status

    mounting_meta = getattr(geo, "_mounting_meta", [])
    mounting_results: list[HoleResult] = []
    radial_meas: list[MeasurementResult] = []
    measurements: list[MeasurementResult] = [center_meas]

    for idx, (mx, my, mr) in enumerate(geo.mounting_holes, start=1):
        meta = mounting_meta[idx - 1] if idx - 1 < len(mounting_meta) else {"circularity": 0.0, "fill": 0.0}
        conf = _confidence_from_meta(meta)
        radial_px = float(np.hypot(mx - ch_x, my - ch_y))

        dia_meas = _evaluate(
            f"H{idx}_diameter", to_mm(2 * mr),
            nom["mounting_hole_diameter"], tol["mounting_hole_diameter"], conf,
        )
        rad_meas = _evaluate(
            f"H{idx}_radial", to_mm(radial_px),
            nom["mounting_hole_radial_distance"], tol["mounting_hole_radial_distance"], conf,
        )
        radial_meas.append(rad_meas)
        measurements.extend([dia_meas, rad_meas])

        hole_status = Status.PASS
        if Status.FAIL in (dia_meas.status, rad_meas.status):
            hole_status = Status.FAIL
        elif Status.UNCERTAIN in (dia_meas.status, rad_meas.status):
            hole_status = Status.UNCERTAIN

        mounting_results.append(HoleResult(
            hole_id=f"H{idx}",
            center_px=(mx, my),
            center_mm=((mx - ch_x) / scale, (my - ch_y) / scale)
            if calibration_valid
            else (float("nan"), float("nan")),
            diameter_px=2 * mr,
            diameter_mm=to_mm(2 * mr),
            circularity=meta["circularity"],
            confidence=conf,
            status=hole_status,
        ))

    # Cross measurements: opposing pairs (i, i+N/2) after angular sort.
    cross_meas: list[MeasurementResult] = []
    n = len(geo.mounting_holes)
    if n >= 2 and n % 2 == 0:
        half = n // 2
        for i in range(half):
            a = geo.mounting_holes[i]
            b = geo.mounting_holes[i + half]
            cross_px = float(np.hypot(a[0] - b[0], a[1] - b[1]))
            conf = min(
                _confidence_from_meta(mounting_meta[i]) if i < len(mounting_meta) else 0.0,
                _confidence_from_meta(mounting_meta[i + half]) if i + half < len(mounting_meta) else 0.0,
            )
            cm = _evaluate(
                f"cross_H{i+1}_H{i+half+1}", to_mm(cross_px),
                nom["opposing_hole_cross_distance"], tol["opposing_hole_cross_distance"], conf,
            )
            cross_meas.append(cm)
            measurements.append(cm)

    # Angular spacing consistency.
    angular_meas: list[MeasurementResult] = []
    angles = sorted(
        float(np.degrees(np.arctan2(my - ch_y, mx - ch_x)) % 360)
        for mx, my, _ in geo.mounting_holes
    )
    if len(angles) >= 2:
        expected_step = 360.0 / len(angles)
        for i in range(len(angles)):
            step = (angles[(i + 1) % len(angles)] - angles[i]) % 360
            am = _evaluate(
                f"angular_step_{i+1}", step, expected_step,
                tol["angular_spacing_deg"], 0.9, unit="deg",
            )
            angular_meas.append(am)

    # Concentricity: offset of bolt-circle centroid from center hole.
    concentricity = None
    if geo.mounting_holes:
        cxs = np.mean([m[0] for m in geo.mounting_holes])
        cys = np.mean([m[1] for m in geo.mounting_holes])
        offset_px = float(np.hypot(cxs - ch_x, cys - ch_y))
        concentricity = _evaluate(
            "concentricity", to_mm(offset_px), 0.0,
            tol["concentricity_mm"], 0.9,
        )
        measurements.append(concentricity)

    # Aggregate dimensional status.
    all_status = [m.status for m in measurements]
    if Status.UNCERTAIN in all_status or not calibration_valid:
        overall = Status.UNCERTAIN if not calibration_valid else (
            Status.FAIL if Status.FAIL in all_status else Status.UNCERTAIN
        )
    elif Status.FAIL in all_status:
        overall = Status.FAIL
    else:
        overall = Status.PASS

    return DimensionalQCResult(
        status=overall,
        center_hole=center_hole,
        mounting_holes=mounting_results,
        radial_measurements=radial_meas,
        cross_measurements=cross_meas,
        angular_spacing=angular_meas,
        concentricity=concentricity,
        measurements=measurements,
    )
