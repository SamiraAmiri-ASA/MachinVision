"""Renders the annotated inspection image and QC summary panel."""
from __future__ import annotations

import numpy as np
import cv2

from .models import DiscGeometry, DimensionalQCResult, SurfaceQCResult, Status, InspectionReport

# BGR colors
BLUE = (255, 0, 0)
GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 220, 220)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

_STATUS_COLOR = {Status.PASS: GREEN, Status.FAIL: RED, Status.UNCERTAIN: YELLOW}


def _label(img, text, org, color=WHITE, scale=0.5, bg=BLACK):
    """Draw text with a filled background box for legibility."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 4, y + 2), bg, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def annotate(
    img: np.ndarray,
    geo: DiscGeometry,
    dim: DimensionalQCResult,
    surf: SurfaceQCResult,
    report: InspectionReport,
) -> np.ndarray:
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    ocx, ocy, oradius = geo.outer_circle
    ch_x, ch_y, ch_r = geo.center_hole

    # Outer boundary (blue).
    cv2.circle(out, (int(ocx), int(ocy)), int(oradius), BLUE, 2)

    # Center hole (green) + label.
    cv2.circle(out, (int(ch_x), int(ch_y)), int(ch_r), GREEN, 2)
    cv2.drawMarker(out, (int(ch_x), int(ch_y)), GREEN, cv2.MARKER_CROSS, 14, 2)
    if dim.center_hole:
        s = dim.center_hole.status.value
        _label(out, f"Center Dia: {dim.center_hole.diameter_mm:.2f} mm [{s}]",
               (int(ch_x) + 10, int(ch_y) - 10), _STATUS_COLOR[dim.center_hole.status])

    # Mounting holes (red) with IDs, radial and cross lines.
    for hole in dim.mounting_holes:
        mx, my = hole.center_px
        cv2.circle(out, (int(mx), int(my)), int(hole.diameter_px / 2), RED, 2)
        cv2.drawMarker(out, (int(mx), int(my)), RED, cv2.MARKER_CROSS, 12, 2)
        cv2.line(out, (int(ch_x), int(ch_y)), (int(mx), int(my)), YELLOW, 1)
        _label(out, f"{hole.hole_id} {hole.diameter_mm:.2f}mm",
               (int(mx) + 8, int(my) - 8), _STATUS_COLOR[hole.status])

    # Radial distance labels at line midpoints.
    for hole, rm in zip(dim.mounting_holes, dim.radial_measurements):
        mx, my = hole.center_px
        midx, midy = int((ch_x + mx) / 2), int((ch_y + my) / 2)
        _label(out, f"{rm.measured_value:.2f}mm", (midx, midy), _STATUS_COLOR[rm.status], 0.45)

    # Cross lines between opposing holes.
    n = len(dim.mounting_holes)
    if n >= 2 and n % 2 == 0:
        half = n // 2
        for i, cm in enumerate(dim.cross_measurements):
            a = dim.mounting_holes[i].center_px
            b = dim.mounting_holes[i + half].center_px
            cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), BLUE, 1)
            midx, midy = int((a[0] + b[0]) / 2), int((a[1] + b[1]) / 2)
            _label(out, f"X:{cm.measured_value:.2f}mm", (midx, midy + 15),
                   _STATUS_COLOR[cm.status], 0.45)

    # Surface defects: semi-transparent red overlay + labels.
    overlay = out.copy()
    for d in surf.defects:
        x, y, w, h = d.bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), RED, -1)
        _label(out, f"D{d.defect_id} {d.area_mm2:.2f}mm2",
               (x, y - 4), RED, 0.4)
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

    _draw_summary_panel(out, dim, surf, report)
    return out


def _draw_summary_panel(out, dim, surf, report: InspectionReport):
    """QC summary + metadata panel in the top-left corner."""
    h, w = out.shape[:2]
    panel_w, panel_h = 300, 210
    cv2.rectangle(out, (0, 0), (panel_w, panel_h), BLACK, -1)
    cv2.rectangle(out, (0, 0), (panel_w, panel_h), WHITE, 1)

    def row(y, key, status: Status):
        cv2.putText(out, f"{key}:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        cv2.putText(out, status.value, (170, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    _STATUS_COLOR[status], 2, cv2.LINE_AA)

    center_st = dim.center_hole.status if dim.center_hole else Status.UNCERTAIN
    holes_st = _agg([hs.status for hs in dim.mounting_holes])
    cross_st = _agg([m.status for m in dim.cross_measurements])
    radial_st = _agg([m.status for m in dim.radial_measurements])

    row(30, "CENTER", center_st)
    row(52, "HOLES", holes_st)
    row(74, "CROSS", cross_st)
    row(96, "RADIAL", radial_st)
    row(118, "SURFACE", surf.status)

    overall = _map_overall(report.overall_result)
    cv2.putText(out, f"OVERALL: {report.overall_result.value}", (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, overall, 2, cv2.LINE_AA)

    meta = [
        f"PN: {report.part_number}",
        f"ID: {report.inspection_id[:8]}",
        f"Calib: {'VALID' if report.calibration.valid else 'INVALID'}",
        report.timestamp[:19],
    ]
    for i, line in enumerate(meta):
        cv2.putText(out, line, (10, 168 + i * 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, WHITE, 1, cv2.LINE_AA)


def _agg(statuses):
    if not statuses:
        return Status.UNCERTAIN
    if Status.FAIL in statuses:
        return Status.FAIL
    if Status.UNCERTAIN in statuses:
        return Status.UNCERTAIN
    return Status.PASS


def _map_overall(result):
    name = result.value
    if "PASS" in name:
        return GREEN
    if "FAIL" in name or "REJECT" in name:
        return RED
    return YELLOW