"""End-to-end inspection pipeline with real QC decision logic."""
from __future__ import annotations

import logging
import datetime
import uuid
import json
from pathlib import Path
from typing import Optional

import cv2

from .models import (
    InspectionReport, InspectionResult, Status, DimensionalQCResult, SurfaceQCResult,
)
from .utils import load_image, check_image_quality
from .calibration import CameraCalibration
from .geometry import find_disc_features, FeatureDetectionError
from .measurement import evaluate_dimensions
from .surface import detect_surface_defects
from .annotator import annotate

logger = logging.getLogger(__name__)


class BrakeDiscInspector:
    def __init__(self, config: dict, calibration: CameraCalibration,
                 output_dir: str = "output", debug: bool = False):
        self.config = config
        self.calibration = calibration
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug

    def inspect(self, image_path: str, reference_path: Optional[str] = None) -> InspectionReport:
        inspection_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        qt = self.config["quality_thresholds"]

        img = load_image(image_path)
        img = self.calibration.undistort(img)  # no-op if intrinsics absent

        quality = check_image_quality(
            img,
            min_sharpness=qt["min_image_sharpness"],
            min_disc_area_fraction=qt.get("min_disc_area_fraction", 0.10),
        )

        base = InspectionReport(
            inspection_id=inspection_id, timestamp=timestamp,
            part_number=self.config["part_number"], image_path=image_path,
            overall_result=InspectionResult.IMAGE_REJECTED,
            image_quality=quality, calibration=self.calibration.get_status(),
        )

        if not quality.acceptable:
            base.warnings.append(quality.rejection_reason or "Image rejected.")
            return base

        # Feature detection.
        try:
            geo = find_disc_features(img, self.config, self.calibration.pixels_per_mm or 1.0)
        except FeatureDetectionError as e:
            base.overall_result = InspectionResult.INSPECTION_UNCERTAIN
            base.warnings.append(f"Feature detection failed: {e}")
            return base

        # Feature-based scale calibration (only if explicitly permitted).
        calibration_valid = self.calibration.pixels_per_mm is not None
        if not calibration_valid and self.config.get("allow_feature_scale", False):
            ch_r = geo.center_hole[2]
            nominal_d = self.config["nominal_dimensions"]["center_hole_diameter"]
            self.calibration.set_scale(2 * ch_r, nominal_d, "center_hole_feature")
            geo.pixels_per_mm = self.calibration.pixels_per_mm
            calibration_valid = True
            base.warnings.append("Scale derived from nominal center-hole diameter (documented fallback).")

        base.calibration = self.calibration.get_status()

        # Dimensional QC.
        dim = evaluate_dimensions(geo, self.config, calibration_valid)

        # Surface QC.
        reference = load_image(reference_path) if reference_path else None
        surf, defect_mask = detect_surface_defects(
            img, self.config, geo, calibration_valid, reference=reference
        )

        base.dimensional_qc = dim
        base.surface_qc = surf
        base.overall_result = self._decide_overall(dim, surf, calibration_valid)

        # Outputs.
        annotated = annotate(img, geo, dim, surf, base)
        annotated_path = self.output_dir / f"{inspection_id[:8]}_annotated.png"
        cv2.imwrite(str(annotated_path), annotated)
        base.annotated_image_path = str(annotated_path)

        mask_path = self.output_dir / f"{inspection_id[:8]}_defect_mask.png"
        cv2.imwrite(str(mask_path), defect_mask)
        base.defect_mask_path = str(mask_path)

        self._write_json(base)
        return base

    def _decide_overall(self, dim: DimensionalQCResult, surf: SurfaceQCResult,
                        calibration_valid: bool) -> InspectionResult:
        if not calibration_valid:
            return InspectionResult.INSPECTION_UNCERTAIN
        statuses = [dim.status, surf.status]
        if Status.FAIL in statuses:
            return InspectionResult.FAIL
        if Status.UNCERTAIN in statuses:
            return InspectionResult.UNCERTAIN
        return InspectionResult.PASS

    def _write_json(self, report: InspectionReport) -> None:
        path = self.output_dir / f"{report.inspection_id[:8]}_report.json"
        with open(path, "w") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2)
        logger.info("Report written: %s", path)