from __future__ import annotations
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    ERROR = "ERROR"
    UNCERTAIN = "UNCERTAIN"
    IMAGE_REJECTED = "IMAGE_REJECTED"


class InspectionResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"
    IMAGE_REJECTED = "IMAGE_REJECTED"


@dataclass
class DiscGeometry:
    outer_circle: tuple[float, float, float]
    inner_circle: tuple[float, float, float]
    mounting_holes: list[tuple[float, float, float]] = field(default_factory=list)
    pixels_per_mm: float | None = None
    rotation_deg: float | None = None
    confidence: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class MeasurementResult:
    name: str
    nominal: float
    actual: float
    tolerance: float
    deviation: float
    status: Status
    unit: str = "mm"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HoleResult:
    index: int
    center: tuple[float, float]
    diameter_mm: float
    nominal_diameter_mm: float | None = None
    diameter_deviation_mm: float | None = None
    radial_distance_mm: float | None = None
    nominal_radius_mm: float | None = None
    radial_deviation_mm: float | None = None
    status: Status = Status.PASS
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DimensionalQCResult:
    status: Status
    measurements: dict[str, Any] = field(default_factory=dict)
    deviations: dict[str, Any] = field(default_factory=dict)
    holes: list[HoleResult] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DefectResult:
    defect_type: str = "unknown"
    area_px: float = 0.0
    area_mm2: float | None = None
    centroid: tuple[float, float] | None = None
    bounding_box: tuple[int, int, int, int] | None = None
    severity: str = "unknown"
    score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SurfaceQCResult:
    status: Status
    defects: list[DefectResult] = field(default_factory=list)
    defect_count: int = 0
    mask_path: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageQualityResult:
    status: Status
    blur_score: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationResult:
    pixels_per_mm: float | None = None
    source: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class InspectionReport:
    inspection_id: str
    timestamp: str
    part_number: str
    image_path: str
    overall_result: InspectionResult | Status
    image_quality: Any = None
    calibration: Any = None
    warnings: list[str] = field(default_factory=list)
    dimensional_qc: DimensionalQCResult | None = None
    surface_qc: SurfaceQCResult | None = None
    annotated_image_path: str = ""
    defect_mask_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> InspectionResult | Status:
        return self.overall_result

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return asdict(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)