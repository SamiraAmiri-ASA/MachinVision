from pathlib import Path
from typing import Any

import yaml

_REQUIRED_SECTIONS = ("quality_thresholds", "nominal_dimensions", "tolerances", "surface_qc")


def load_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path)

    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    with spec_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise TypeError(f"Invalid spec YAML format: {spec_path}")

    missing = [key for key in ("part_number", *_REQUIRED_SECTIONS) if key not in data]
    if missing:
        raise ValueError(f"Spec {spec_path} is missing required keys: {', '.join(missing)}")

    return data
