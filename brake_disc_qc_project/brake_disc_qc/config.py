from pathlib import Path
from typing import Any

import yaml


def load_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path)

    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    with spec_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid spec YAML format: {spec_path}")

    return data
