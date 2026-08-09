import argparse
import json
from pathlib import Path

from brake_disc_qc.calibration import CameraCalibration
from brake_disc_qc.config import load_spec
from brake_disc_qc.inspector import BrakeDiscInspector

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _spec_path(model: str | None, spec: Path | None) -> Path:
    if spec is not None:
        return spec
    if model:
        candidate = Path(model)
        if candidate.suffix in {".yaml", ".yml"} or candidate.parent != Path("."):
            return candidate
        return PROJECT_ROOT / "config" / "models" / f"{model}.yaml"
    return PROJECT_ROOT / "config" / "default_spec.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a brake-disc image")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model", help="model name (for example disc_A123) or YAML path")
    parser.add_argument("--spec", type=Path, help="explicit YAML spec; overrides --model")
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--pixels-per-mm", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--output", type=Path, help="optional additional report JSON path")

    args = parser.parse_args()

    spec = load_spec(_spec_path(args.model, args.spec))
    calibration = CameraCalibration(
        pixels_per_mm=args.pixels_per_mm,
        source="cli" if args.pixels_per_mm is not None else "none",
    )

    inspector = BrakeDiscInspector(
        config=spec,
        calibration=calibration,
        output_dir=str(args.output_dir),
        debug=False,
    )

    report = inspector.inspect(
        image_path=str(args.image),
        reference_path=str(args.reference) if args.reference else None,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"Overall result: {report.overall_result.value}")
    if report.annotated_image_path:
        print(f"Annotated image: {report.annotated_image_path}")


if __name__ == "__main__":
    main()
