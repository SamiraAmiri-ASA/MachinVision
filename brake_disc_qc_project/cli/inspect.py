from pathlib import Path
from dataclasses import asdict
import argparse
import json

from brake_disc_qc.calibration import CameraCalibration
from brake_disc_qc.config import load_spec
from brake_disc_qc.inspector import BrakeDiscInspector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--spec", default=Path("config/default_spec.yaml"), type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    spec = load_spec(args.spec)
    calibration = CameraCalibration()

    
    inspector = BrakeDiscInspector(
    config=spec,
    calibration=calibration,
    output_dir=str(args.output.parent),
    debug=False,
    )

    report = inspector.inspect(
        image_path=args.image,
        reference_path=args.reference,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)

    print(f"Overall result: {report.overall_result}")


if __name__ == "__main__":
    main()