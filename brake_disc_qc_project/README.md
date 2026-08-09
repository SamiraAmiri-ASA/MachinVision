# Brake Disc QC

ابزار خط فرمان برای تشخیص هندسه، اندازه‌گیری و عیب‌های سطحی دیسک ترمز از تصویر عمودی.

## راه‌اندازی

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/pytest -q
```

بازرسی با مقیاس معلوم:

```bash
.venv/bin/brake-disc-qc --image disc.png --pixels-per-mm 8 --output-dir output
```

اگر `--pixels-per-mm` حذف شود، کانفیگ پیش‌فرض با استفاده از قطر اسمی سوراخ مرکزی
مقیاس را تخمین می‌زند و این موضوع را در هشدارهای گزارش ثبت می‌کند. برای یک مدل مشخص از
`--model disc_A123` یا برای فایل سفارشی از `--spec path/to/spec.yaml` استفاده کنید.

خروجی شامل تصویر حاشیه‌نویسی‌شده، ماسک عیب و گزارش JSON در پوشه‌ی خروجی است.

## اجرا با Docker

ساخت ایمیج:

```bash
docker build -t brake-disc-qc .
```

تصاویر ورودی را در پوشه‌ی `data` قرار دهید و فرمان زیر را اجرا کنید:

```bash
mkdir -p data output
docker run --rm \
  -v "$(pwd)/data:/data:ro" \
  -v "$(pwd)/output:/output" \
  brake-disc-qc \
  --image /data/disc.png \
  --pixels-per-mm 8 \
  --output-dir /output
```

همین کار با Compose:

```bash
mkdir -p data output
docker compose run --rm inspector \
  --image /data/disc.png \
  --pixels-per-mm 8 \
  --output-dir /output
```

کانتینر با کاربر غیر‌ریشه اجرا می‌شود. برای استفاده از مدل آماده نیز می‌توان گزینه‌ی
`--model disc_A123` یا `--model disc_B456` را به فرمان اضافه کرد.

## Live mobile camera inspection

`cli/live_inspect.py` is a separate, additive entrypoint that captures frames from a
mobile-phone camera (or any webcam/stream) and runs each selected frame through the
same `BrakeDiscInspector` pipeline used by the offline `cli/inspect.py`. It does not
change offline CLI behavior or any detection/measurement/surface logic.

Acceptable camera sources:

- **Android IP Webcam URL** (or DroidCam / Iriun Webcam / any OpenCV-compatible stream):

  ```bash
  .venv/bin/python -m cli.live_inspect \
    --source http://192.168.1.25:8080/video \
    --model disc_A123 \
    --output-dir ./output/live \
    --pixels-per-mm 2.2 \
    --display
  ```

- **Local webcam index** (when the phone appears as a system webcam, or for a laptop camera):

  ```bash
  .venv/bin/python -m cli.live_inspect --source 0 --model disc_A123 --display
  ```

- **Snapshot URL** (repeatedly fetches a single still image, e.g. IP Webcam's `/shot.jpg`):

  ```bash
  .venv/bin/python -m cli.live_inspect \
    --source http://192.168.1.25:8080/shot.jpg \
    --model disc_A123 \
    --snapshot-url
  ```

- **Manual capture mode** (inspect only when you press `s`; press `q` to quit):

  ```bash
  .venv/bin/python -m cli.live_inspect --source 0 --model disc_A123 --display --manual
  ```

  `--manual` and `--display` need a real OpenCV GUI build (`opencv-python`) to capture
  key presses. This project depends on `opencv-python-headless`, which has no GUI
  support; if no GUI is detected, manual mode automatically falls back to a terminal
  prompt (`s` + Enter to inspect, `q` + Enter to quit) instead of crashing.

### `--pixels-per-mm` (scale calibration)

`--pixels-per-mm` defaults to **2.2**, which is only an **initial estimate** taken from
one reference image of `disc_A123`. It is **not** a calibrated value: mobile camera
distance, zoom, and lens distortion all change the real px/mm ratio. Before trusting
any dimensional QC result from the live pipeline, measure the true pixels-per-mm for
your specific camera/distance setup (e.g. from a known reference dimension in the
frame) and pass it explicitly:

```bash
.venv/bin/python -m cli.live_inspect --source 0 --model disc_A123 --pixels-per-mm 6.4
```

### Output layout

Each inspected frame gets a timestamped, sequence-numbered tag (e.g.
`20260809_143012_001`) so results are never overwritten:

```
output/live/
  frames/20260809_143012_001.jpg            # only with --save-frames
  reports/20260809_143012_001.json
  reports/20260809_143012_001.csv
  annotated/20260809_143012_001_annotated.jpg
  annotated/20260809_143012_001_defect_mask.png
```

Without `--save-frames`, the raw captured frame is written to a single reused scratch
file instead of being kept per-inspection (the pipeline still needs a file on disk to
read from). A failed inspection on one frame is logged and skipped; it does not stop
the live loop.

### Developer note

To try live inspection end-to-end without a real phone, use your laptop's webcam in
manual mode:

```bash
python -m cli.live_inspect --source 0 --model disc_A123 --pixels-per-mm 2.2 --display --manual
```

Press `s` to inspect the current frame, `q` to quit. Check `./output/live/reports/` for
the JSON/CSV results and `./output/live/annotated/` for the annotated image.
