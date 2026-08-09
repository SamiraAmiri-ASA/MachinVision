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
