ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE} AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY pyproject.toml README.md ./
COPY brake_disc_qc ./brake_disc_qc
COPY cli ./cli

RUN python -m pip wheel --wheel-dir /wheels .


FROM ${PYTHON_IMAGE} AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN python -m pip install /wheels/*.whl && rm -rf /wheels

# Model specifications are data files read by the CLI at runtime.
COPY --chown=app:app config ./config

RUN mkdir /data /output && chown app:app /data /output

USER app

ENTRYPOINT ["brake-disc-qc"]
CMD ["--help"]
