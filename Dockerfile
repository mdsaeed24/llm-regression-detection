# Prompt regression runner: CLI, golden dataset, prompts, reporting.
#
#   docker build -t regress .
#   docker run --rm -e DEEPSEEK_API_KEY -v "$PWD/out:/data" regress \
#       run --prompt prompts/v2.yaml --baseline baselines/baseline.json
#
# Secrets come from the environment (never baked into the image). Any setting
# can be overridden with REGRESS__<SECTION>__<KEY>, e.g.
#   -e REGRESS__THRESHOLDS__ACCURACY_CRITICAL_PP=5
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEEPEVAL_TELEMETRY_OPT_OUT=1

WORKDIR /app

# Dependencies first so code changes do not invalidate the install layer.
COPY pyproject.toml ./
RUN mkdir regress && touch regress/__init__.py \
    && pip install . \
    && pip uninstall -y regress && rm -rf regress build

COPY regress ./regress
RUN pip install --no-deps .

COPY config.yaml ./
COPY prompts ./prompts
COPY data/golden.json ./data/golden.json
COPY baselines ./baselines

# Run history and reports go to a mounted volume, not the image.
ENV REGRESS__STORAGE__DB_PATH=/data/runs.sqlite \
    REGRESS__REPORTING__OUTPUT_DIR=/data/reports
RUN useradd --create-home --uid 10001 regress && mkdir /data && chown regress /data
USER 10001
VOLUME ["/data"]

ENTRYPOINT ["python", "-m", "regress"]
CMD ["--help"]
