# Tempo demo image. Build: docker build -t tempo .   Run: docker run -p 8000:10000 -e RADEON_API_KEY=... tempo
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TEMPO_HOST=0.0.0.0 TEMPO_TRUST_PROXY=1 PORT=10000

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Train the forecaster while building the image, so the server starts in seconds.
RUN python -c "from pathlib import Path; from tempo.api.server import load_models; load_models(Path('.cache/models.joblib'))"

RUN useradd --create-home app && mkdir -p data && chown -R app:app /app
USER app
EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','10000')+'/api/health',timeout=4)"
CMD ["python", "-m", "tempo.api.server"]
