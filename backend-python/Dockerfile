FROM python:3.11-slim AS builder

# Variables de entorno para build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Instalar dependencias del sistema necesarias para compilar
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar solo requirements
COPY requirements.txt .

# Crear virtualenv e instalar dependencias
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalar torch CPU-only (mucho más ligero que con CUDA)
RUN pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Instalar resto de dependencias
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================
# Etapa 2: Runtime - Imagen final
# ==========================================
FROM python:3.11-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    TRANSFORMERS_CACHE=/app/.cache \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Instalar SOLO las dependencias runtime necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Para PDFs
    libpoppler-cpp-dev \
    poppler-utils \
    # Para OCR
    tesseract-ocr \
    tesseract-ocr-spa \
    libtesseract-dev \
    # Para health check
    curl \
    # Limpieza
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copiar virtualenv desde builder
COPY --from=builder /opt/venv /opt/venv

# Copiar código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p uploads .cache

# Usuario no-root (seguridad)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT}/health')"

# Exponer puerto
EXPOSE $PORT

# Comando de inicio
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]