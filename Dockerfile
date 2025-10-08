# ==========================================
# CeciAI - Dockerfile Otimizado
# Python 3.11 | Alta Performance | Baixo Custo
# ==========================================

# ========== STAGE 1: Builder ==========
FROM python:3.11.9-slim-bookworm AS builder

# Metadados
LABEL maintainer="CeciAI Team"
LABEL version="0.3.0"
LABEL description="Sistema inteligente de trading de criptomoedas"

# Variáveis de ambiente para otimização
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Instalar dependências do sistema (mínimas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório de trabalho
WORKDIR /build

# Copiar apenas requirements primeiro (cache layer)
COPY requirements.txt .

# Instalar dependências Python em /install
RUN pip install --prefix=/install --no-warn-script-location \
    -r requirements.txt

# ========== STAGE 2: Runtime ==========
FROM python:3.11.9-slim-bookworm AS runtime

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/install/bin:$PATH" \
    PYTHONOPTIMIZE=2

# Instalar apenas runtime essenciais
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Criar usuário não-root para segurança
RUN useradd -m -u 1000 -s /bin/bash ceciuser && \
    mkdir -p /app /app/data /app/logs && \
    chown -R ceciuser:ceciuser /app

# Copiar dependências do builder
COPY --from=builder /install /usr/local

# Definir diretório de trabalho
WORKDIR /app

# Copiar código da aplicação
COPY --chown=ceciuser:ceciuser . .

# Mudar para usuário não-root
USER ceciuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Expor porta (se necessário para API/Dashboard)
EXPOSE 8000

# Comando padrão
CMD ["python", "main.py"]

# ========== STAGE 3: Development (opcional) ==========
FROM runtime AS development

USER root

# Instalar ferramentas de desenvolvimento
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências de dev
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    ruff \
    mypy \
    ipython

USER ceciuser

CMD ["/bin/bash"]
