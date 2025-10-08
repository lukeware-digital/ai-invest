#!/bin/bash
# ==========================================
# CeciAI - Docker Entrypoint
# Script de inicialização do container
# ==========================================

set -e

echo "🚀 Starting CeciAI..."

# Verificar variáveis de ambiente obrigatórias
if [ -z "$COINAPI_KEY" ]; then
    echo "⚠️  WARNING: COINAPI_KEY not set"
fi

# Criar diretórios necessários
mkdir -p /app/data/{historical,realtime,cache,models}
mkdir -p /app/logs

# Verificar conexão com Ollama
if [ ! -z "$OLLAMA_HOST" ]; then
    echo "🔍 Checking Ollama connection..."
    until curl -f "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; do
        echo "⏳ Waiting for Ollama..."
        sleep 2
    done
    echo "✅ Ollama is ready"
fi

# Verificar conexão com Redis
if [ ! -z "$REDIS_HOST" ]; then
    echo "🔍 Checking Redis connection..."
    until python -c "import redis; r = redis.Redis(host='$REDIS_HOST'); r.ping()" > /dev/null 2>&1; do
        echo "⏳ Waiting for Redis..."
        sleep 2
    done
    echo "✅ Redis is ready"
fi

# Executar comando
echo "✅ All checks passed. Starting application..."
exec "$@"
