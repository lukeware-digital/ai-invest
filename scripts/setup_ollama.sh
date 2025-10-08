#!/bin/bash
# ==========================================
# CeciAI - Setup Ollama + Modelos LLM
# Instala Ollama e baixa melhores modelos
# ==========================================

set -e

echo "╔══════════════════════════════════════════════════════╗"
echo "║                                                      ║"
echo "║     CeciAI - Setup Ollama + Modelos LLM             ║"
echo "║                                                      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Cores
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ==================== DETECTAR OS ====================

echo -e "${YELLOW}🔍 Detectando sistema operacional...${NC}"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo -e "${GREEN}✅ Linux detectado${NC}"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    echo -e "${GREEN}✅ macOS detectado${NC}"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
    echo -e "${GREEN}✅ Windows detectado${NC}"
else
    echo -e "${RED}❌ Sistema operacional não suportado: $OSTYPE${NC}"
    exit 1
fi

# ==================== INSTALAR OLLAMA ====================

echo ""
echo -e "${YELLOW}📦 Instalando Ollama...${NC}"

if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✅ Ollama já está instalado${NC}"
    ollama --version
else
    if [[ "$OS" == "linux" ]]; then
        # Linux
        echo -e "${YELLOW}Instalando Ollama no Linux...${NC}"
        curl -fsSL https://ollama.ai/install.sh | sh
        
    elif [[ "$OS" == "macos" ]]; then
        # macOS
        echo -e "${YELLOW}Instalando Ollama no macOS...${NC}"
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo -e "${RED}❌ Homebrew não encontrado. Instale manualmente: https://ollama.ai/download${NC}"
            exit 1
        fi
        
    elif [[ "$OS" == "windows" ]]; then
        # Windows
        echo -e "${YELLOW}Para Windows, baixe e instale manualmente:${NC}"
        echo -e "${YELLOW}https://ollama.ai/download/windows${NC}"
        exit 0
    fi
    
    echo -e "${GREEN}✅ Ollama instalado com sucesso${NC}"
fi

# ==================== INICIAR OLLAMA ====================

echo ""
echo -e "${YELLOW}🚀 Iniciando Ollama...${NC}"

# Verificar se Ollama está rodando
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}Iniciando serviço Ollama...${NC}"
    
    if [[ "$OS" == "linux" ]]; then
        # Linux - iniciar como serviço
        if command -v systemctl &> /dev/null; then
            sudo systemctl start ollama
            sudo systemctl enable ollama
        else
            # Iniciar em background
            nohup ollama serve > /tmp/ollama.log 2>&1 &
            sleep 3
        fi
    elif [[ "$OS" == "macos" ]]; then
        # macOS - iniciar em background
        nohup ollama serve > /tmp/ollama.log 2>&1 &
        sleep 3
    fi
    
    # Verificar se iniciou
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Ollama iniciado${NC}"
    else
        echo -e "${RED}❌ Erro ao iniciar Ollama${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Ollama já está rodando${NC}"
fi

# ==================== BAIXAR MODELOS LLM ====================

echo ""
echo -e "${YELLOW}📥 Baixando modelos LLM otimizados...${NC}"
echo ""

# Modelos recomendados (ordem: melhor custo-benefício)
declare -a MODELS=(
    "llama3.2:3b"           # 3B params - Rápido, leve (2GB RAM)
    "llama3.2:1b"           # 1B params - Ultra-rápido (1GB RAM)
    "qwen2.5:3b"            # 3B params - Excelente para análise
    "phi3:mini"             # 3.8B params - Microsoft, otimizado
    "gemma2:2b"             # 2B params - Google, eficiente
)

# Modelos opcionais (maiores, mais precisos)
declare -a OPTIONAL_MODELS=(
    "llama3.2:8b"           # 8B params - Mais preciso (8GB RAM)
    "mistral:7b"            # 7B params - Excelente qualidade
    "codellama:7b"          # 7B params - Especializado em código
)

echo -e "${GREEN}Modelos principais (recomendados):${NC}"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done

echo ""
echo -e "${YELLOW}Modelos opcionais (maiores):${NC}"
for model in "${OPTIONAL_MODELS[@]}"; do
    echo "  - $model"
done

echo ""
read -p "Baixar todos os modelos principais? (s/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Ss]$ ]]; then
    for model in "${MODELS[@]}"; do
        echo ""
        echo -e "${YELLOW}📥 Baixando $model...${NC}"
        
        if ollama pull "$model"; then
            echo -e "${GREEN}✅ $model baixado${NC}"
        else
            echo -e "${RED}❌ Erro ao baixar $model${NC}"
        fi
    done
else
    echo -e "${YELLOW}Baixando apenas llama3.2:3b (recomendado)...${NC}"
    ollama pull llama3.2:3b
fi

echo ""
read -p "Baixar modelos opcionais (maiores)? (s/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Ss]$ ]]; then
    for model in "${OPTIONAL_MODELS[@]}"; do
        echo ""
        echo -e "${YELLOW}📥 Baixando $model...${NC}"
        
        if ollama pull "$model"; then
            echo -e "${GREEN}✅ $model baixado${NC}"
        else
            echo -e "${RED}❌ Erro ao baixar $model${NC}"
        fi
    done
fi

# ==================== TESTAR MODELOS ====================

echo ""
echo -e "${YELLOW}🧪 Testando modelos...${NC}"

# Listar modelos instalados
echo ""
echo -e "${GREEN}Modelos instalados:${NC}"
ollama list

# Testar modelo principal
echo ""
echo -e "${YELLOW}Testando llama3.2:3b...${NC}"
echo ""

RESPONSE=$(ollama run llama3.2:3b "What is 2+2? Answer with just the number." --verbose=false 2>/dev/null || echo "Error")

if [[ "$RESPONSE" == *"4"* ]]; then
    echo -e "${GREEN}✅ Modelo funcionando corretamente${NC}"
else
    echo -e "${RED}⚠️  Resposta inesperada: $RESPONSE${NC}"
fi

# ==================== CONFIGURAR GPU (OPCIONAL) ====================

echo ""
echo -e "${YELLOW}🎮 Verificando suporte a GPU...${NC}"

if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU detectada${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    
    echo ""
    echo -e "${GREEN}Ollama usará GPU automaticamente${NC}"
    echo -e "${YELLOW}Para configurar camadas GPU:${NC}"
    echo "  export OLLAMA_GPU_LAYERS=35"
else
    echo -e "${YELLOW}⚠️  GPU NVIDIA não detectada${NC}"
    echo -e "${YELLOW}Ollama usará CPU (mais lento)${NC}"
fi

# ==================== RESUMO ====================

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║                                                      ║"
echo "║     ✅ Setup Concluído com Sucesso!                 ║"
echo "║                                                      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

echo -e "${GREEN}📊 Resumo:${NC}"
echo "  • Ollama instalado e rodando"
echo "  • Modelos LLM baixados"
echo "  • Porta: http://localhost:11434"
echo ""

echo -e "${YELLOW}💡 Próximos passos:${NC}"
echo "  1. Configure .env com OLLAMA_HOST=http://localhost:11434"
echo "  2. Execute: python main.py"
echo "  3. Teste a API: curl http://localhost:8000/health"
echo ""

echo -e "${GREEN}🚀 CeciAI está pronto para uso!${NC}"
