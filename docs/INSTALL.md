# 📦 Pré-Requisitos e Instalação - CeciAI

**Lista completa de tudo que precisa instalar para rodar o CeciAI.**

---

## 📋 Índice

1. [Pré-Requisitos](#-pré-requisitos)
2. [Instalação Local (Desenvolvimento)](#-instalação-local-desenvolvimento)
3. [Próximos Passos](#-próximos-passos)

---

## 🎯 Pré-Requisitos

### 1️⃣ Python 3.11+ (OBRIGATÓRIO)

**O que é:** Linguagem de programação que roda a aplicação

**Como verificar se já tem:**
```bash
python3 --version
# ou
python --version
```

**Como instalar:**

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

**Mac:**
```bash
# Opção 1: Homebrew
brew install python@3.11

# Opção 2: Download direto
# https://www.python.org/downloads/macos/
```

**Windows:**
1. Baixe em: https://www.python.org/downloads/windows/
2. Execute o instalador
3. ✅ IMPORTANTE: Marque "Add Python to PATH"

---

### 2️⃣ Git (OBRIGATÓRIO)

**O que é:** Ferramenta para baixar o código do GitHub

**Como verificar se já tem:**
```bash
git --version
```

**Como instalar:**

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install git
```

**Mac:**
```bash
brew install git
# Ou vem junto com Xcode Command Line Tools
xcode-select --install
```

**Windows:**
- Baixe em: https://git-scm.com/download/win
- Execute o instalador

---

### 3️⃣ Ollama (OBRIGATÓRIO)

**O que é:** Software de IA local (LLM) - é o "cérebro" dos 9 agentes inteligentes

**Como verificar se já tem:**
```bash
ollama --version
```

**Como instalar:**

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Mac:**
```bash
# Opção 1: Homebrew
brew install ollama

# Opção 2: Download direto
# https://ollama.com/download/mac
```

**Windows:**
1. Baixe em: https://ollama.com/download/windows
2. Execute o instalador OllamaSetup.exe
3. Ollama inicia automaticamente

**Após instalar, baixe o modelo de IA:**
```bash
ollama pull llama3.2:3b
```

---

### 4️⃣ pip (OBRIGATÓRIO)

**O que é:** Gerenciador de pacotes Python (instala bibliotecas)

**Como verificar se já tem:**
```bash
pip --version
# ou
pip3 --version
```

**Como instalar:**

**Linux:**
```bash
sudo apt install python3-pip
```

**Mac:**
```bash
# Vem junto com Python
# Se não tiver:
python3 -m ensurepip --upgrade
```

**Windows:**
- Vem junto com Python (se marcou "Add to PATH")

---

## 🔧 Dependências do Sistema (OBRIGATÓRIO no Linux)

**O que é:** Bibliotecas do sistema operacional que o Python precisa

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    python3-dev \
    libssl-dev \
    libffi-dev \
    curl \
    wget
```

**Mac:**
```bash
# Instalar Command Line Tools
xcode-select --install
```

**Windows:**
- Não precisa (tudo vem no instalador do Python)

---

## 💾 Banco de Dados (INCLUÍDO - não precisa instalar)

**O que usa:** 
- ✅ **DuckDB** - Banco de dados embutido (vem nas bibliotecas Python)
- ✅ **Redis** - Cache opcional (não obrigatório)

**Você NÃO precisa instalar:**
- ❌ PostgreSQL
- ❌ MySQL
- ❌ MongoDB
- ❌ Redis Server

**Por quê?** DuckDB é um arquivo local, como SQLite. Já vem incluído!

---

## 🐳 Docker & Docker Compose (OPCIONAL)

**O que é:** Plataforma para rodar aplicações em containers (não obrigatório)

**Quando precisa:** Só se quiser usar containers ao invés de rodar direto

**Como verificar se já tem:**
```bash
docker --version
docker-compose --version
```

**Como instalar:**

**Linux (Ubuntu/Debian):**
```bash
# Docker
sudo apt update
sudo apt install docker.io

# Docker Compose
sudo apt install docker-compose

# Adicionar seu usuário ao grupo docker
sudo usermod -aG docker $USER
# Fazer logout e login novamente
```

**Mac:**
```bash
# Instalar Docker Desktop
brew install --cask docker

# Ou baixar de: https://www.docker.com/products/docker-desktop/
```

**Windows:**
- Baixe Docker Desktop: https://www.docker.com/products/docker-desktop/
- Execute o instalador

**⚠️ ATENÇÃO:** Docker é OPCIONAL! Se não quiser instalar, use o método sem Docker.

---

## 📊 Streamlit (OPCIONAL - para Dashboard)

**O que é:** Framework para criar dashboard visual

**Quando precisa:** Só se quiser usar o dashboard visual

**Como instalar:**
```bash
# Vem no requirements.txt, mas se quiser instalar separado:
pip install streamlit plotly
```

---

## 🖥️ GPU/CUDA (OPCIONAL - para acelerar IA)

**O que é:** Suporte para placa de vídeo NVIDIA acelerar processamento

**Quando precisa:** Só se tiver placa NVIDIA e quiser usar GPU

**Como verificar se tem:**
```bash
nvidia-smi
```

**Como instalar CUDA:**

**Linux:**
```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Verificar
nvcc --version
```

**Windows/Mac:**
- Baixe CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

**⚠️ ATENÇÃO:** GPU é OPCIONAL! Sistema funciona perfeitamente com CPU.

---

## 🌐 Navegador Web (OBRIGATÓRIO)

**O que é:** Para acessar a API e Dashboard

**Recomendados:**
- ✅ Google Chrome
- ✅ Firefox
- ✅ Edge
- ✅ Safari

---

## 📝 Editor de Código (RECOMENDADO)

**O que é:** Para ver e editar código (opcional mas útil)

**Recomendados:**
- ✅ VS Code - https://code.visualstudio.com/
- ✅ PyCharm - https://www.jetbrains.com/pycharm/
- ✅ Sublime Text
- ✅ Vim/Nano (terminal)

---

## 📊 Resumo do Que Instalar

### ✅ OBRIGATÓRIO (não funciona sem isso):

| Software | Função | Como Instalar |
|----------|--------|---------------|
| **Python 3.11+** | Linguagem da aplicação | python.org/downloads |
| **Git** | Baixar código | git-scm.com/downloads |
| **Ollama** | IA local (LLM) | ollama.com/download |
| **pip** | Instalar bibliotecas | Vem com Python |
| **Navegador** | Acessar API/Dashboard | Chrome, Firefox, etc |

### ⭐ RECOMENDADO (ajuda mas não é obrigatório):

| Software | Função | Como Instalar |
|----------|--------|---------------|
| **Docker** | Containers (alternativa) | docker.com |
| **VS Code** | Editor de código | code.visualstudio.com |
| **Streamlit** | Dashboard visual | pip install streamlit |

### ❌ NÃO PRECISA INSTALAR:

- ❌ PostgreSQL / MySQL / MongoDB
- ❌ Redis Server
- ❌ Node.js
- ❌ Java
- ❌ Apache / Nginx
- ❌ CUDA (funciona sem GPU)

---

## ✅ Checklist de Instalação

Antes de continuar, verifique se instalou:

- [ ] Python 3.11+ (`python3 --version`)
- [ ] Git (`git --version`)
- [ ] Ollama (`ollama --version`)
- [ ] pip (`pip --version`)
- [ ] Modelo Ollama (`ollama list` mostra llama3.2:3b)
- [ ] Navegador web instalado
- [ ] (Opcional) Docker (`docker --version`)
- [ ] (Opcional) VS Code ou editor

---

## 🚀 Próximos Passos

Agora que instalou tudo, vá para:

👉 **[QUICK_START.md](docs/QUICK_START.md)** - Como fazer a aplicação rodar

---

## 🚀 Instalação Local (Desenvolvimento)

### Passo a Passo Completo

#### 1️⃣ Clonar Repositório

```bash
git clone https://github.com/lukeware-digital/ai-invest.git
cd ai-invest
```

#### 2️⃣ Criar Ambiente Virtual Python

```bash
# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (CMD)
python -m venv .venv
.venv\Scripts\activate.bat
```

**✅ Verificar:** Seu prompt deve mostrar `(.venv)` no início

#### 3️⃣ Instalar Dependências Python

```bash
# Atualizar pip
pip install --upgrade pip

# Instalar todas as dependências
pip install -r requirements.txt

# Instalar Streamlit para dashboard (opcional)
pip install streamlit plotly
```

**⏱️ Tempo:** ~3-5 minutos

#### 4️⃣ Instalar Ollama (LLM Local Gratuito)

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Mac:**
```bash
# Opção 1: Homebrew
brew install ollama

# Opção 2: Download direto
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
1. Baixe de: https://ollama.com/download
2. Execute o instalador
3. Ollama inicia automaticamente

#### 5️⃣ Baixar Modelo LLM

```bash
# Modelo principal (2GB)
ollama pull llama3.2:3b

# Verificar instalação
ollama list
```

**Saída esperada:**
```
NAME              ID           SIZE    MODIFIED
llama3.2:3b       abc123       2.0 GB  X minutes ago
```

#### 6️⃣ Iniciar Ollama (Se não iniciou automaticamente)

```bash
# Linux/Mac (em background)
nohup ollama serve > /dev/null 2>&1 &

# Ou em um terminal separado
ollama serve

# Windows - Ollama inicia automaticamente como serviço
```

#### 7️⃣ Criar Estrutura de Dados (Opcional)

```bash
# Criar diretórios se não existirem
mkdir -p data/historical data/models data/cache data/realtime logs
```

#### 8️⃣ Baixar Dados Históricos (Opcional mas Recomendado)

```bash
# Se você tem uma chave CoinAPI
export COINAPI_KEY="your-key-here"
python utils/download_historical_data.py

# Ou criar dados de exemplo para testes
python -c "
import pandas as pd
from datetime import datetime, timedelta

dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
df = pd.DataFrame({
    'timestamp': dates,
    'open': [50000 + i*10 for i in range(100)],
    'high': [51000 + i*10 for i in range(100)],
    'low': [49000 + i*10 for i in range(100)],
    'close': [50500 + i*10 for i in range(100)],
    'volume': [1000000 + i*1000 for i in range(100)]
})
df.to_parquet('data/historical/BTC_USD_1h.parquet')
print('✅ Dados de exemplo criados!')
"
```

---

## ✅ Verificar Instalação

Execute o script de teste completo:

```bash
python scripts/test_complete_system.py
```

**Saída esperada:**
```
🧪 CeciAI - Teste Completo do Sistema

✅ PASSOU: Configurações do Sistema
✅ PASSOU: Ollama LLM está respondendo
✅ PASSOU: Modelos ML (fallback ativo)
✅ PASSOU: Agente 1: Market Expert
✅ PASSOU: Agente 2: Data Analyzer
✅ PASSOU: Agente 3: Technical Analyst
✅ PASSOU: Agente 4: Candlestick Specialist
✅ PASSOU: Agente 5: Investment Evaluator
✅ PASSOU: Agente 6: Time Horizon Advisor
✅ PASSOU: Agente 7: Trade Classifier
✅ PASSOU: Agente 8: Day-Trade Executor
✅ PASSOU: Agente 9: Long-Term Executor
✅ PASSOU: Pipeline Completo (9 agentes)
✅ PASSOU: API REST
✅ PASSOU: Capital Management
✅ PASSOU: Backtesting Engine

🎉 SISTEMA 100% FUNCIONAL!
```

---

## 🚀 Iniciar o Sistema

### Opção 1: API Apenas

```bash
# Terminal 1: API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Testar
curl http://localhost:8000/health
```

**Acesso:** http://localhost:8000/docs (Documentação interativa)

### Opção 2: API + Dashboard

```bash
# Terminal 1: API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Dashboard
streamlit run dashboard.py --server.port 8050
```

**Acessos:**
- API: http://localhost:8000
- Dashboard: http://localhost:8050

---

## ❓ Problemas Comuns

### ❌ Ollama não responde

```bash
# Verificar se está rodando
ollama list

# Iniciar se necessário
ollama serve
```

### ❌ ModuleNotFoundError

```bash
# Ativar ambiente virtual
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Reinstalar dependências
pip install -r requirements.txt
```

### ❌ Porta 8000 em uso

```bash
# Usar porta diferente
uvicorn api.main:app --port 8001
```

---

## 📚 Próximos Passos

Após instalação local:

1. ✅ **Desenvolvimento** - Sistema rodando localmente
2. 📖 **Como usar** - Veja [QUICK_START.md](QUICK_START.md)
3. 🚀 **Produção** - Deploy completo em [DEPLOYMENT.md](DEPLOYMENT.md)

---

**🎉 Instalação completa!**

- **Desenvolvimento:** Sistema rodando em http://localhost:8000
- **Produção:** Consulte [DEPLOYMENT.md](DEPLOYMENT.md)

