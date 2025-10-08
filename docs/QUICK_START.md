# 🚀 Como Fazer o CeciAI Funcionar

**Objetivo:** Instalar o que precisa e fazer a aplicação rodar  
**Tempo:** 10 minutos

---

## 📦 O Que Você Precisa

### Obrigatório:
- ✅ **Python 3.11+** - [Baixar aqui](https://www.python.org/downloads/)
- ✅ **Ollama** - LLM gratuito - [Baixar aqui](https://ollama.com/download)
- ✅ **8 GB RAM** 
- ✅ **5 GB espaço disco**

### Opcional:
- 🐳 Docker (se quiser usar containers)

---

## 🚀 Como Fazer Rodar (Passo a Passo)

### 1️⃣ Baixar o Código

```bash
git clone https://github.com/lukeware-digital/ai-invest.git
cd ai-invest
```

### 2️⃣ Instalar Python (se não tiver)

```bash
# Verificar se tem Python
python3 --version

# Deve mostrar: Python 3.11.x ou superior
```

**Não tem Python?** Baixe em: https://www.python.org/downloads/

### 3️⃣ Criar Ambiente Virtual

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

✅ **Sucesso:** Deve aparecer `(.venv)` no início da linha do terminal

### 4️⃣ Instalar Bibliotecas Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

⏱️ **Aguarde:** ~3-5 minutos baixando bibliotecas

### 5️⃣ Instalar Ollama (IA Local)

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Mac:**
```bash
brew install ollama
# Ou baixar de: https://ollama.com/download
```

**Windows:**
- Baixe: https://ollama.com/download
- Execute o instalador
- Pronto! (inicia automaticamente)

### 6️⃣ Baixar Modelo de IA

```bash
# Baixar modelo (2GB)
ollama pull llama3.2:3b

# Verificar se baixou
ollama list
```

✅ **Deve mostrar:** `llama3.2:3b` na lista

### 7️⃣ Iniciar Ollama

```bash
# Linux/Mac (em segundo plano)
ollama serve &

# Windows - já está rodando automaticamente

# Verificar se está rodando
curl http://localhost:11434
```

### 8️⃣ Criar Dados de Exemplo

```bash
# Criar diretórios necessários
mkdir -p data/historical

# Criar dados de teste
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
print('✅ Dados criados!')
"
```

---

## ▶️ Como Rodar a Aplicação

### Opção A: Apenas API (Mais Simples)

```bash
# Ativar ambiente virtual (se não estiver ativado)
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Rodar API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

✅ **Funcionou?** Acesse: http://localhost:8000/docs

### Opção B: API + Dashboard Visual

**Terminal 1 - API:**
```bash
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Dashboard:**
```bash
source .venv/bin/activate
streamlit run dashboard.py --server.port 8050
```

✅ **Acessos:**
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8050

---

## 🧪 Como Testar se Funcionou

### Teste 1: API está respondendo?

```bash
curl http://localhost:8000/health
```

✅ **Deve retornar:** `{"status":"healthy",...}`

### Teste 2: Fazer uma análise

```bash
curl -X POST "http://localhost:8000/api/v1/analyze-candles" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC/USD",
    "candles": [
      {"timestamp": "2025-10-08T10:00:00Z", "open": 50000, "high": 50200, "low": 49900, "close": 50100, "volume": 1000000},
      {"timestamp": "2025-10-08T11:00:00Z", "open": 50100, "high": 50300, "low": 50000, "close": 50200, "volume": 1100000},
      {"timestamp": "2025-10-08T12:00:00Z", "open": 50200, "high": 50400, "low": 50100, "close": 50300, "volume": 1200000}
    ],
    "capital_available": 10000
  }'
```

✅ **Deve retornar:** Análise com decisão BUY/SELL/HOLD

### Teste 3: Dashboard funcionando?

Abra no navegador: http://localhost:8050

✅ **Deve mostrar:** Dashboard com gráficos e métricas

---

## 📊 Comandos Úteis

```bash
# Sistema
make build          # Build completo
make up             # Iniciar containers
make down           # Parar containers
make logs           # Ver logs
make clean          # Limpar tudo

# Testes
python scripts/test_complete_system.py    # Teste completo
python scripts/test_api_live.py          # Teste API
python scripts/train_ml_models.py        # Treinar ML

# API
uvicorn api.main:app --reload            # Desenvolvimento
uvicorn api.main:app --host 0.0.0.0      # Produção

# Dados
python utils/download_historical_data.py  # Download dados
python utils/data_updater.py             # Atualizar dados
```

---

## 🎯 Próximos Passos

1. ✅ **Sistema funcionando** - Parabéns!
2. 📖 **Entender arquitetura** - Leia [ARCHITECTURE.md](ARCHITECTURE.md)
3. 🧪 **Testar API** - Veja [API_USAGE.md](API_USAGE.md)
4. 🚀 **Deploy produção** - Siga [DEPLOYMENT.md](DEPLOYMENT.md)
5. 💰 **Paper trading** - Comece com capital virtual
6. 📊 **Backtesting** - Teste estratégias com dados históricos

---

**🎉 Sistema pronto! Agora você pode comprar e vender no momento certo!** 🚀📈

**Tempo total:** ~10 minutos ⏱️