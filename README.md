# 🤖 CeciAI - Sistema Inteligente de Trading com IA

**Sistema de análise e trading de criptomoedas usando 9 Agentes LLM especializados**

[![Status](https://img.shields.io/badge/status-funcional-success)](https://github.com/lukeware-digital/ai-invest)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 📋 Visão Geral

CeciAI é um sistema completo de trading que combina:

- 🧠 **9 Agentes LLM** especializados (via Ollama - gratuito e local)
- 📊 **Análise Técnica** completa (60+ padrões de candlestick + 15 indicadores)
- 🤖 **Machine Learning** (LSTM, CNN, XGBoost)
- 💰 **Gestão de Capital** com circuit breaker
- 📈 **Backtesting e Paper Trading**
- ⚡ **Arquitetura assíncrona** para máxima performance
- 💾 **DuckDB + Redis** local (zero custo cloud)

### 🎯 Objetivo

Identificar o **momento exato** de compra e venda de criptomoedas através de análise inteligente multi-agente.

---

## 🚀 Quick Start

### Pré-requisitos

- ✅ **Python 3.11+**
- ✅ **Git**
- ✅ **Ollama** (LLM local)
- ✅ **8GB RAM**
- ✅ **5GB espaço disco**

### Instalação Rápida (5 passos)

```bash
# 1. Clonar repositório
git clone https://github.com/lukeware-digital/ai-invest.git
cd ai-invest

# 2. Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt

# 4. Instalar Ollama e modelo
curl -fsSL https://ollama.com/install.sh | sh  # Linux/Mac
# Windows: baixar de https://ollama.com/download
ollama pull llama3.2:3b

# 5. Iniciar aplicação
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

✅ **Pronto!** Acesse: http://localhost:8000/docs

---

## 📖 Documentação

| Documento | Descrição |
|-----------|-----------|
| **[INSTALL.md](docs/INSTALL.md)** | Pré-requisitos e instalação completa |
| **[QUICK_START.md](docs/QUICK_START.md)** | Como fazer rodar em 10 minutos |
| **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Como o sistema funciona |
| **[API_USAGE.md](docs/API_USAGE.md)** | Endpoints e exemplos de uso |
| **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** | Deploy em produção |

### Navegação Rápida

- 🚀 **Quero usar agora:** [QUICK_START.md](docs/QUICK_START.md)
- 🧠 **Quero entender:** [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- 📡 **Quero integrar:** [API_USAGE.md](docs/API_USAGE.md)
- 🌐 **Quero fazer deploy:** [DEPLOYMENT.md](docs/DEPLOYMENT.md)

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────┐
│                    CeciAI Pipeline                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  📊 Dados           🧪 Análise         🤖 Decisão      │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐     │
│  │ CoinAPI  │─────▶│Technical │─────▶│9 Agentes │     │
│  │ DuckDB   │      │Analysis  │      │   LLM    │     │
│  │ Redis    │      │ML Models │      │Pipeline  │     │
│  └──────────┘      └──────────┘      └──────────┘     │
│                                             │           │
│                                             ▼           │
│                                      ┌──────────┐      │
│                                      │ Capital  │      │
│                                      │Management│      │
│                                      └──────────┘      │
│                                             │           │
│                                             ▼           │
│                                      ┌──────────┐      │
│                                      │ Strategy │      │
│                                      │Execution │      │
│                                      └──────────┘      │
└─────────────────────────────────────────────────────────┘
```

---

## 🤖 Os 9 Agentes LLM

### Pipeline de Decisão

```
FASE 1: Contexto (Sequencial)
├── Agent 1: Market Expert       → Contexto de mercado
└── Agent 2: Data Analyzer       → Qualidade dos dados

FASE 2: Análise Técnica (Paralelo)
├── Agent 3: Technical Analyst   → Indicadores técnicos
└── Agent 4: Candlestick Expert  → Padrões de candles

FASE 3: Avaliação (Sequencial)
└── Agent 5: Investment Evaluator → Score 0-100

FASE 4: Estratégia (Paralelo)
├── Agent 6: Time Horizon Advisor → Timeframe ideal
└── Agent 7: Trade Classifier     → Tipo de operação

FASE 5: Execução (Condicional)
├── Agent 8: Day-Trade Executor   → SE day trade
└── Agent 9: Long-Term Executor   → SE position trade
```

### Características

- ✅ **Execução Paralela** onde possível
- ✅ **Roteamento Inteligente** (Agent 7 decide 8 ou 9)
- ✅ **Tratamento de Erros** com fallbacks
- ✅ **Logging Detalhado** de cada etapa

---

## 💻 Uso Básico

### 1. Iniciar Sistema

```bash
# Ativar ambiente virtual
source .venv/bin/activate

# Iniciar API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# (Opcional) Iniciar Dashboard
streamlit run dashboard.py --server.port 8050
```

### 2. Testar API

```bash
curl -X POST "http://localhost:8000/api/v1/analyze-candles" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC/USD",
    "candles": [
      {"timestamp": "2025-10-08T10:00:00Z", "open": 50000, "high": 50200, "low": 49900, "close": 50100, "volume": 1000000},
      {"timestamp": "2025-10-08T11:00:00Z", "open": 50100, "high": 50300, "low": 50000, "close": 50200, "volume": 1100000}
    ],
    "capital_available": 10000
  }'
```

### 3. Uso via Python

```python
import asyncio
import pandas as pd
from agents.pipeline import AgentPipeline

async def analyze():
    # Dados de exemplo
    df = pd.DataFrame({
        'timestamp': ['2025-10-08T10:00:00Z', '2025-10-08T11:00:00Z'],
        'open': [50000, 50100],
        'high': [50200, 50300],
        'low': [49900, 50000],
        'close': [50100, 50200],
        'volume': [1000000, 1100000]
    })
    
    # Executar análise
    pipeline = AgentPipeline()
    result = await pipeline.execute(
        df=df,
        symbol='BTC/USD',
        timeframe='1h',
        capital_available=10000.0
    )
    
    # Ver decisão
    decision = result['final_decision']
    print(f"🎯 Decisão: {decision['decision']}")
    print(f"📊 Confiança: {decision['confidence']:.0%}")
    print(f"💰 Valor: ${decision.get('quantity_usd', 0):,.2f}")

asyncio.run(analyze())
```

---

## 📊 Funcionalidades Implementadas

### ✅ Core
- [x] 9 Agentes LLM especializados
- [x] Pipeline Orchestrator
- [x] API FastAPI assíncrona
- [x] DuckDB + Redis local

### ✅ Análise
- [x] 60+ padrões de candlestick
- [x] 15+ indicadores técnicos (RSI, MACD, Bollinger, EMA, SMA, ADX, ATR, etc)
- [x] LSTM para previsão de preços
- [x] CNN para reconhecimento de padrões
- [x] XGBoost para classificação

### ✅ Trading
- [x] Estratégia Scalping
- [x] Estratégia Swing Trading
- [x] Gestão de Capital com circuit breaker
- [x] Backtesting Engine
- [x] Paper Trading real-time

### ✅ Dados
- [x] CoinAPI client assíncrono
- [x] Cache multi-camada (Redis + DuckDB)
- [x] Backup automático
- [x] Data Updater agendado

---

## 🔧 Estrutura do Projeto

```
ceci-ai/
├── api/                      # API FastAPI
│   └── main.py               # Endpoints
├── agents/                   # 9 Agentes LLM
│   ├── agent_*.py           # Agentes individuais
│   ├── pipeline.py          # Orchestrator
│   └── ml_models/           # ML (LSTM, CNN, XGBoost)
├── strategies/              # Estratégias de trading
│   ├── scalping.py
│   └── swing_trading.py
├── config/                  # Configurações
│   └── capital_management.py
├── backtesting/            # Backtesting
│   ├── backtest_engine.py
│   └── paper_trading.py
├── utils/                   # Utilitários
│   ├── coinapi_client.py
│   ├── database.py
│   ├── technical_indicators.py
│   └── candlestick_patterns.py
├── data/                    # Dados locais
│   ├── ceciai.duckdb
│   └── historical/
├── docs/                    # Documentação
│   ├── INSTALL.md
│   ├── QUICK_START.md
│   ├── ARCHITECTURE.md
│   ├── API_USAGE.md
│   └── DEPLOYMENT.md
└── dashboard.py            # Dashboard Streamlit
```

---

## 🛡️ Gestão de Risco

O sistema inclui gestão de risco integrada:

```python
RISK_PARAMETERS = {
    "max_daily_loss": 0.03,          # 3% do capital
    "max_position_size": 0.20,       # 20% por posição
    "stop_loss": "MANDATORY",        # Sempre obrigatório
    "take_profit_ratio": 1.5,        # Mínimo 1.5:1
    "max_concurrent_positions": 5,
    "circuit_breaker": {
        "consecutive_losses": 3,     # Pausa após 3 perdas
        "pause_duration": "1h"
    }
}
```

---

## 💰 Custos

### Modo Development (Padrão)

- ✅ Ollama: **Gratuito** (LLM local)
- ✅ DuckDB: **Gratuito** (banco local)
- ✅ Redis: **Gratuito** (cache local)
- ✅ CoinAPI: **Gratuito** (free tier: 100 requests/dia)

**Total: $0/mês** 💰

### Modo Production (Opcional)

- CoinAPI Pro: $79/mês
- VPS (8GB RAM): $20-40/mês

**Total: ~$100/mês**

---

## 🖥️ Hardware Recomendado

### Mínimo
- CPU: 4 cores
- RAM: 8GB
- Disco: 20GB SSD

### Recomendado
- CPU: 8+ cores
- RAM: 16-32GB
- Disco: 50GB NVMe SSD
- GPU: NVIDIA (opcional, para acelerar)

---

## 📈 Comandos Úteis

```bash
# Sistema
uvicorn api.main:app --reload              # Iniciar API
streamlit run dashboard.py --server.port 8050  # Dashboard

# Testes
python scripts/test_complete_system.py     # Teste completo
python scripts/test_api_live.py            # Teste API

# Dados
python utils/download_historical_data.py   # Baixar dados
python utils/data_updater.py               # Atualizar dados
```

---

## 📊 Status do Projeto

```
✅ Sistema Core               100% ████████████████████
✅ 9 Agentes LLM              100% ████████████████████
✅ Análise Técnica            100% ████████████████████
✅ Machine Learning           100% ████████████████████
✅ Gestão de Capital          100% ████████████████████
✅ Backtesting                100% ████████████████████
✅ Paper Trading              100% ████████████████████
✅ Documentação               100% ████████████████████

TOTAL: 100% ██████████████████████████████████████
```

**🎉 Sistema 100% funcional e pronto para uso!**

---

## ⚠️ Disclaimer

**IMPORTANTE:** Este sistema é para fins educacionais e de pesquisa. Trading de criptomoedas envolve risco significativo. Nunca invista mais do que pode perder. Sempre teste em paper trading antes de usar capital real.

---

## 📝 Licença

MIT License - Veja arquivo [LICENSE](LICENSE) para detalhes

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

---

## 📧 Suporte

- 📖 **Documentação:** [docs/](docs/)
- 🐛 **Issues:** [GitHub Issues](https://github.com/lukeware-digital/ai-invest/issues)
- 💬 **Discussões:** [GitHub Discussions](https://github.com/lukeware-digital/ai-invest/discussions)

---

**Desenvolvido com ❤️ usando Ollama (gratuito) + Python + FastAPI**

**🚀 Sistema 100% local, zero custo cloud, alta performance!**

**Versão:** 1.0.0 | **Última atualização:** 08/10/2025
