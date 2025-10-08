# 🚀 CeciAI - Guia de Início Rápido

**Tempo estimado:** 5-10 minutos  
**Pré-requisitos:** Python 3.11+, Docker (opcional)

---

## ⚡ Instalação Rápida

### Opção 1: Docker (Recomendado)

```bash
# 1. Clonar repositório
git clone <repo-url>
cd ceci-ai

# 2. Build completo (instala tudo automaticamente)
make build

# 3. Iniciar sistema
make up

# 4. Testar
curl http://localhost:8000/health
```

### Opção 2: Instalação Manual

```bash
# 1. Ambiente Python
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. Dependências
pip install -r requirements.txt

# 3. Ollama (LLM local)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b

# 4. Configurar ambiente
cp .env.example .env
# Editar .env se necessário

# 5. Testar sistema
python scripts/test_complete_system.py
```

---

## 🧪 Primeiro Teste

### Teste Completo do Sistema

```bash
# Executar todos os testes
python scripts/test_complete_system.py

# Saída esperada:
# ✅ PASSOU: Configurações
# ✅ PASSOU: Ollama LLM
# ✅ PASSOU: Modelos ML
# ✅ PASSOU: Agentes (9/9)
# ✅ PASSOU: Pipeline Completo
# ✅ PASSOU: API REST
# ✅ PASSOU: Capital Management
# ✅ PASSOU: Backtesting
# 🎉 SISTEMA 100% FUNCIONAL!
```

### Teste da API

```bash
# Iniciar API (se não estiver rodando)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Testar análise de candles
python scripts/test_api_live.py

# Ou usar curl
curl -X POST "http://localhost:8000/api/v1/analyze-candles" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC/USD",
    "candles": [
      {"timestamp": "2025-10-08T10:00:00Z", "open": 50000, "high": 50200, "low": 49900, "close": 50100, "volume": 1000000},
      {"timestamp": "2025-10-08T11:00:00Z", "open": 50100, "high": 50300, "low": 50000, "close": 50200, "volume": 1100000}
    ],
    "capital_available": 10000,
    "strategy": "scalping"
  }'
```

---

## 📊 Uso Básico

### Análise Simples

```python
import asyncio
import pandas as pd
from agents.pipeline import AgentPipeline

async def analyze_market():
    # Dados de exemplo (substitua por dados reais)
    data = {
        'timestamp': ['2025-10-08T10:00:00Z', '2025-10-08T11:00:00Z'],
        'open': [50000, 50100],
        'high': [50200, 50300], 
        'low': [49900, 50000],
        'close': [50100, 50200],
        'volume': [1000000, 1100000]
    }
    df = pd.DataFrame(data)
    
    # Executar análise completa
    pipeline = AgentPipeline()
    result = await pipeline.execute(
        df=df,
        symbol='BTC/USD',
        timeframe='1h',
        capital_available=10000.0
    )
    
    # Resultado
    decision = result['final_decision']
    print(f"🎯 Decisão: {decision['decision']}")
    print(f"📊 Confiança: {decision['confidence']:.0%}")
    print(f"💰 Valor sugerido: ${decision.get('quantity_usd', 0):,.2f}")
    
    if decision['decision'] == 'BUY':
        print(f"📈 Entry: ${decision.get('entry_price', 0):,.2f}")
        print(f"🛑 Stop Loss: ${decision.get('stop_loss', 0):,.2f}")
        print(f"🎯 Take Profit: ${decision.get('take_profit_1', 0):,.2f}")

# Executar
asyncio.run(analyze_market())
```

### Backtesting Rápido

```python
import asyncio
from backtesting.backtest_engine import BacktestEngine
from strategies.scalping import ScalpingStrategy

async def quick_backtest():
    # Carregar dados históricos (exemplo)
    df = pd.read_csv('data/historical/BTC_USD_1h.csv')  # Se existir
    
    # Configurar backtest
    backtest = BacktestEngine(initial_capital=10000)
    strategy = ScalpingStrategy()
    
    # Executar
    results = await backtest.run(df, strategy, None)
    
    # Métricas principais
    print(f"💰 Retorno Total: {results['total_return']:.2%}")
    print(f"📊 Win Rate: {results['metrics']['win_rate']:.2%}")
    print(f"📈 Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
    print(f"📉 Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
    print(f"🔢 Total Trades: {results['metrics']['total_trades']}")

asyncio.run(quick_backtest())
```

---

## 🔧 Configuração

### Variáveis de Ambiente (.env)

```bash
# Ollama (LLM)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL_PRIMARY=llama3.2:3b

# Trading
INITIAL_CAPITAL=10000
MAX_DAILY_LOSS=0.03
MAX_POSITION_SIZE=0.20
ENABLE_TRADING=false
USE_PAPER_TRADING=true

# CoinAPI (opcional - para dados reais)
COINAPI_KEY=your-key-here
COINAPI_MODE=development

# Ambiente
CECIAI_ENV=development
LOG_LEVEL=INFO
```

### Personalizar Capital

```python
from config.capital_management import CapitalManager

# Configurar capital inicial
capital_mgr = CapitalManager(
    initial_capital=50000,      # $50k inicial
    max_daily_loss=0.02,        # Máximo 2% perda/dia
    max_position_size=0.15,     # Máximo 15% por posição
    max_concurrent_positions=3   # Máximo 3 posições simultâneas
)
```

---

## 🐛 Troubleshooting

### Problema: Ollama não responde

```bash
# Verificar se está rodando
ollama list

# Se não estiver, iniciar
ollama serve

# Baixar modelo se necessário
ollama pull llama3.2:3b
```

### Problema: Erro de GPU

```bash
# Verificar CUDA
nvidia-smi

# Se não funcionar, usar CPU
export CUDA_VISIBLE_DEVICES=""
```

### Problema: API não inicia

```bash
# Verificar porta
lsof -i :8000

# Usar porta diferente
uvicorn api.main:app --port 8001
```

### Problema: Modelos ML não encontrados

```bash
# Treinar modelos
python scripts/train_ml_models.py

# Verificar se foram criados
ls -la data/models/
```

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

## ✅ Checklist de Validação

Após instalação, verificar:

- [ ] `ollama list` mostra llama3.2:3b
- [ ] `python scripts/test_complete_system.py` passa todos os testes
- [ ] `curl http://localhost:8000/health` retorna 200 OK
- [ ] `ls data/models/` mostra modelos ML treinados
- [ ] Análise de exemplo executa sem erros

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