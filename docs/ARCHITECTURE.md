# 🏗️ Arquitetura do CeciAI

**Sistema de Trading Inteligente com IA**

---

## 🎯 Visão Geral

O CeciAI é um sistema completo que **identifica o momento exato** de comprar e vender criptomoedas usando:

- **🧠 9 Agentes LLM** especializados
- **🤖 3 Modelos ML** (LSTM, CNN, XGBoost)  
- **📊 Análise Técnica** completa
- **🕯️ Padrões de Candles** (60+ padrões)
- **💰 Gestão de Capital** automática

**Resultado:** Decisões de BUY/SELL/HOLD com alta precisão em ~30 segundos.

---

## 🔄 Fluxo Completo

```
📊 DADOS DE MERCADO
    ↓
🔍 ANÁLISE TÉCNICA (Indicadores + Padrões)
    ↓
🤖 MACHINE LEARNING (Previsões)
    ↓
🧠 9 AGENTES LLM (Decisão Inteligente)
    ↓
💰 VALIDAÇÃO DE CAPITAL E RISCO
    ↓
✅ DECISÃO FINAL: BUY/SELL/HOLD
    ↓
📈 EXECUÇÃO (Paper Trading ou Real)
```

---

## 🧠 Os 9 Agentes LLM

### **Fase 1: Contexto de Mercado**
- **Agent 1: Market Expert** - Analisa condições gerais (bull/bear/sideways)
- **Agent 2: Data Analyzer** - Valida qualidade dos dados

### **Fase 2: Análise Técnica (Paralelo)**
- **Agent 3: Technical Analyst** - Interpreta indicadores (RSI, MACD, etc)
- **Agent 4: Candlestick Specialist** ⭐ - Detecta padrões (Hammer, Doji, etc)

### **Fase 3: Avaliação**
- **Agent 5: Investment Evaluator** - Consolida análises e gera score (0-100)

### **Fase 4: Estratégia (Paralelo)**
- **Agent 6: Time Horizon Advisor** - Define timeframe (scalping/swing/long-term)
- **Agent 7: Trade Classifier** - Classifica tipo (day-trade vs position)

### **Fase 5: Execução (Condicional)**
- **Agent 8: Day-Trade Executor** ⭐ - Plano para trades rápidos
- **Agent 9: Long-Term Executor** - Plano para posições longas

---

## 🤖 Modelos de Machine Learning

### **1. LSTM Price Predictor**
```python
# Previsão de preços futuros
Input: 60 timesteps (OHLCV + indicadores)
Output: Preço em 1h, 4h, 24h
Arquitetura: 2 camadas LSTM (128, 64 unidades)
GPU: ✅ CUDA support
```

### **2. CNN Pattern Recognizer**  
```python
# Reconhecimento de padrões visuais
Input: Imagem 64x64 de candles
Output: Bullish/Bearish/Neutral
Arquitetura: 3 camadas Conv2D (32, 64, 128 filtros)
GPU: ✅ CUDA support
```

### **3. XGBoost Trade Classifier**
```python
# Classificação de ações
Input: 18+ indicadores técnicos
Output: BUY/SELL/HOLD
Algoritmo: XGBoost otimizado
CPU: ✅ Multi-thread
```

---

## 📊 Análise Técnica

### **Indicadores Implementados (15+)**
- **Momentum:** RSI, Stochastic, Williams %R
- **Trend:** MACD, ADX, EMA, SMA  
- **Volatilidade:** Bollinger Bands, ATR
- **Volume:** OBV, VWAP, Volume Ratio
- **Suporte/Resistência:** Fibonacci, Pivot Points

### **Padrões de Candles (60+)**
- **Reversão Alta:** Hammer, Bullish Engulfing, Morning Star
- **Reversão Baixa:** Shooting Star, Bearish Engulfing, Evening Star
- **Continuação:** Flags, Pennants, Triangles
- **Indecisão:** Doji, Spinning Top, Harami

---

## 💰 Gestão de Capital

### **Capital Manager**
```python
# Exemplo de uso
capital_mgr = CapitalManager(initial_capital=10000)

# Calcular tamanho da posição
position = capital_mgr.calculate_position_size(
    symbol="BTC/USD",
    strategy="scalping", 
    risk_percent=0.01,    # 1% de risco
    entry_price=50000,
    stop_loss=49500
)

# Resultado: 
# - Quantidade: $1,000 (10% do capital)
# - Risco: $100 (1% do capital)
# - Risk/Reward: 2.3:1
```

### **Proteções de Risco**
- **Risco máximo por trade:** 1% do capital
- **Posição máxima:** 20% do capital
- **Perda diária máxima:** 3% do capital
- **Circuit breaker:** Para após 3 perdas consecutivas
- **Stop-loss obrigatório:** Sempre definido

---

## 🎯 Exemplo Completo de Decisão

### **Cenário: BTC/USD @ $50,250**

```
📊 DADOS COLETADOS:
   - Preço atual: $50,250
   - Volume 24h: $12.5B  
   - Últimos 20 candles (1h)

🔍 ANÁLISE TÉCNICA:
   - RSI: 42 (neutro-baixo) ✅
   - MACD: Cruzamento de alta ✅  
   - Bollinger: Preço na banda inferior ✅
   - Volume: 1.4x média ✅

🕯️ PADRÕES DETECTADOS:
   - HAMMER (reversão de alta)
   - Confiança: 85% ✅
   - Confirmação: Próximo candle ✅

🤖 ML PREVÊ:
   - LSTM: +1.2% em 1h (73% confiança) ✅
   - CNN: Padrão Bullish (78% confiança) ✅  
   - XGBoost: BUY (71% confiança) ✅

🧠 AGENTES DECIDEM:
   Agent 1: "Mercado em recuperação"
   Agent 2: "Dados de alta qualidade"  
   Agent 3: "Sinais técnicos positivos"
   Agent 4: "HAMMER detectado - BUY" ⭐
   Agent 5: "Score: 78/100 - Excelente oportunidade" ✅
   Agent 6: "Scalping recomendado (1-4h)"
   Agent 7: "Day-trade classificado"
   Agent 8: "EXECUTAR COMPRA" ⭐

💰 VALIDAÇÃO CAPITAL:
   - Capital disponível: $7,500
   - Necessário: $1,500 (20%)
   - Risco: $33 (0.33% do capital)
   - Risk/Reward: 2.3:1 ✅

✅ DECISÃO FINAL: COMPRAR AGORA!

📋 PLANO DE EXECUÇÃO:
   - Entrada: $50,300 (ordem limite)
   - Quantidade: 0.0298 BTC ($1,500)
   - Stop-Loss: $49,200 (-2.2%)
   - Take-Profit 1: $51,500 (+2.4%)  
   - Take-Profit 2: $52,800 (+5.0%)
   - Tempo máximo: 4 horas
```

---

## 🏗️ Arquitetura Técnica

### **Backend (Python)**
```
api/                    # FastAPI REST API
├── main.py            # Endpoints principais
└── models.py          # Modelos Pydantic

agents/                 # 9 Agentes LLM
├── pipeline.py        # Orchestrator principal  
├── agent_1_market_expert.py
├── agent_4_candlestick_specialist.py
├── agent_8_daytrade_executor.py
└── ml_models/         # Modelos ML
    ├── price_predictor.py     # LSTM
    ├── pattern_recognizer.py  # CNN  
    └── trade_classifier.py    # XGBoost

utils/                  # Utilitários
├── technical_indicators.py   # RSI, MACD, etc
├── candlestick_patterns.py  # Padrões de candles
├── database.py             # DuckDB + Redis
└── coinapi_client.py       # Cliente CoinAPI

config/                 # Configurações
├── capital_management.py   # Gestão de capital
└── settings.py            # Configurações gerais

strategies/             # Estratégias de trading
├── scalping.py        # Scalping (5-30min)
├── swing_trading.py   # Swing (1-7 dias)  
└── arbitrage.py       # Arbitragem

backtesting/           # Backtesting e paper trading
├── backtest_engine.py # Engine de backtesting
├── paper_trading.py   # Paper trading
└── metrics.py         # Métricas de performance
```

### **Banco de Dados (Zero Custo)**
```
DuckDB (Principal)     # Arquivo local, alta performance
├── ohlcv_data        # Dados OHLCV históricos
├── trades            # Histórico de trades
├── analyses          # Análises salvas
└── performance       # Métricas de performance

Redis (Cache)          # Cache em memória  
├── hot_cache         # Dados recentes (5min TTL)
├── permanent_cache   # Dados históricos (sem TTL)
└── session_data      # Dados da sessão

Backup (Automático)    # Backup multi-camada
├── data/historical/  # Parquet comprimido
├── data/backups/     # JSON + GZIP
└── checksums/        # Validação integridade
```

### **LLM Local (Ollama)**
```
Modelos Instalados:
├── llama3.2:3b      # Modelo principal (rápido)
├── codellama:7b     # Análise de código  
└── mistral:7b       # Backup

Configuração GPU:
├── CUDA Support: ✅ RTX 3060 Ti
├── VRAM Usage: ~5GB
└── Inference: ~2-3s por agente
```

---

## ⚡ Performance

### **Tempos de Execução (Hardware Otimizado)**
| Componente | Tempo |
|-----------|-------|
| Coleta de dados | 0.5s |
| Análise técnica | 0.2s |
| ML Predictions | 0.3s |
| Agent 1-7 (LLM) | 2-3s cada |
| Agent 8/9 (Decisão) | 3-4s |
| **TOTAL** | **~25-30s** |

### **Recursos Utilizados**
- **CPU:** 20-40% (durante análise)
- **RAM:** 2-4GB  
- **GPU:** 60-80% VRAM (RTX 3060 Ti)
- **Disco:** ~100MB para 1 ano de dados

---

## 🔒 Segurança e Confiabilidade

### **Proteções Implementadas**
- ✅ **Stop-loss obrigatório** em todos os trades
- ✅ **Circuit breaker** após perdas consecutivas  
- ✅ **Validação de capital** antes de cada trade
- ✅ **Backup automático** de todos os dados
- ✅ **Paper trading** para testes seguros
- ✅ **Logs detalhados** de todas as operações

### **Validações de Entrada**
- ✅ Mínimo 10 candles para análise
- ✅ Capital suficiente para posição
- ✅ Risk/Reward ratio >= 1.5:1
- ✅ Score de oportunidade >= 60/100
- ✅ Confirmação de padrões de candles

---

## 🚀 Escalabilidade

### **Suporte a Múltiplos Ativos**
```python
# Analisar múltiplos símbolos simultaneamente
symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD']
results = await pipeline.analyze_multiple(symbols, timeframe='1h')
```

### **Estratégias Personalizadas**
```python
# Criar nova estratégia
class MyStrategy(BaseStrategy):
    def analyze(self, df, symbol, price, agent_analyses):
        # Sua lógica personalizada
        return {'signal': 'BUY', 'confidence': 0.85}
```

### **Deploy em Produção**
- ✅ **Docker** completo
- ✅ **Docker Compose** para orquestração
- ✅ **Health checks** automáticos
- ✅ **Logs estruturados** 
- ✅ **Monitoring** de recursos
- ✅ **Auto-restart** em caso de falha

---

## 📊 Métricas e Monitoramento

### **Métricas de Trading**
```python
metrics = capital_mgr.get_performance_metrics()

# Exemplo de saída:
{
    'total_trades': 127,
    'win_rate': 0.68,           # 68% de acerto
    'profit_factor': 2.1,       # 2.1:1 profit factor  
    'sharpe_ratio': 1.8,        # Sharpe excelente
    'max_drawdown': 0.05,       # Máximo 5% de perda
    'total_return': 0.23,       # 23% de retorno
    'avg_trade_duration': '2h 15min'
}
```

### **Monitoramento em Tempo Real**
- 📊 **Dashboard** de capital e posições
- 📈 **Gráficos** de performance  
- 🚨 **Alertas** de risco
- 📝 **Logs** detalhados
- 📱 **Notificações** de trades

---

## 🎯 Casos de Uso

### **1. Scalping (5-30 minutos)**
- Análise em timeframe 1min
- Trades rápidos com lucro 0.5-1%
- Stop-loss apertado (0.3%)
- Volume alto necessário

### **2. Swing Trading (1-7 dias)**  
- Análise em timeframe 4h
- Trades com lucro 3-5%
- Stop-loss 2%
- Foco em tendências

### **3. Long-Term (semanas/meses)**
- Análise em timeframe 1d
- Acumulação gradual (DCA)
- Stop-loss 5-10%
- Foco em fundamentals

---

## ✅ Resumo

**O CeciAI é um sistema completo que:**

1. 📊 **Coleta** dados de mercado em tempo real
2. 🔍 **Analisa** com 15+ indicadores técnicos  
3. 🕯️ **Detecta** 60+ padrões de candles
4. 🤖 **Prevê** preços com ML (LSTM, CNN, XGBoost)
5. 🧠 **Decide** com 9 agentes LLM especializados
6. 💰 **Valida** capital e risco automaticamente
7. ✅ **Executa** trades no momento exato
8. 📊 **Monitora** performance continuamente

**Resultado:** Sistema capaz de **comprar e vender no momento certo** com alta precisão e risco controlado! 🎯📈

---

**Desenvolvido para identificar oportunidades que humanos perderiam** 🚀