# 📡 CeciAI - Guia da API

**API REST para análise inteligente de trading**

---

## 🎯 Visão Geral

A API CeciAI oferece análise completa de trading usando:
- **9 Agentes LLM** especializados
- **3 Modelos ML** (LSTM, CNN, XGBoost)
- **Análise técnica** completa
- **Padrões de candles** (60+ padrões)

**Resultado:** Sinal de BUY/SELL/HOLD com plano de execução completo.

---

## 🚀 Início Rápido

### 1. Iniciar API

```bash
# Opção 1: Docker
make up

# Opção 2: Python direto
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Verificar se está funcionando
curl http://localhost:8000/health
```

### 2. Primeiro Teste

```bash
# Executar script de teste
python scripts/test_api_live.py

# Ou testar manualmente
curl -X POST "http://localhost:8000/api/v1/analyze-candles" \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json
```

---

## 📊 Endpoints Principais

### **POST /api/v1/analyze-candles**
Análise completa de histórico de candles com decisão de trading.

### **POST /api/v1/analyze** 
Análise avançada com configurações personalizadas.

### **GET /health**
Health check de todos os serviços.

### **GET /**
Informações da API e status.

---

## 🕯️ Endpoint: Análise de Candles

### **POST /api/v1/analyze-candles**

**Descrição:** Analisa histórico de candles e retorna sinal de BUY/SELL/HOLD com plano completo.

### Request

```json
{
  "symbol": "BTC/USD",
  "candles": [
    {
      "timestamp": "2025-10-08T10:00:00Z",
      "open": 50000.00,
      "high": 50500.00,
      "low": 49800.00,
      "close": 50300.00,
      "volume": 1000000.00
    },
    {
      "timestamp": "2025-10-08T11:00:00Z", 
      "open": 50300.00,
      "high": 50600.00,
      "low": 50100.00,
      "close": 50400.00,
      "volume": 1100000.00
    }
  ],
  "capital_available": 10000.00,
  "strategy": "scalping"
}
```

### Parâmetros

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `symbol` | string | Sim | Par de trading (ex: BTC/USD) |
| `candles` | array | Sim | Histórico de candles (mínimo 10) |
| `capital_available` | float | Não | Capital disponível em USD (padrão: 10000) |
| `strategy` | string | Não | Estratégia (scalping, swing, arbitrage) |

### Response Completa

```json
{
  "request_id": "req_20251008_123000_abc123",
  "timestamp": "2025-10-08T12:30:00Z",
  "processing_time": 28.45,
  "symbol": "BTC/USD",
  
  "signal": "BUY",
  "confidence": 0.78,
  "opportunity_score": 78,
  
  "entry_price": 50300.00,
  "stop_loss": 49200.00,
  "take_profit_1": 51500.00,
  "take_profit_2": 52800.00,
  
  "quantity_usd": 1500.00,
  "quantity_units": 0.0298,
  "risk_amount_usd": 33.00,
  "potential_profit_usd": 75.00,
  "risk_reward_ratio": 2.3,
  
  "reasoning": "HAMMER detectado com alta confiança (85%). Indicadores técnicos alinhados: RSI em 42, MACD com cruzamento de alta, preço na banda inferior de Bollinger. ML prevê alta de 2.8% em 4h com 73% de confiança. Agentes recomendam execução imediata.",
  
  "patterns_detected": [
    {
      "name": "Hammer",
      "type": "bullish_reversal", 
      "confidence": 0.85,
      "significance": "high"
    },
    {
      "name": "Bullish Engulfing",
      "type": "bullish_reversal",
      "confidence": 0.72,
      "significance": "medium"
    }
  ],
  
  "technical_analysis": {
    "rsi": 42.5,
    "macd": 120.5,
    "macd_signal": 115.3,
    "macd_histogram": 5.2,
    "bb_position": "lower_band",
    "bb_squeeze": false,
    "ema_9": 50150.0,
    "ema_21": 49980.0,
    "ema_50": 49750.0,
    "trend": "uptrend",
    "trend_strength": "moderate",
    "price_change_pct": 2.5,
    "volume_ratio": 1.42,
    "support_level": 49200.0,
    "resistance_level": 52000.0
  },
  
  "ml_predictions": {
    "lstm_price_1h": 50850.0,
    "lstm_price_4h": 51650.0,
    "lstm_confidence": 0.73,
    "cnn_pattern": "bullish",
    "cnn_confidence": 0.78,
    "xgboost_signal": "BUY",
    "xgboost_confidence": 0.71
  },
  
  "agent_analyses": {
    "agent_1_market_expert": {
      "market_regime": "recovery",
      "sentiment": "cautiously_optimistic",
      "recommendation": "favorable_for_long"
    },
    "agent_4_candlestick_specialist": {
      "key_patterns": ["Hammer", "Bullish Engulfing"],
      "pattern_summary": "Strong bullish reversal signals",
      "recommendation": "BUY",
      "confidence": 0.82
    },
    "agent_8_daytrade_executor": {
      "decision": "EXECUTE_BUY",
      "entry_strategy": "limit_order",
      "exit_strategy": "take_profit_and_stop_loss",
      "time_horizon": "4_hours"
    }
  },
  
  "validations": {
    "capital_check": "PASS",
    "risk_check": "PASS", 
    "rr_ratio_check": "PASS",
    "pattern_confirmation": "PASS",
    "ml_consensus": "PASS"
  },
  
  "execution_plan": {
    "order_type": "limit",
    "entry_price": 50300.00,
    "stop_loss": {
      "price": 49200.00,
      "percent": -2.2,
      "type": "stop_market"
    },
    "take_profit_1": {
      "price": 51500.00,
      "percent": 2.4,
      "quantity_percent": 50
    },
    "take_profit_2": {
      "price": 52800.00,
      "percent": 5.0,
      "quantity_percent": 50
    },
    "max_hold_time": "4h",
    "trailing_stop": false
  }
}
```

---

## 🚀 Exemplos Práticos

### Python

```python
import requests
import json
from datetime import datetime, timedelta

# Gerar candles de exemplo
def generate_sample_candles(count=20):
    candles = []
    base_price = 50000
    base_time = datetime.now() - timedelta(hours=count)
    
    for i in range(count):
        price = base_price + (i * 50) + (i % 3 - 1) * 100
        candles.append({
            "timestamp": (base_time + timedelta(hours=i)).isoformat() + "Z",
            "open": price,
            "high": price + 200,
            "low": price - 100,
            "close": price + 100,
            "volume": 1000000 + (i * 50000)
        })
    
    return candles

# Fazer análise
def analyze_market(symbol="BTC/USD", capital=10000):
    url = "http://localhost:8000/api/v1/analyze-candles"
    
    payload = {
        "symbol": symbol,
        "candles": generate_sample_candles(20),
        "capital_available": capital,
        "strategy": "scalping"
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"🎯 Sinal: {result['signal']}")
        print(f"📊 Confiança: {result['confidence']:.0%}")
        print(f"💰 Entry: ${result['entry_price']:,.2f}")
        print(f"🛑 Stop Loss: ${result['stop_loss']:,.2f}")
        print(f"🎯 Take Profit: ${result['take_profit_1']:,.2f}")
        print(f"💵 Quantidade: ${result['quantity_usd']:,.2f}")
        print(f"⚖️ Risk/Reward: {result['risk_reward_ratio']:.2f}:1")
        print(f"\n💭 Justificativa:")
        print(f"   {result['reasoning']}")
        
        return result
    else:
        print(f"❌ Erro: {response.status_code}")
        print(response.text)
        return None

# Executar
if __name__ == "__main__":
    result = analyze_market()
```

### JavaScript (Node.js)

```javascript
const axios = require('axios');

// Gerar candles de exemplo
function generateSampleCandles(count = 20) {
    const candles = [];
    const basePrice = 50000;
    const baseTime = new Date(Date.now() - count * 60 * 60 * 1000);
    
    for (let i = 0; i < count; i++) {
        const price = basePrice + (i * 50) + ((i % 3) - 1) * 100;
        const timestamp = new Date(baseTime.getTime() + i * 60 * 60 * 1000);
        
        candles.push({
            timestamp: timestamp.toISOString(),
            open: price,
            high: price + 200,
            low: price - 100,
            close: price + 100,
            volume: 1000000 + (i * 50000)
        });
    }
    
    return candles;
}

// Análise de mercado
async function analyzeMarket(symbol = 'BTC/USD', capital = 10000) {
    const url = 'http://localhost:8000/api/v1/analyze-candles';
    
    const payload = {
        symbol: symbol,
        candles: generateSampleCandles(20),
        capital_available: capital,
        strategy: 'scalping'
    };
    
    try {
        const response = await axios.post(url, payload);
        const result = response.data;
        
        console.log(`🎯 Sinal: ${result.signal}`);
        console.log(`📊 Confiança: ${(result.confidence * 100).toFixed(0)}%`);
        console.log(`💰 Entry: $${result.entry_price.toLocaleString()}`);
        console.log(`🛑 Stop Loss: $${result.stop_loss.toLocaleString()}`);
        console.log(`🎯 Take Profit: $${result.take_profit_1.toLocaleString()}`);
        console.log(`💵 Quantidade: $${result.quantity_usd.toLocaleString()}`);
        console.log(`⚖️ Risk/Reward: ${result.risk_reward_ratio.toFixed(2)}:1`);
        console.log(`\n💭 Justificativa:`);
        console.log(`   ${result.reasoning}`);
        
        return result;
    } catch (error) {
        console.error('❌ Erro:', error.response?.status || error.message);
        console.error(error.response?.data || error.message);
        return null;
    }
}

// Executar
analyzeMarket().then(result => {
    if (result) {
        console.log('\n✅ Análise concluída com sucesso!');
    }
});
```

### cURL

```bash
# Análise simples
curl -X POST "http://localhost:8000/api/v1/analyze-candles" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC/USD",
    "candles": [
      {
        "timestamp": "2025-10-08T10:00:00Z",
        "open": 50000,
        "high": 50200,
        "low": 49900,
        "close": 50100,
        "volume": 1000000
      },
      {
        "timestamp": "2025-10-08T11:00:00Z",
        "open": 50100,
        "high": 50300,
        "low": 50000,
        "close": 50200,
        "volume": 1100000
      }
    ],
    "capital_available": 10000,
    "strategy": "scalping"
  }' | jq

# Com arquivo
curl -X POST "http://localhost:8000/api/v1/analyze-candles" \
  -H "Content-Type: application/json" \
  -d @examples/btc_analysis.json | jq
```

---

## 🧪 Testando a API

### 1. Health Check

```bash
curl http://localhost:8000/health

# Resposta esperada:
{
  "status": "healthy",
  "timestamp": "2025-10-08T12:30:00Z",
  "services": {
    "ollama": "connected",
    "ml_models": "loaded",
    "database": "connected"
  }
}
```

### 2. Script de Teste Automático

```bash
# Executar teste completo
python scripts/test_api_live.py

# Saída esperada:
# 🧪 Testando API CeciAI...
# ✅ Health check: OK
# ✅ Análise de candles: OK
# ✅ Validação de response: OK
# ✅ Performance: 28.5s (OK)
# 🎉 Todos os testes passaram!
```

### 3. Swagger UI (Documentação Interativa)

```bash
# Abrir no navegador
open http://localhost:8000/docs

# Ou ReDoc
open http://localhost:8000/redoc
```

---

## ⚙️ Configuração Avançada

### Variáveis de Ambiente

```bash
# .env
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL_PRIMARY=llama3.2:3b

# Trading
INITIAL_CAPITAL=10000
MAX_DAILY_LOSS=0.03
ENABLE_TRADING=false
```

### Personalizar Estratégias

```python
# Estratégias disponíveis
strategies = [
    "scalping",      # 5-30min, lucro 0.5-1%
    "swing",         # 1-7 dias, lucro 3-5%  
    "arbitrage",     # Segundos, lucro 0.1-0.3%
    "long_term"      # Semanas/meses, lucro 10%+
]

# Usar estratégia específica
payload = {
    "symbol": "BTC/USD",
    "candles": candles,
    "strategy": "swing",  # Swing trading
    "capital_available": 50000
}
```

---

## 📊 Interpretando Resultados

### Sinais

- **BUY**: Oportunidade de compra identificada
- **SELL**: Oportunidade de venda identificada  
- **HOLD**: Aguardar melhor momento

### Níveis de Confiança

- **0.0 - 0.5**: Baixa confiança (evitar)
- **0.5 - 0.7**: Confiança moderada
- **0.7 - 0.85**: Alta confiança ✅
- **0.85 - 1.0**: Confiança muito alta ✅

### Score de Oportunidade

- **0-40**: Oportunidade fraca
- **40-60**: Oportunidade moderada
- **60-80**: Boa oportunidade ✅
- **80-100**: Excelente oportunidade ✅

### Validações

- **PASS**: Validação aprovada ✅
- **FAIL**: Validação reprovada (trade não recomendado) ❌

---

## 🐛 Troubleshooting

### Erro: "Connection refused"

```bash
# Verificar se API está rodando
curl http://localhost:8000/health

# Se não estiver, iniciar
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Erro: "Ollama not available"

```bash
# Verificar Ollama
ollama list

# Iniciar se necessário
ollama serve

# Baixar modelo
ollama pull llama3.2:3b
```

### Erro: "Minimum 10 candles required"

```bash
# Certifique-se de enviar pelo menos 10 candles
# Verifique o formato dos timestamps (ISO 8601)
```

### Response muito lento (> 60s)

```bash
# Verificar recursos do sistema
htop

# Usar modelo menor se necessário
export OLLAMA_MODEL_PRIMARY=llama3.2:1b
```

---

## 📈 Performance

### Tempos Esperados

| Operação | Tempo Esperado |
|----------|----------------|
| Health Check | < 100ms |
| Análise 10 candles | 15-25s |
| Análise 20 candles | 25-35s |
| Análise 50 candles | 35-50s |

### Otimizações

- Use cache para dados históricos
- Processe múltiplos símbolos em paralelo
- Configure workers do Uvicorn para produção

```bash
# Produção com múltiplos workers
uvicorn api.main:app --workers 4 --host 0.0.0.0 --port 8000
```

---

## 🔒 Segurança

### Boas Práticas

1. **Nunca** exponha a API publicamente sem autenticação
2. Use HTTPS em produção
3. Implemente rate limiting
4. Valide todos os inputs
5. Monitore logs de acesso

### Rate Limiting (Futuro)

```python
# Exemplo de implementação
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/analyze-candles")
@limiter.limit("10/minute")  # Máximo 10 requests por minuto
async def analyze_candles(request: Request, data: CandleAnalysisRequest):
    # ...
```

---

## 📞 Suporte

- **Documentação**: `/docs` (Swagger UI)
- **Health Check**: `/health`
- **Logs**: `logs/api.log`
- **Testes**: `python scripts/test_api_live.py`

---

**API pronta para análises inteligentes de trading! 🚀📈**

**Versão:** 1.0.0  
**Última atualização:** 2025-10-08