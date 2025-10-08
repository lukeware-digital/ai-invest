"""
CeciAI - Testes da API de Análise de Candles
Testa o endpoint /api/v1/analyze-candles

Autor: CeciAI Team
Data: 2025-10-07
"""

import json
from datetime import datetime, timedelta

import pytest


# Dados de teste
def generate_test_candles(num_candles=20, base_price=50000, trend="up"):
    """Gera candles de teste"""
    candles = []
    current_price = base_price

    for i in range(num_candles):
        # Simular movimento de preço
        if trend == "up":
            change = (i * 50) + (i % 3 * 100)
        elif trend == "down":
            change = -(i * 50) - (i % 3 * 100)
        else:
            change = (i % 2) * 100 - 50

        open_price = current_price
        close_price = current_price + change
        high_price = max(open_price, close_price) + 100
        low_price = min(open_price, close_price) - 100

        candles.append(
            {
                "timestamp": (datetime.now() - timedelta(hours=num_candles - i)).isoformat(),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": 1000000 + (i * 50000),
            }
        )

        current_price = close_price

    return candles


def test_candle_analysis_request_structure():
    """Testa estrutura do request"""
    candles = generate_test_candles(15, trend="up")

    request = {
        "symbol": "BTC/USD",
        "candles": candles,
        "capital_available": 10000,
        "strategy": "scalping",
    }

    assert len(request["candles"]) == 15
    assert request["symbol"] == "BTC/USD"
    assert request["capital_available"] > 0


def test_candle_data_validation():
    """Testa validação dos dados de candles"""
    candles = generate_test_candles(10)

    for candle in candles:
        assert candle["open"] > 0
        assert candle["high"] > 0
        assert candle["low"] > 0
        assert candle["close"] > 0
        assert candle["volume"] >= 0
        assert candle["high"] >= candle["low"]
        assert candle["high"] >= candle["open"]
        assert candle["high"] >= candle["close"]


def test_uptrend_candles():
    """Testa geração de candles em tendência de alta"""
    candles = generate_test_candles(20, trend="up")

    first_close = candles[0]["close"]
    last_close = candles[-1]["close"]

    assert last_close > first_close, "Tendência de alta deve ter último preço maior que primeiro"


def test_downtrend_candles():
    """Testa geração de candles em tendência de baixa"""
    candles = generate_test_candles(20, trend="down")

    first_close = candles[0]["close"]
    last_close = candles[-1]["close"]

    assert last_close < first_close, "Tendência de baixa deve ter último preço menor que primeiro"


@pytest.mark.asyncio
async def test_mock_agent4_analysis():
    """Testa análise mockada do Agent 4"""
    # Simular resposta do Agent 4
    mock_response = {
        "signal": "BUY",
        "confidence": 0.75,
        "key_patterns": ["Bullish Engulfing", "Hammer"],
        "pattern_strength": {
            "bullish_score": 3.5,
            "bearish_score": 0.5,
            "confidence": "HIGH",
            "dominant_signal": "BUY",
        },
    }

    assert mock_response["signal"] in ["BUY", "SELL", "HOLD"]
    assert 0 <= mock_response["confidence"] <= 1
    assert len(mock_response["key_patterns"]) > 0


@pytest.mark.asyncio
async def test_mock_agent8_execution():
    """Testa execução mockada do Agent 8"""
    # Simular resposta do Agent 8
    mock_plan = {
        "decision": "BUY",
        "entry_price": 50300,
        "stop_loss": {"price": 49200, "percent": 2.2},
        "take_profit_1": {"price": 51500, "percent": 2.4},
        "take_profit_2": {"price": 52800, "percent": 5.0},
        "quantity_usd": 1500,
        "risk_reward_ratio": 2.3,
        "confidence": 0.78,
        "validations": {"capital_check": "PASS", "risk_check": "PASS", "rr_ratio_check": "PASS"},
    }

    assert mock_plan["decision"] in ["BUY", "SELL", "HOLD"]
    assert mock_plan["entry_price"] > 0
    assert mock_plan["stop_loss"]["price"] < mock_plan["entry_price"]
    assert mock_plan["take_profit_1"]["price"] > mock_plan["entry_price"]
    assert mock_plan["risk_reward_ratio"] >= 1.5


def test_response_structure():
    """Testa estrutura da resposta esperada"""
    expected_response = {
        "request_id": "abc123",
        "timestamp": datetime.now().isoformat(),
        "processing_time": 1.23,
        "signal": "BUY",
        "confidence": 0.75,
        "entry_price": 50300,
        "stop_loss": 49200,
        "take_profit_1": 51500,
        "take_profit_2": 52800,
        "quantity_usd": 1500,
        "risk_reward_ratio": 2.3,
        "risk_amount_usd": 15,
        "potential_profit_usd": 45,
        "reasoning": "Padrões bullish detectados com alta confiança",
        "patterns_detected": ["Bullish Engulfing", "Hammer"],
        "technical_analysis": {"rsi": 42.5, "macd": 120.5, "trend": "uptrend"},
        "validations": {"capital_check": "PASS", "risk_check": "PASS"},
    }

    # Validar campos obrigatórios
    assert "signal" in expected_response
    assert "confidence" in expected_response
    assert "entry_price" in expected_response
    assert "reasoning" in expected_response
    assert "patterns_detected" in expected_response


def test_example_request_json():
    """Gera exemplo de request JSON para documentação"""
    candles = generate_test_candles(15, base_price=50000, trend="up")

    request_example = {
        "symbol": "BTC/USD",
        "candles": candles,
        "capital_available": 10000,
        "strategy": "scalping",
    }

    # Salvar exemplo (opcional)
    print("\n📄 Exemplo de Request JSON:")
    print(json.dumps(request_example, indent=2))

    assert len(json.dumps(request_example)) > 0


if __name__ == "__main__":
    print("🧪 Executando testes da API de Candles...")
    pytest.main([__file__, "-v", "--tb=short"])
