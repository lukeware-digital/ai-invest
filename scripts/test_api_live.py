#!/usr/bin/env python3
"""
CeciAI - Script de Teste da API (Live)
Testa o endpoint /api/v1/analyze-candles com dados reais

Uso:
    python scripts/test_api_live.py

Autor: CeciAI Team
Data: 2025-10-07
"""

import sys
from datetime import datetime, timedelta

import requests


def generate_test_candles(num_candles=20, base_price=50000, trend="up"):
    """Gera candles de teste simulados"""
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
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(close_price),
                "volume": float(1000000 + (i * 50000)),
            }
        )

        current_price = close_price

    return candles


def test_api_endpoint(base_url="http://localhost:8000"):
    """Testa o endpoint de análise de candles"""

    print("🧪 Testando API CeciAI - Análise de Candles\n")
    print(f"📡 URL: {base_url}/api/v1/analyze-candles\n")

    # Gerar candles de teste
    print("📊 Gerando 20 candles de teste (tendência de alta)...")
    candles = generate_test_candles(20, base_price=50000, trend="up")

    # Preparar request
    request_data = {
        "symbol": "BTC/USD",
        "candles": candles,
        "capital_available": 10000.0,
        "strategy": "scalping",
    }

    print(f"   • Símbolo: {request_data['symbol']}")
    print(f"   • Candles: {len(request_data['candles'])}")
    print(f"   • Capital: ${request_data['capital_available']:,.2f}")
    print(f"   • Estratégia: {request_data['strategy']}")
    print(f"   • Preço inicial: ${candles[0]['close']:,.2f}")
    print(f"   • Preço final: ${candles[-1]['close']:,.2f}\n")

    # Fazer request
    print("🚀 Enviando request para API...")
    try:
        response = requests.post(
            f"{base_url}/api/v1/analyze-candles", json=request_data, timeout=30
        )

        # Verificar resposta
        if response.status_code == 200:
            print("✅ Request bem-sucedido!\n")

            result = response.json()

            # Exibir resultados
            print("=" * 60)
            print("📊 RESULTADO DA ANÁLISE")
            print("=" * 60)
            print(f"Request ID:      {result['request_id']}")
            print(f"Tempo Processo:  {result['processing_time']:.2f}s")
            print(f"\n🎯 DECISÃO:       {result['signal']}")
            print(f"Confiança:       {result['confidence']:.0%}")
            print("\n💰 PREÇOS:")
            print(f"   Entry:        ${result['entry_price']:,.2f}")
            print(f"   Stop Loss:    ${result['stop_loss']:,.2f}")
            print(f"   Take Profit 1: ${result['take_profit_1']:,.2f}")
            print(f"   Take Profit 2: ${result['take_profit_2']:,.2f}")
            print("\n📈 EXECUÇÃO:")
            print(f"   Quantidade:   ${result['quantity_usd']:,.2f}")
            print(f"   Risco:        ${result['risk_amount_usd']:,.2f}")
            print(f"   Lucro Potencial: ${result['potential_profit_usd']:,.2f}")
            print(f"   Risk/Reward:  {result['risk_reward_ratio']:.2f}:1")
            print("\n🕯️  PADRÕES DETECTADOS:")
            for pattern in result["patterns_detected"]:
                print(f"   • {pattern}")
            print("\n📊 INDICADORES TÉCNICOS:")
            ta = result["technical_analysis"]
            print(f"   RSI:          {ta['rsi']:.2f}")
            print(f"   MACD:         {ta['macd']:.2f}")
            print(f"   Tendência:    {ta['trend']}")
            print(f"   Variação:     {ta['price_change_pct']:.2f}%")
            print("\n💭 JUSTIFICATIVA:")
            print(f"   {result['reasoning']}")
            print("\n✅ VALIDAÇÕES:")
            for key, value in result["validations"].items():
                print(f"   {key}: {value}")
            print("=" * 60)

            return True

        else:
            print(f"❌ Erro na request: {response.status_code}")
            print(f"Detalhes: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Erro: Não foi possível conectar à API")
        print("   Certifique-se de que a API está rodando:")
        print("   uvicorn api.main:app --reload")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False


def test_health_endpoint(base_url="http://localhost:8000"):
    """Testa o endpoint de health check"""
    print("\n🏥 Testando Health Check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print("✅ API está saudável!")
            print(f"   Status: {health['status']}")
            print(f"   Versão: {health['version']}")
            print(f"   Agentes: {health.get('services', {})}")
            return True
        else:
            print(f"⚠️  Health check retornou: {response.status_code}")
            return False
    except Exception:
        print("❌ Health check falhou")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🤖 CeciAI - Teste da API de Análise de Candles")
    print("=" * 60 + "\n")

    # Testar health primeiro
    if not test_health_endpoint():
        print("\n⚠️  API pode não estar rodando. Iniciando teste mesmo assim...\n")

    # Testar endpoint principal
    success = test_api_endpoint()

    if success:
        print("\n✅ Todos os testes passaram!")
        sys.exit(0)
    else:
        print("\n❌ Alguns testes falharam")
        sys.exit(1)
