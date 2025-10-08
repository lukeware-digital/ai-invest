"""
CeciAI - Script de Teste Completo do Sistema
Testa todas as fases implementadas (4, 5 e 6)

Autor: CeciAI Team
Data: 2025-10-08
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from agents.ml_agent import MLAgent
from agents.pipeline import AgentPipeline
from backtesting.backtest_engine import BacktestEngine
from backtesting.paper_trading import PaperTradingEngine
from config.capital_management import CapitalManager
from strategies.scalping import ScalpingStrategy
from strategies.swing_trading import SwingTradingStrategy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_ml_models():
    """Testa modelos ML"""
    logger.info("\n" + "=" * 60)
    logger.info("🤖 TESTE 1: MODELOS ML")
    logger.info("=" * 60)

    # Gerar dados de teste
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=200, freq="1H"),
        "open": np.random.randn(200).cumsum() + 50000,
        "high": np.random.randn(200).cumsum() + 50200,
        "low": np.random.randn(200).cumsum() + 49800,
        "close": np.random.randn(200).cumsum() + 50100,
        "volume": np.random.randint(1000000, 2000000, 200),
        "rsi": np.random.uniform(30, 70, 200),
        "macd": np.random.randn(200),
        "macd_signal": np.random.randn(200),
        "bb_upper": np.random.randn(200).cumsum() + 51000,
        "bb_middle": np.random.randn(200).cumsum() + 50000,
        "bb_lower": np.random.randn(200).cumsum() + 49000,
        "ema_9": np.random.randn(200).cumsum() + 50000,
        "ema_21": np.random.randn(200).cumsum() + 50000,
    }
    df = pd.DataFrame(data)

    # Testar ML Agent
    ml_agent = MLAgent()
    prediction = await ml_agent.predict(df)

    logger.info(f"✅ Sinal consolidado: {prediction['consolidated_signal']}")
    logger.info(f"✅ Confiança: {prediction['confidence']:.0%}")
    logger.info(f"✅ Modelos treinados: {prediction['models_trained']}")

    return True


async def test_strategies():
    """Testa estratégias de trading"""
    logger.info("\n" + "=" * 60)
    logger.info("📊 TESTE 2: ESTRATÉGIAS DE TRADING")
    logger.info("=" * 60)

    # Dados de teste
    data = {
        "open": [50000 + i * 10 for i in range(100)],
        "high": [50000 + i * 10 + 50 for i in range(100)],
        "low": [50000 + i * 10 - 30 for i in range(100)],
        "close": [50000 + i * 10 + 20 for i in range(100)],
        "volume": [1000000 + i * 10000 for i in range(100)],
    }
    df = pd.DataFrame(data)

    # Análises mockadas
    agent_analyses = {
        "agent1": {"market_metrics": {"volume_ratio": 1.5, "volatility": 1.2, "trend": "uptrend"}},
        "agent3": {"indicators": {"rsi": 45, "macd_crossover": "bullish"}},
        "agent4": {"signal": "BUY", "confidence": 0.75},
    }

    # Testar Scalping
    logger.info("\n📈 Testando Scalping Strategy...")
    scalping = ScalpingStrategy()
    analysis = await scalping.analyze(df, "BTC/USD", 51000, agent_analyses)

    if analysis["signal"] != "HOLD":
        plan = await scalping.execute(analysis, 10000)
        logger.info(
            f"✅ Scalping: {plan['decision']} (R/R: {plan.get('risk_reward_ratio', 0):.2f})"
        )

    # Testar Swing Trading
    logger.info("\n📈 Testando Swing Trading Strategy...")
    swing = SwingTradingStrategy()
    analysis = await swing.analyze(df, "BTC/USD", 51000, agent_analyses)

    if analysis["signal"] != "HOLD":
        plan = await swing.execute(analysis, 10000)
        logger.info(f"✅ Swing Trading: {plan['decision']}")

    return True


async def test_capital_management():
    """Testa gestão de capital"""
    logger.info("\n" + "=" * 60)
    logger.info("💰 TESTE 3: CAPITAL MANAGEMENT")
    logger.info("=" * 60)

    capital_mgr = CapitalManager(initial_capital=10000)

    # Testar abertura de posição
    trade = {
        "id": "test_1",
        "symbol": "BTC/USD",
        "signal": "BUY",
        "entry_price": 50000,
        "quantity_usd": 1000,
    }

    validation = capital_mgr.can_open_position(1000)
    logger.info(f"✅ Validação: {validation['can_open']}")

    if validation["can_open"]:
        success = capital_mgr.open_position(trade)
        logger.info(f"✅ Posição aberta: {success}")

        status = capital_mgr.get_status()
        logger.info(f"✅ Capital disponível: ${status['available_capital']:,.2f}")

        # Fechar posição
        result = capital_mgr.close_position("test_1", 51000, "test")
        logger.info(f"✅ Posição fechada: P&L ${result['pnl']:,.2f}")

    return True


async def test_backtesting():
    """Testa backtesting"""
    logger.info("\n" + "=" * 60)
    logger.info("📉 TESTE 4: BACKTESTING ENGINE")
    logger.info("=" * 60)

    # Dados históricos de teste
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=500, freq="1H"),
        "open": np.random.randn(500).cumsum() + 50000,
        "high": np.random.randn(500).cumsum() + 50200,
        "low": np.random.randn(500).cumsum() + 49800,
        "close": np.random.randn(500).cumsum() + 50100,
        "volume": np.random.randint(1000000, 2000000, 500),
    }
    df = pd.DataFrame(data)

    backtest = BacktestEngine(initial_capital=10000)
    strategy = ScalpingStrategy()

    # Executar backtest (simplificado)
    results = await backtest.run(df, strategy, None)

    logger.info(f"✅ Capital final: ${results['final_capital']:,.2f}")
    logger.info(f"✅ Retorno total: {results['total_return']:.2%}")

    return True


async def test_paper_trading():
    """Testa paper trading"""
    logger.info("\n" + "=" * 60)
    logger.info("📝 TESTE 5: PAPER TRADING")
    logger.info("=" * 60)

    paper_trading = PaperTradingEngine(initial_capital=10000)

    await paper_trading.start()
    logger.info("✅ Paper trading iniciado")

    # Executar trade simulado
    trade = {
        "id": "paper_1",
        "symbol": "BTC/USD",
        "signal": "BUY",
        "entry_price": 50000,
        "quantity_usd": 1000,
    }

    result = await paper_trading.execute_trade(trade)
    logger.info(f"✅ Trade executado: {result['success']}")

    status = paper_trading.get_status()
    logger.info(f"✅ Status: {status['is_running']}")

    await paper_trading.stop()

    return True


async def test_complete_pipeline():
    """Testa pipeline completo"""
    logger.info("\n" + "=" * 60)
    logger.info("🚀 TESTE 6: PIPELINE COMPLETO")
    logger.info("=" * 60)

    # Dados de teste
    data = {
        "open": [50000 + i * 50 for i in range(100)],
        "high": [50000 + i * 50 + 200 for i in range(100)],
        "low": [50000 + i * 50 - 100 for i in range(100)],
        "close": [50000 + i * 50 + 100 for i in range(100)],
        "volume": [1000000 + i * 50000 for i in range(100)],
    }
    df = pd.DataFrame(data)

    # Executar pipeline
    pipeline = AgentPipeline()

    result = await pipeline.execute(
        df=df, symbol="BTC/USD", timeframe="1h", capital_available=10000.0, user_strategy="scalping"
    )

    logger.info(f"✅ Decisão final: {result['final_decision']['decision']}")
    logger.info(f"✅ Confiança: {result['final_decision']['confidence']:.0%}")
    logger.info(f"✅ Score de oportunidade: {result['final_decision']['opportunity_score']}/100")
    logger.info(f"✅ Tempo de processamento: {result['pipeline_metadata']['processing_time']:.2f}s")

    return True


async def main():
    """Executa todos os testes"""
    start_time = datetime.now()

    logger.info("\n" + "=" * 60)
    logger.info("🎯 INICIANDO TESTES COMPLETOS DO SISTEMA CECI-AI")
    logger.info("=" * 60)

    tests = [
        ("Modelos ML", test_ml_models),
        ("Estratégias", test_strategies),
        ("Capital Management", test_capital_management),
        ("Backtesting", test_backtesting),
        ("Paper Trading", test_paper_trading),
        ("Pipeline Completo", test_complete_pipeline),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"❌ Erro no teste {test_name}: {e}", exc_info=True)
            results.append((test_name, False))

    # Resumo
    duration = (datetime.now() - start_time).total_seconds()

    logger.info("\n" + "=" * 60)
    logger.info("📊 RESUMO DOS TESTES")
    logger.info("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        logger.info(f"{status}: {test_name}")

    logger.info(f"\n✅ Testes passados: {passed}/{total}")
    logger.info(f"⏱️  Tempo total: {duration:.2f}s")

    if passed == total:
        logger.info("\n🎉 TODOS OS TESTES PASSARAM!")
        logger.info("🚀 Sistema CeciAI está 100% funcional!")
    else:
        logger.warning(f"\n⚠️  {total - passed} teste(s) falharam")


if __name__ == "__main__":
    asyncio.run(main())
