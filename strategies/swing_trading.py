"""
CeciAI - Swing Trading Strategy
Estratégia de swing trading para trades de médio prazo (1-7 dias)

Características:
- Timeframe: 4h, 1d
- Hold time: 1-7 dias
- Target: 3-5% de lucro
- Stop loss: 2%

Autor: CeciAI Team
Data: 2025-10-08
"""

import logging
from typing import Any

import pandas as pd

from .base_strategy import BaseStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SwingTradingStrategy(BaseStrategy):
    """Estratégia de Swing Trading para trades de médio prazo"""

    def __init__(self):
        super().__init__(
            name="Swing Trading",
            timeframe="4h",
            max_risk_per_trade=0.01,  # 1% por trade
            min_risk_reward=2.0,
            max_stop_loss=0.02,  # 2%
        )

        self.target_profit_pct = 0.04  # 4% target
        self.max_hold_time_days = 7
        self.min_confidence = 0.65

    async def analyze(
        self, df: pd.DataFrame, symbol: str, current_price: float, agent_analyses: dict[str, Any]
    ) -> dict[str, Any]:
        """Analisa oportunidade de swing trading"""
        logger.info(f"📊 Analisando swing trade para {symbol}")

        analysis = {
            "strategy": self.name,
            "symbol": symbol,
            "current_price": current_price,
            "signal": "HOLD",
            "confidence": 0.0,
            "reasons": [],
        }

        agent1 = agent_analyses.get("agent1", {})
        agent3 = agent_analyses.get("agent3", {})

        # Verificar tendência
        trend = agent1.get("market_metrics", {}).get("trend", "neutral")
        if trend == "uptrend":
            analysis["signal"] = "BUY"
            analysis["reasons"].append("Tendência de alta confirmada")
        elif trend == "downtrend":
            analysis["signal"] = "SELL"
            analysis["reasons"].append("Tendência de baixa confirmada")

        # RSI
        rsi = agent3.get("indicators", {}).get("rsi", 50)
        if rsi < 35:
            analysis["signal"] = "BUY"
            analysis["reasons"].append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 65:
            analysis["signal"] = "SELL"
            analysis["reasons"].append(f"RSI overbought ({rsi:.1f})")

        # Confiança
        analysis["confidence"] = 0.70 if len(analysis["reasons"]) >= 2 else 0.50

        return analysis

    async def execute(self, analysis: dict[str, Any], capital_available: float) -> dict[str, Any]:
        """Executa swing trade"""
        signal = analysis.get("signal", "HOLD")
        current_price = analysis.get("current_price", 0)

        if signal == "HOLD":
            return {"decision": "HOLD", "reasoning": "Sem sinal claro"}

        # Preços
        if signal == "BUY":
            entry_price = current_price
            stop_loss = entry_price * (1 - self.max_stop_loss)
            take_profit = entry_price * (1 + self.target_profit_pct)
        else:
            entry_price = current_price
            stop_loss = entry_price * (1 + self.max_stop_loss)
            take_profit = entry_price * (1 - self.target_profit_pct)

        quantity_usd = self.calculate_position_size(entry_price, stop_loss, capital_available)
        validations = self.validate_trade(
            entry_price, stop_loss, take_profit, quantity_usd, capital_available
        )

        return {
            "decision": signal if validations["is_valid"] else "HOLD",
            "strategy": self.name,
            "entry_price": entry_price,
            "stop_loss": {"price": stop_loss},
            "take_profit": {"price": take_profit},
            "quantity_usd": quantity_usd,
            "validations": validations,
        }
