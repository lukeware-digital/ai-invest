"""
CeciAI - Scalping Strategy
Estratégia de scalping para trades rápidos (5-30 minutos)

Características:
- Timeframe: 1min, 5min
- Hold time: 5-30 minutos
- Target: 0.5-1% de lucro
- Stop loss: 0.3%
- Alta frequência, baixo risco

Autor: CeciAI Team
Data: 2025-10-08
"""

import logging
from typing import Any

import pandas as pd

from .base_strategy import BaseStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScalpingStrategy(BaseStrategy):
    """
    Estratégia de Scalping

    Foca em movimentos rápidos de preço com:
    - Entradas e saídas rápidas
    - Múltiplos trades por dia
    - Baixo risco por trade
    - Alta taxa de acerto necessária
    """

    def __init__(self):
        super().__init__(
            name="Scalping",
            timeframe="1min",
            max_risk_per_trade=0.005,  # 0.5% por trade
            min_risk_reward=1.5,
            max_stop_loss=0.003,  # 0.3%
        )

        # Parâmetros específicos do scalping
        self.target_profit_pct = 0.008  # 0.8% target
        self.max_hold_time_minutes = 30
        self.min_volume_ratio = 1.2  # Volume 20% acima da média
        self.min_confidence = 0.70

    async def analyze(
        self, df: pd.DataFrame, symbol: str, current_price: float, agent_analyses: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Analisa oportunidade de scalping.

        Critérios:
        1. Momentum forte (RSI entre 40-60)
        2. Volume acima da média
        3. Volatilidade adequada
        4. Sinais técnicos alinhados
        5. Padrões de candles favoráveis
        """
        logger.info(f"📊 Analisando oportunidade de scalping para {symbol}")

        analysis = {
            "strategy": self.name,
            "symbol": symbol,
            "current_price": current_price,
            "signal": "HOLD",
            "confidence": 0.0,
            "reasons": [],
            "warnings": [],
        }

        try:
            # Extrair análises dos agentes
            agent1 = agent_analyses.get("agent1", {})
            agent3 = agent_analyses.get("agent3", {})
            agent4 = agent_analyses.get("agent4", {})

            # 1. Verificar volume
            volume_ratio = agent1.get("market_metrics", {}).get("volume_ratio", 1.0)
            if volume_ratio < self.min_volume_ratio:
                analysis["warnings"].append(f"Volume baixo ({volume_ratio:.2f}x)")
            else:
                analysis["reasons"].append(f"Volume adequado ({volume_ratio:.2f}x)")

            # 2. Verificar RSI (ideal para scalping: 40-60)
            rsi = agent3.get("indicators", {}).get("rsi", 50)
            if 40 <= rsi <= 60:
                analysis["reasons"].append(f"RSI ideal para scalping ({rsi:.1f})")
            elif rsi < 30:
                analysis["signal"] = "BUY"
                analysis["reasons"].append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                analysis["signal"] = "SELL"
                analysis["reasons"].append(f"RSI overbought ({rsi:.1f})")

            # 3. Verificar MACD (momentum)
            macd_crossover = agent3.get("indicators", {}).get("macd_crossover", "neutral")
            if macd_crossover == "bullish":
                analysis["reasons"].append("MACD bullish crossover")
                if analysis["signal"] == "HOLD":
                    analysis["signal"] = "BUY"
            elif macd_crossover == "bearish":
                analysis["reasons"].append("MACD bearish crossover")
                if analysis["signal"] == "HOLD":
                    analysis["signal"] = "SELL"

            # 4. Verificar padrões de candles
            pattern_signal = agent4.get("signal", "HOLD")
            pattern_confidence = agent4.get("confidence", 0.5)

            if pattern_confidence >= self.min_confidence:
                analysis["reasons"].append(f"Padrão forte detectado ({pattern_signal})")

                if pattern_signal == "BUY" and analysis["signal"] != "SELL":
                    analysis["signal"] = "BUY"
                elif pattern_signal == "SELL" and analysis["signal"] != "BUY":
                    analysis["signal"] = "SELL"

            # 5. Verificar volatilidade
            volatility = agent1.get("market_metrics", {}).get("volatility", 0)
            if volatility > 3.0:
                analysis["warnings"].append(f"Alta volatilidade ({volatility:.2f}%)")
            elif volatility < 0.5:
                analysis["warnings"].append(f"Baixa volatilidade ({volatility:.2f}%)")

            # 6. Calcular confiança
            confidence_factors = []

            # Volume
            if volume_ratio >= self.min_volume_ratio:
                confidence_factors.append(0.2)

            # RSI
            if 40 <= rsi <= 60 or rsi < 30 or rsi > 70:
                confidence_factors.append(0.2)

            # MACD
            if macd_crossover in ["bullish", "bearish"]:
                confidence_factors.append(0.2)

            # Padrões
            if pattern_confidence >= self.min_confidence:
                confidence_factors.append(pattern_confidence * 0.4)

            analysis["confidence"] = sum(confidence_factors)

            # 7. Decisão final
            if analysis["signal"] == "HOLD":
                analysis["reasons"].append("Nenhum sinal claro para scalping")

            logger.info(
                f"✅ Análise concluída: {analysis['signal']} (confiança: {analysis['confidence']:.0%})"
            )

            return analysis

        except Exception as e:
            logger.error(f"Erro na análise: {e}", exc_info=True)
            analysis["signal"] = "HOLD"
            analysis["warnings"].append(f"Erro na análise: {e!s}")
            return analysis

    async def execute(self, analysis: dict[str, Any], capital_available: float) -> dict[str, Any]:
        """
        Executa trade de scalping.

        Returns:
            Plano de execução com preços e quantidades
        """
        logger.info("🚀 Executando estratégia de scalping")

        signal = analysis.get("signal", "HOLD")
        current_price = analysis.get("current_price", 0)

        if signal == "HOLD" or current_price == 0:
            return {
                "decision": "HOLD",
                "reasoning": "Nenhum sinal claro para scalping",
                "confidence": analysis.get("confidence", 0.0),
            }

        # Definir preços
        if signal == "BUY":
            entry_price = current_price
            stop_loss = entry_price * (1 - self.max_stop_loss)
            take_profit_1 = entry_price * (1 + self.target_profit_pct * 0.6)  # 60% do target
            take_profit_2 = entry_price * (1 + self.target_profit_pct)  # 100% do target
        else:  # SELL
            entry_price = current_price
            stop_loss = entry_price * (1 + self.max_stop_loss)
            take_profit_1 = entry_price * (1 - self.target_profit_pct * 0.6)
            take_profit_2 = entry_price * (1 - self.target_profit_pct)

        # Calcular tamanho da posição
        quantity_usd = self.calculate_position_size(entry_price, stop_loss, capital_available)

        # Validar trade
        validations = self.validate_trade(
            entry_price, stop_loss, take_profit_1, quantity_usd, capital_available
        )

        # Construir plano
        plan = {
            "decision": signal if validations["is_valid"] else "HOLD",
            "strategy": self.name,
            "confidence": analysis.get("confidence", 0.0),
            "entry_price": entry_price,
            "stop_loss": {
                "price": stop_loss,
                "distance_pct": abs(entry_price - stop_loss) / entry_price * 100,
            },
            "take_profit_1": {
                "price": take_profit_1,
                "distance_pct": abs(take_profit_1 - entry_price) / entry_price * 100,
            },
            "take_profit_2": {
                "price": take_profit_2,
                "distance_pct": abs(take_profit_2 - entry_price) / entry_price * 100,
            },
            "quantity_usd": quantity_usd,
            "risk_amount": abs(entry_price - stop_loss) / entry_price * quantity_usd,
            "reward_amount": abs(take_profit_1 - entry_price) / entry_price * quantity_usd,
            "risk_reward_ratio": abs(take_profit_1 - entry_price) / abs(entry_price - stop_loss)
            if abs(entry_price - stop_loss) > 0
            else 0,
            "max_hold_time": f"{self.max_hold_time_minutes} minutos",
            "reasoning": " | ".join(analysis.get("reasons", [])),
            "warnings": analysis.get("warnings", []),
            "validations": validations,
        }

        if not validations["is_valid"]:
            plan["decision"] = "HOLD"
            plan["reasoning"] = f"Validação falhou: {', '.join(validations['errors'])}"

        logger.info(f"✅ Plano de execução: {plan['decision']}")

        return plan


# Exemplo de uso
if __name__ == "__main__":
    import asyncio

    async def test_scalping():
        # Dados de teste
        data = {
            "open": [50000 + i * 10 for i in range(100)],
            "high": [50000 + i * 10 + 50 for i in range(100)],
            "low": [50000 + i * 10 - 30 for i in range(100)],
            "close": [50000 + i * 10 + 20 for i in range(100)],
            "volume": [1000000 + i * 10000 for i in range(100)],
        }
        df = pd.DataFrame(data)

        # Análises mockadas dos agentes
        agent_analyses = {
            "agent1": {"market_metrics": {"volume_ratio": 1.5, "volatility": 1.2}},
            "agent3": {"indicators": {"rsi": 45, "macd_crossover": "bullish"}},
            "agent4": {"signal": "BUY", "confidence": 0.75},
        }

        # Testar estratégia
        strategy = ScalpingStrategy()

        analysis = await strategy.analyze(df, "BTC/USD", 51000, agent_analyses)

        print("\n📊 ANÁLISE:")
        print(f"Sinal: {analysis['signal']}")
        print(f"Confiança: {analysis['confidence']:.0%}")
        print(f"Razões: {analysis['reasons']}")

        if analysis["signal"] != "HOLD":
            plan = await strategy.execute(analysis, 10000)

            print("\n🚀 PLANO DE EXECUÇÃO:")
            print(f"Decisão: {plan['decision']}")
            print(f"Entry: ${plan['entry_price']:,.2f}")
            print(f"Stop Loss: ${plan['stop_loss']['price']:,.2f}")
            print(f"Take Profit 1: ${plan['take_profit_1']['price']:,.2f}")
            print(f"Quantidade: ${plan['quantity_usd']:,.2f}")
            print(f"R/R: {plan['risk_reward_ratio']:.2f}")

    asyncio.run(test_scalping())
