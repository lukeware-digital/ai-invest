"""
CeciAI - Agent 3: Technical Analyst Strategist
Especialista em análise técnica e indicadores

Responsabilidades:
- Analisar indicadores técnicos (RSI, MACD, Bollinger Bands, etc)
- Identificar sinais de compra/venda baseados em TA
- Avaliar força e direção da tendência
- Detectar divergências e confluências
- Fornecer níveis de suporte e resistência

Autor: CeciAI Team
Data: 2025-10-08
"""

import json
import logging
from typing import Any

import ollama
import pandas as pd

from utils.technical_indicators import calculate_bollinger_bands, calculate_macd, calculate_rsi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalAnalystStrategist:
    """
    Agent 3: Especialista em Análise Técnica

    Analisa indicadores técnicos e fornece sinais baseados em TA.
    """

    def __init__(self, model: str = "llama3.2:3b"):
        """
        Inicializa o agente.

        Args:
            model: Modelo Ollama a usar
        """
        self.model = model
        logger.info(f"Agent 3 (Technical Analyst) inicializado com modelo {model}")

    async def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        market_context: dict[str, Any] | None = None,
        data_quality: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Analisa indicadores técnicos.

        Args:
            df: DataFrame com dados OHLCV
            symbol: Par de trading
            market_context: Contexto do Agent 1
            data_quality: Análise do Agent 2

        Returns:
            Dict com análise técnica
        """
        if market_context is None:
            market_context = {}
        if data_quality is None:
            data_quality = {}

        try:
            # Calcular indicadores
            indicators = self._calculate_indicators(df)

            # Identificar sinais
            signals = self._identify_signals(indicators, df)

            # Detectar níveis
            levels = self._detect_levels(df)

            # Preparar prompt para LLM
            prompt = self._build_prompt(
                df, symbol, indicators, signals, levels, market_context, data_quality
            )

            # Consultar LLM
            logger.info("Agent 3: Consultando LLM para análise técnica...")
            response = ollama.generate(model=self.model, prompt=prompt, format="json")

            # Parsear resposta
            analysis = self._parse_response(response["response"])

            # Adicionar dados calculados
            analysis["indicators"] = indicators
            analysis["signals"] = signals
            analysis["levels"] = levels
            analysis["agent"] = "agent_3_ta_strategist"

            logger.info(f"Agent 3: Sinal técnico: {analysis.get('technical_signal', 'N/A')}")

            return analysis

        except Exception as e:
            logger.error(f"Agent 3: Erro na análise: {e}", exc_info=True)
            return self._get_default_response(str(e))

    def _calculate_indicators(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calcula indicadores técnicos"""

        indicators = {}

        # RSI
        rsi = calculate_rsi(df["close"])
        indicators["rsi"] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        indicators["rsi_signal"] = (
            "oversold"
            if indicators["rsi"] < 30
            else "overbought"
            if indicators["rsi"] > 70
            else "neutral"
        )

        # MACD
        macd_data = calculate_macd(df["close"])
        indicators["macd"] = (
            float(macd_data["macd"].iloc[-1]) if not pd.isna(macd_data["macd"].iloc[-1]) else 0.0
        )
        indicators["macd_signal"] = (
            float(macd_data["signal"].iloc[-1])
            if not pd.isna(macd_data["signal"].iloc[-1])
            else 0.0
        )
        indicators["macd_histogram"] = indicators["macd"] - indicators["macd_signal"]
        indicators["macd_crossover"] = (
            "bullish" if indicators["macd"] > indicators["macd_signal"] else "bearish"
        )

        # Bollinger Bands
        bb_data = calculate_bollinger_bands(df["close"])
        current_price = float(df["close"].iloc[-1])
        bb_upper = (
            float(bb_data["upper"].iloc[-1])
            if not pd.isna(bb_data["upper"].iloc[-1])
            else current_price * 1.02
        )
        bb_lower = (
            float(bb_data["lower"].iloc[-1])
            if not pd.isna(bb_data["lower"].iloc[-1])
            else current_price * 0.98
        )
        bb_middle = (
            float(bb_data["middle"].iloc[-1])
            if not pd.isna(bb_data["middle"].iloc[-1])
            else current_price
        )

        indicators["bb_upper"] = bb_upper
        indicators["bb_middle"] = bb_middle
        indicators["bb_lower"] = bb_lower
        indicators["bb_position"] = (
            "upper"
            if current_price > bb_upper
            else "lower"
            if current_price < bb_lower
            else "middle"
        )
        indicators["bb_width"] = ((bb_upper - bb_lower) / bb_middle) * 100

        # EMAs
        ema_9 = df["close"].ewm(span=9).mean()
        ema_21 = df["close"].ewm(span=21).mean()
        indicators["ema_9"] = float(ema_9.iloc[-1])
        indicators["ema_21"] = float(ema_21.iloc[-1])
        indicators["ema_crossover"] = (
            "bullish" if indicators["ema_9"] > indicators["ema_21"] else "bearish"
        )

        return indicators

    def _identify_signals(self, indicators: dict, df: pd.DataFrame) -> dict[str, Any]:
        """Identifica sinais de trading"""

        signals = {"bullish": [], "bearish": [], "neutral": []}

        # RSI signals
        if indicators["rsi"] < 30:
            signals["bullish"].append("RSI oversold (<30)")
        elif indicators["rsi"] > 70:
            signals["bearish"].append("RSI overbought (>70)")
        else:
            signals["neutral"].append(f"RSI neutral ({indicators['rsi']:.1f})")

        # MACD signals
        if indicators["macd_crossover"] == "bullish" and indicators["macd_histogram"] > 0:
            signals["bullish"].append("MACD bullish crossover")
        elif indicators["macd_crossover"] == "bearish" and indicators["macd_histogram"] < 0:
            signals["bearish"].append("MACD bearish crossover")

        # Bollinger Bands signals
        if indicators["bb_position"] == "lower":
            signals["bullish"].append("Price at lower BB (potential bounce)")
        elif indicators["bb_position"] == "upper":
            signals["bearish"].append("Price at upper BB (potential reversal)")

        # EMA signals
        if indicators["ema_crossover"] == "bullish":
            signals["bullish"].append("EMA 9 > EMA 21 (bullish trend)")
        else:
            signals["bearish"].append("EMA 9 < EMA 21 (bearish trend)")

        # Contar sinais
        bullish_count = len(signals["bullish"])
        bearish_count = len(signals["bearish"])

        if bullish_count > bearish_count:
            signals["dominant"] = "bullish"
            signals["strength"] = "strong" if bullish_count >= 3 else "moderate"
        elif bearish_count > bullish_count:
            signals["dominant"] = "bearish"
            signals["strength"] = "strong" if bearish_count >= 3 else "moderate"
        else:
            signals["dominant"] = "neutral"
            signals["strength"] = "weak"

        return signals

    def _detect_levels(self, df: pd.DataFrame) -> dict[str, Any]:
        """Detecta níveis de suporte e resistência"""

        current_price = float(df["close"].iloc[-1])

        # Suporte e resistência simples (highs e lows recentes)
        recent_highs = df["high"].tail(20).nlargest(3)
        recent_lows = df["low"].tail(20).nsmallest(3)

        resistance_levels = [float(h) for h in recent_highs if h > current_price][:2]
        support_levels = [float(low) for low in recent_lows if low < current_price][:2]

        return {
            "current_price": current_price,
            "resistance": resistance_levels,
            "support": support_levels,
            "nearest_resistance": resistance_levels[0]
            if resistance_levels
            else current_price * 1.02,
            "nearest_support": support_levels[0] if support_levels else current_price * 0.98,
        }

    def _build_prompt(
        self,
        df: pd.DataFrame,
        symbol: str,
        indicators: dict,
        signals: dict,
        levels: dict,
        market_context: dict,
        data_quality: dict,
    ) -> str:
        """Constrói prompt para o LLM"""

        prompt = f"""
Você é um analista técnico profissional com 20 anos de experiência em trading.

SÍMBOLO: {symbol}
PREÇO ATUAL: ${levels['current_price']:,.2f}

INDICADORES TÉCNICOS:
- RSI: {indicators['rsi']:.2f} ({indicators['rsi_signal']})
- MACD: {indicators['macd']:.2f}
- MACD Signal: {indicators['macd_signal']:.2f}
- MACD Histogram: {indicators['macd_histogram']:.2f}
- MACD Crossover: {indicators['macd_crossover']}
- Bollinger Bands: {indicators['bb_position']}
- BB Width: {indicators['bb_width']:.2f}%
- EMA 9: ${indicators['ema_9']:,.2f}
- EMA 21: ${indicators['ema_21']:,.2f}
- EMA Crossover: {indicators['ema_crossover']}

SINAIS IDENTIFICADOS:
Bullish: {', '.join(signals['bullish']) if signals['bullish'] else 'Nenhum'}
Bearish: {', '.join(signals['bearish']) if signals['bearish'] else 'Nenhum'}
Dominante: {signals['dominant']} ({signals['strength']})

NÍVEIS:
- Resistência: {', '.join([f'${r:,.2f}' for r in levels['resistance']]) if levels['resistance'] else 'N/A'}
- Suporte: {', '.join([f'${s:,.2f}' for s in levels['support']]) if levels['support'] else 'N/A'}

CONTEXTO DE MERCADO:
- Regime: {market_context.get('market_regime', 'N/A')}
- Sentimento: {market_context.get('sentiment', 'N/A')}
- Qualidade dos Dados: {data_quality.get('overall_quality', 'N/A')}

TAREFA:
Analise os indicadores técnicos e forneça uma recomendação de trading.

Considere:
1. Confluência de sinais (múltiplos indicadores apontando mesma direção)
2. Força dos sinais
3. Proximidade de níveis importantes
4. Contexto de mercado

RESPONDA EM JSON (sem markdown, apenas JSON puro):
{{
  "technical_signal": "BUY|SELL|HOLD",
  "confidence": 0.XX,
  "signal_strength": "strong|moderate|weak",
  "reasoning": "Explicação detalhada em 2-3 frases",
  "key_indicators": ["Indicador 1", "Indicador 2", "Indicador 3"],
  "confluences": ["Confluência 1", "Confluência 2"],
  "divergences": ["Divergência 1", "Divergência 2"],
  "entry_zone": {{"min": X.XX, "max": X.XX}},
  "target_levels": [X.XX, X.XX],
  "stop_loss_level": X.XX,
  "risk_assessment": "low|medium|high"
}}
"""
        return prompt

    def _parse_response(self, response: str) -> dict:
        """Parseia resposta do LLM"""
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]

            return json.loads(response)

        except Exception as e:
            logger.error(f"Agent 3: Erro ao parsear resposta: {e}")
            return {
                "technical_signal": "HOLD",
                "confidence": 0.5,
                "signal_strength": "weak",
                "reasoning": f"Erro ao processar análise: {e!s}",
                "key_indicators": [],
                "confluences": [],
                "divergences": [],
                "entry_zone": {"min": 0, "max": 0},
                "target_levels": [],
                "stop_loss_level": 0,
                "risk_assessment": "high",
            }

    def _get_default_response(self, error_msg: str) -> dict:
        """Retorna resposta padrão em caso de erro"""
        return {
            "technical_signal": "HOLD",
            "confidence": 0.0,
            "signal_strength": "weak",
            "reasoning": f"Erro no sistema: {error_msg}",
            "key_indicators": [],
            "confluences": [],
            "divergences": [],
            "entry_zone": {"min": 0, "max": 0},
            "target_levels": [],
            "stop_loss_level": 0,
            "risk_assessment": "high",
            "indicators": {},
            "signals": {},
            "levels": {},
            "agent": "agent_3_ta_strategist",
            "error": error_msg,
        }


# Exemplo de uso
if __name__ == "__main__":
    import asyncio

    async def test_agent():
        # Dados de teste
        data = {
            "open": [50000 + i * 50 for i in range(50)],
            "high": [50000 + i * 50 + 200 for i in range(50)],
            "low": [50000 + i * 50 - 100 for i in range(50)],
            "close": [50000 + i * 50 + 100 for i in range(50)],
            "volume": [1000000 + i * 50000 for i in range(50)],
        }
        df = pd.DataFrame(data)

        agent = TechnicalAnalystStrategist()
        result = await agent.analyze(df, "BTC/USD")

        print("\n📊 AGENT 3 - TECHNICAL ANALYST:")
        print(f"Sinal Técnico: {result['technical_signal']}")
        print(f"Confiança: {result['confidence']:.0%}")
        print(f"Força: {result['signal_strength']}")
        print(f"Justificativa: {result['reasoning']}")

    asyncio.run(test_agent())
