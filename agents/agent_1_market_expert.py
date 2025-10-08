"""
CeciAI - Agent 1: Market Expert
Especialista em análise de contexto de mercado

Responsabilidades:
- Analisar contexto geral do mercado
- Identificar regime de mercado (bull/bear/lateral)
- Avaliar sentimento de mercado
- Analisar volume e liquidez
- Identificar condições macro

Autor: CeciAI Team
Data: 2025-10-08
"""

import json
import logging
from typing import Any

import ollama
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketExpert:
    """
    Agent 1: Especialista em Contexto de Mercado

    Primeiro agente do pipeline. Analisa o contexto geral do mercado
    e fornece insights sobre regime, sentimento e condições macro.
    """

    def __init__(self, model: str = "llama3.2:3b"):
        """
        Inicializa o agente.

        Args:
            model: Modelo Ollama a usar
        """
        self.model = model
        logger.info(f"Agent 1 (Market Expert) inicializado com modelo {model}")

    async def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Analisa o contexto de mercado.

        Args:
            df: DataFrame com dados OHLCV
            symbol: Par de trading
            timeframe: Timeframe dos dados
            context: Contexto adicional

        Returns:
            Dict com análise de mercado
        """
        if context is None:
            context = {}

        try:
            # Calcular métricas de mercado
            market_metrics = self._calculate_market_metrics(df)

            # Preparar prompt para LLM
            prompt = self._build_prompt(df, symbol, timeframe, market_metrics, context)

            # Consultar LLM
            logger.info("Agent 1: Consultando LLM para análise de mercado...")
            response = ollama.generate(model=self.model, prompt=prompt, format="json")

            # Parsear resposta
            analysis = self._parse_response(response["response"])

            # Adicionar métricas calculadas
            analysis["market_metrics"] = market_metrics
            analysis["agent"] = "agent_1_market_expert"

            logger.info(f"Agent 1: Regime identificado: {analysis.get('market_regime', 'N/A')}")

            return analysis

        except Exception as e:
            logger.error(f"Agent 1: Erro na análise: {e}", exc_info=True)
            return self._get_default_response(str(e))

    def _calculate_market_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calcula métricas de mercado"""

        # Preços
        current_price = float(df["close"].iloc[-1])
        price_change_1h = (
            float(((df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2]) * 100)
            if len(df) >= 2
            else 0
        )
        price_change_24h = (
            float(((df["close"].iloc[-1] - df["close"].iloc[-24]) / df["close"].iloc[-24]) * 100)
            if len(df) >= 24
            else 0
        )

        # Volatilidade
        returns = df["close"].pct_change()
        volatility = float(returns.std() * 100)

        # Volume
        avg_volume = float(df["volume"].mean())
        current_volume = float(df["volume"].iloc[-1])
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # Tendência
        sma_short = df["close"].rolling(window=min(7, len(df))).mean().iloc[-1]
        sma_long = df["close"].rolling(window=min(20, len(df))).mean().iloc[-1]

        if sma_short > sma_long * 1.02:
            trend = "uptrend"
            trend_strength = "strong" if sma_short > sma_long * 1.05 else "moderate"
        elif sma_short < sma_long * 0.98:
            trend = "downtrend"
            trend_strength = "strong" if sma_short < sma_long * 0.95 else "moderate"
        else:
            trend = "sideways"
            trend_strength = "weak"

        # Range
        high_24h = float(df["high"].tail(24).max()) if len(df) >= 24 else float(df["high"].max())
        low_24h = float(df["low"].tail(24).min()) if len(df) >= 24 else float(df["low"].min())
        range_pct = ((high_24h - low_24h) / low_24h) * 100

        return {
            "current_price": current_price,
            "price_change_1h": price_change_1h,
            "price_change_24h": price_change_24h,
            "volatility": volatility,
            "avg_volume": avg_volume,
            "current_volume": current_volume,
            "volume_ratio": volume_ratio,
            "trend": trend,
            "trend_strength": trend_strength,
            "high_24h": high_24h,
            "low_24h": low_24h,
            "range_24h_pct": range_pct,
        }

    def _build_prompt(
        self, df: pd.DataFrame, symbol: str, timeframe: str, metrics: dict, context: dict
    ) -> str:
        """Constrói prompt para o LLM"""

        prompt = f"""
Você é um especialista em análise de mercado de criptomoedas com 20 anos de experiência.

DADOS DO MERCADO:
- Símbolo: {symbol}
- Timeframe: {timeframe}
- Preço Atual: ${metrics['current_price']:,.2f}
- Variação 1h: {metrics['price_change_1h']:.2f}%
- Variação 24h: {metrics['price_change_24h']:.2f}%

MÉTRICAS DE MERCADO:
- Volatilidade: {metrics['volatility']:.2f}%
- Volume Atual: {metrics['current_volume']:,.0f}
- Volume Médio: {metrics['avg_volume']:,.0f}
- Ratio Volume: {metrics['volume_ratio']:.2f}x
- Tendência: {metrics['trend']} ({metrics['trend_strength']})
- Range 24h: {metrics['range_24h_pct']:.2f}%
- High 24h: ${metrics['high_24h']:,.2f}
- Low 24h: ${metrics['low_24h']:,.2f}

TAREFA:
Analise o contexto geral do mercado e forneça insights sobre:
1. Regime de mercado (bull/bear/sideways)
2. Sentimento dominante (fear/greed/neutral)
3. Qualidade do mercado (high/medium/low)
4. Condições de liquidez
5. Recomendação geral

Considere:
- A volatilidade atual vs histórica
- O volume de negociação
- A força da tendência
- O range de preços

RESPONDA EM JSON (sem markdown, apenas JSON puro):
{{
  "market_regime": "bull|bear|sideways",
  "regime_confidence": 0.XX,
  "sentiment": "fear|greed|neutral",
  "sentiment_score": 0-100,
  "market_quality": "high|medium|low",
  "liquidity": "high|medium|low",
  "volatility_assessment": "high|normal|low",
  "recommendation": "favorable|neutral|unfavorable",
  "reasoning": "Explicação detalhada em 2-3 frases",
  "key_observations": ["Observação 1", "Observação 2", "Observação 3"],
  "risk_factors": ["Fator de risco 1", "Fator de risco 2"],
  "opportunities": ["Oportunidade 1", "Oportunidade 2"]
}}
"""
        return prompt

    def _parse_response(self, response: str) -> dict:
        """Parseia resposta do LLM"""
        try:
            # Remover markdown se presente
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]

            return json.loads(response)

        except Exception as e:
            logger.error(f"Agent 1: Erro ao parsear resposta: {e}")
            return {
                "market_regime": "sideways",
                "regime_confidence": 0.5,
                "sentiment": "neutral",
                "sentiment_score": 50,
                "market_quality": "medium",
                "liquidity": "medium",
                "volatility_assessment": "normal",
                "recommendation": "neutral",
                "reasoning": f"Erro ao processar análise: {e!s}",
                "key_observations": [],
                "risk_factors": ["Análise indisponível"],
                "opportunities": [],
            }

    def _get_default_response(self, error_msg: str) -> dict:
        """Retorna resposta padrão em caso de erro"""
        return {
            "market_regime": "sideways",
            "regime_confidence": 0.0,
            "sentiment": "neutral",
            "sentiment_score": 50,
            "market_quality": "low",
            "liquidity": "unknown",
            "volatility_assessment": "unknown",
            "recommendation": "unfavorable",
            "reasoning": f"Erro no sistema: {error_msg}",
            "key_observations": [],
            "risk_factors": ["Sistema indisponível"],
            "opportunities": [],
            "market_metrics": {},
            "agent": "agent_1_market_expert",
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

        agent = MarketExpert()
        result = await agent.analyze(df, "BTC/USD", "1h")

        print("\n📊 AGENT 1 - MARKET EXPERT:")
        print(f"Regime: {result['market_regime']} (confiança: {result['regime_confidence']:.0%})")
        print(f"Sentimento: {result['sentiment']} (score: {result['sentiment_score']})")
        print(f"Qualidade: {result['market_quality']}")
        print(f"Recomendação: {result['recommendation']}")
        print(f"Justificativa: {result['reasoning']}")

    asyncio.run(test_agent())
