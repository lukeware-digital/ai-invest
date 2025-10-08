"""
CeciAI - Agent 4: Candlestick Specialist
Especialista em análise de padrões de candlestick

Responsabilidades:
- Detectar padrões de candlestick
- Avaliar força e confiabilidade dos padrões
- Fornecer sinais de trading baseados em padrões
- Integrar com LLM para análise contextual

Autor: CeciAI Team
Data: 2025-10-07
"""

import json
import logging
from typing import Any

import ollama
import pandas as pd

from utils.candlestick_patterns import CandlestickPatternDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CandlestickSpecialist:
    """
    Agent 4: Especialista em Padrões de Candlestick

    Analisa padrões de candles e fornece sinais de trading.
    """

    def __init__(self, model: str = "llama3.2:3b"):
        """
        Inicializa o agente.

        Args:
            model: Modelo Ollama a usar
        """
        self.model = model
        self.detector = CandlestickPatternDetector()
        logger.info(f"Agent 4 inicializado com modelo {model}")

    async def analyze(
        self, df: pd.DataFrame, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Analisa padrões de candlestick.

        Args:
            df: DataFrame com dados OHLCV
            context: Contexto adicional (tendência, volume, etc)

        Returns:
            Dict com análise e recomendação
        """
        if context is None:
            context = {}

        try:
            # Detectar padrões
            patterns = self.detector.detect_all_patterns(df)
            strength = self.detector.calculate_pattern_strength(patterns)
            summary = self.detector.generate_summary(patterns)

            logger.info(f"Detectados {len(patterns)} padrões de candlestick")

            # Preparar prompt para LLM
            prompt = self._build_prompt(df, patterns, strength, context)

            # Consultar LLM
            response = ollama.generate(model=self.model, prompt=prompt, format="json")

            # Parsear resposta
            analysis = self._parse_response(response["response"])

            # Adicionar dados técnicos
            analysis["patterns_detected"] = len(patterns)
            analysis["pattern_strength"] = strength
            analysis["pattern_summary"] = summary
            analysis["agent"] = "agent_4_candlestick"

            logger.info(
                f"Análise completa: {analysis['signal']} (confiança: {analysis['confidence']:.2f})"
            )

            return analysis

        except Exception as e:
            logger.error(f"Erro na análise: {e}", exc_info=True)
            return self._get_default_response(str(e))

    def _build_prompt(self, df: pd.DataFrame, patterns: list, strength: dict, context: dict) -> str:
        """Constrói prompt para o LLM"""

        latest_candle = df.iloc[-1]

        prompt = f"""
Você é um especialista em análise de padrões de candlestick com 20 anos de experiência em trading.

DADOS DO MERCADO:
- Símbolo: {context.get('symbol', 'N/A')}
- Preço Atual: ${latest_candle['close']:.2f}
- Tendência: {context.get('trend', 'N/A')}
- Volume: {latest_candle.get('volume', 'N/A')}
- Variação 24h: {((latest_candle['close'] - df.iloc[-24]['close']) / df.iloc[-24]['close'] * 100) if len(df) >= 24 else 0:.2f}%

PADRÕES DETECTADOS:
{self.detector.generate_summary(patterns)}

FORÇA DOS PADRÕES:
- Score Bullish: {strength['bullish_score']:.2f}
- Score Bearish: {strength['bearish_score']:.2f}
- Confiança: {strength['confidence']}
- Sinal Dominante: {strength['dominant_signal']}

TAREFA:
Analise os padrões de candlestick detectados e forneça uma recomendação de trading precisa.

Considere:
1. A força e confiabilidade dos padrões
2. O contexto de mercado (tendência, volume)
3. Se os padrões confirmam ou contradizem a tendência
4. Necessidade de confirmação
5. Histórico de acurácia de cada padrão

RESPONDA EM JSON (sem markdown, apenas JSON puro):
{{
  "signal": "BUY|SELL|HOLD",
  "confidence": 0.XX,
  "reasoning": "Explicação detalhada da análise em 2-3 frases",
  "key_patterns": ["Lista dos 3 padrões mais importantes"],
  "confirmation_needed": true|false,
  "confirmation_criteria": "O que observar para confirmar",
  "risk_level": "LOW|MEDIUM|HIGH",
  "timeframe_recommendation": "SCALPING|SWING|HOLD"
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
            logger.error(f"Erro ao parsear resposta: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0.5,
                "reasoning": f"Erro ao parsear resposta do LLM: {e!s}",
                "key_patterns": [],
                "confirmation_needed": True,
                "confirmation_criteria": "Aguardar próximo candle",
                "risk_level": "HIGH",
                "timeframe_recommendation": "HOLD",
            }

    def _get_default_response(self, error_msg: str) -> dict:
        """Retorna resposta padrão em caso de erro"""
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "reasoning": f"Erro na análise: {error_msg}",
            "key_patterns": [],
            "confirmation_needed": True,
            "confirmation_criteria": "Sistema indisponível",
            "risk_level": "HIGH",
            "timeframe_recommendation": "HOLD",
            "patterns_detected": 0,
            "pattern_strength": {
                "bullish_score": 0,
                "bearish_score": 0,
                "confidence": "LOW",
                "dominant_signal": "HOLD",
            },
            "pattern_summary": "Erro na detecção de padrões",
            "agent": "agent_4_candlestick",
            "error": error_msg,
        }


# Exemplo de uso
if __name__ == "__main__":
    import asyncio

    async def test_agent():
        # Dados de teste
        data = {
            "open": [100, 102, 101, 103, 102],
            "high": [105, 106, 104, 107, 106],
            "low": [99, 101, 100, 102, 101],
            "close": [103, 101, 103, 102, 105],
            "volume": [1000, 1100, 950, 1200, 1050],
        }
        df = pd.DataFrame(data)

        agent = CandlestickSpecialist()
        result = await agent.analyze(df, {"symbol": "BTC/USD", "trend": "uptrend"})

        print("\n📊 RESULTADO DA ANÁLISE:")
        print(json.dumps(result, indent=2))

    asyncio.run(test_agent())
