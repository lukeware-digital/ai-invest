"""
CeciAI - Agent 6: Time Horizon Advisor
Conselheiro de horizonte temporal

Responsabilidades:
- Recomendar estratégia (scalping, swing, arbitrage)
- Definir timeframe ideal
- Avaliar adequação ao perfil de risco
- Sugerir duração do trade

Autor: CeciAI Team
Data: 2025-10-08
"""

import json
import logging
from typing import Any

import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeHorizonAdvisor:
    """Agent 6: Conselheiro de Horizonte Temporal"""

    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        logger.info("Agent 6 (Time Horizon Advisor) inicializado")

    async def advise(
        self,
        symbol: str,
        investment_evaluation: dict[str, Any],
        market_context: dict[str, Any],
        user_strategy: str = "scalping",
    ) -> dict[str, Any]:
        """Recomenda horizonte temporal e estratégia"""
        try:
            prompt = f"""
Você é um consultor de estratégias de trading.

SÍMBOLO: {symbol}
ESTRATÉGIA PREFERIDA: {user_strategy}

AVALIAÇÃO:
- Score: {investment_evaluation.get('opportunity_score', 0)}/100
- Recomendação: {investment_evaluation.get('recommendation', 'N/A')}
- Risco: {investment_evaluation.get('risk_level', 'N/A')}
- Timeframe Sugerido: {investment_evaluation.get('timeframe', 'N/A')}

MERCADO:
- Regime: {market_context.get('market_regime', 'N/A')}
- Volatilidade: {market_context.get('market_metrics', {}).get('volatility', 'N/A')}

TAREFA: Recomende a melhor estratégia e horizonte temporal.

RESPONDA EM JSON:
{{
  "recommended_strategy": "scalping|swing|arbitrage",
  "confidence": 0.XX,
  "timeframe": "1min|5min|15min|1h|4h|1d",
  "expected_duration": "minutes|hours|days",
  "reasoning": "Justificativa",
  "alignment_score": 0-100
}}
"""

            response = ollama.generate(model=self.model, prompt=prompt, format="json")
            analysis = json.loads(
                response["response"].strip().replace("```json", "").replace("```", "")
            )
            analysis["agent"] = "agent_6_time_horizon"

            logger.info(f"Agent 6: Estratégia recomendada: {analysis.get('recommended_strategy')}")
            return analysis

        except Exception as e:
            logger.error(f"Agent 6: Erro: {e}")
            return {
                "recommended_strategy": user_strategy,
                "confidence": 0.5,
                "timeframe": "1h",
                "expected_duration": "hours",
                "reasoning": f"Erro: {e!s}",
                "alignment_score": 50,
                "agent": "agent_6_time_horizon",
                "error": str(e),
            }
