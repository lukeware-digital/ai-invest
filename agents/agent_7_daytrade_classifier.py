"""
CeciAI - Agent 7: Day-Trade Classifier
Classificador de tipo de trade

Responsabilidades:
- Classificar trade como day-trade ou long-term
- Rotear para Agent 8 (day-trade) ou Agent 9 (long-term)
- Avaliar adequação do tipo de trade

Autor: CeciAI Team
Data: 2025-10-08
"""

import json
import logging
from typing import Any

import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DayTradeClassifier:
    """Agent 7: Classificador de Tipo de Trade"""

    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        logger.info("Agent 7 (Day-Trade Classifier) inicializado")

    async def classify(
        self, symbol: str, investment_evaluation: dict[str, Any], time_horizon: dict[str, Any]
    ) -> dict[str, Any]:
        """Classifica tipo de trade"""
        try:
            prompt = f"""
Você é um classificador de trades.

SÍMBOLO: {symbol}

AVALIAÇÃO:
- Score: {investment_evaluation.get('opportunity_score', 0)}/100
- Risco: {investment_evaluation.get('risk_level', 'N/A')}

HORIZONTE:
- Estratégia: {time_horizon.get('recommended_strategy', 'N/A')}
- Duração: {time_horizon.get('expected_duration', 'N/A')}

TAREFA: Classifique este trade.

RESPONDA EM JSON:
{{
  "trade_type": "day_trade|long_term",
  "confidence": 0.XX,
  "reasoning": "Justificativa",
  "recommended_executor": "agent_8|agent_9"
}}
"""

            response = ollama.generate(model=self.model, prompt=prompt, format="json")
            classification = json.loads(
                response["response"].strip().replace("```json", "").replace("```", "")
            )

            # Garantir executor correto
            if classification.get("trade_type") == "day_trade":
                classification["recommended_executor"] = "agent_8"
            else:
                classification["recommended_executor"] = "agent_9"

            classification["agent"] = "agent_7_daytrade_classifier"

            logger.info(f"Agent 7: Tipo de trade: {classification.get('trade_type')}")
            return classification

        except Exception as e:
            logger.error(f"Agent 7: Erro: {e}")
            return {
                "trade_type": "day_trade",
                "confidence": 0.5,
                "reasoning": f"Erro: {e!s}",
                "recommended_executor": "agent_8",
                "agent": "agent_7_daytrade_classifier",
                "error": str(e),
            }
