"""
CeciAI - Agent 9: Long-Term Executor
Executor de trades de longo prazo

Responsabilidades:
- Definir plano de acumulação
- Calcular preços para long-term
- Estratégia de DCA (Dollar Cost Averaging)
- Plano de saída gradual

Autor: CeciAI Team
Data: 2025-10-08
"""

import json
import logging
from typing import Any

import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LongTermExecutor:
    """Agent 9: Executor de Long-Term"""

    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        logger.info("Agent 9 (Long-Term Executor) inicializado")

    async def execute(
        self,
        symbol: str,
        current_price: float,
        investment_evaluation: dict[str, Any],
        capital_available: float,
    ) -> dict[str, Any]:
        """Define plano de execução long-term"""
        try:
            prompt = f"""
Você é um executor de investimentos de longo prazo.

SÍMBOLO: {symbol}
PREÇO ATUAL: ${current_price:,.2f}
CAPITAL: ${capital_available:,.2f}

AVALIAÇÃO:
- Score: {investment_evaluation.get('opportunity_score', 0)}/100
- Recomendação: {investment_evaluation.get('recommendation', 'N/A')}
- Alocação Sugerida: {investment_evaluation.get('suggested_allocation', 0):.0%}

TAREFA: Crie um plano de investimento de longo prazo.

RESPONDA EM JSON:
{{
  "decision": "ACCUMULATE|HOLD|REDUCE",
  "accumulation_plan": {{
    "total_allocation_usd": X.XX,
    "entry_phases": 3,
    "phase_amount_usd": X.XX,
    "dca_frequency": "weekly|monthly"
  }},
  "target_prices": {{
    "phase_1": X.XX,
    "phase_2": X.XX,
    "phase_3": X.XX
  }},
  "exit_strategy": {{
    "target_1": {{"price": X.XX, "allocation": 0.33}},
    "target_2": {{"price": X.XX, "allocation": 0.33}},
    "target_3": {{"price": X.XX, "allocation": 0.34}}
  }},
  "stop_loss": X.XX,
  "holding_period": "months|years",
  "confidence": 0.XX,
  "reasoning": "Justificativa"
}}
"""

            response = ollama.generate(model=self.model, prompt=prompt, format="json")
            plan = json.loads(
                response["response"].strip().replace("```json", "").replace("```", "")
            )
            plan["agent"] = "agent_9_longterm"

            logger.info(f"Agent 9: Decisão: {plan.get('decision')}")
            return plan

        except Exception as e:
            logger.error(f"Agent 9: Erro: {e}")
            return {
                "decision": "HOLD",
                "accumulation_plan": {
                    "total_allocation_usd": 0,
                    "entry_phases": 0,
                    "phase_amount_usd": 0,
                    "dca_frequency": "monthly",
                },
                "target_prices": {},
                "exit_strategy": {},
                "stop_loss": 0,
                "holding_period": "unknown",
                "confidence": 0.0,
                "reasoning": f"Erro: {e!s}",
                "agent": "agent_9_longterm",
                "error": str(e),
            }
