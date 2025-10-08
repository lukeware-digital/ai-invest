"""
CeciAI - Agent 8: Day-Trade Executor
Executor de operações de day-trade

Responsabilidades:
- Definir momento EXATO de entrada/saída
- Calcular preços (entry, stop-loss, take-profit)
- Validar capital e risco
- Gerar plano de execução completo

Autor: CeciAI Team
Data: 2025-10-07
"""

import json
import logging
from typing import Any

import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DayTradeExecutor:
    """
    Agent 8: Executor de Day-Trade

    Define momento exato de compra/venda e plano de execução.
    """

    def __init__(self, model: str = "llama3.2:3b"):
        """
        Inicializa o agente.

        Args:
            model: Modelo Ollama a usar
        """
        self.model = model
        logger.info(f"Agent 8 inicializado com modelo {model}")

    async def execute(
        self,
        symbol: str,
        current_price: float,
        agent_analyses: dict[str, Any],
        capital_available: float,
        risk_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Define plano de execução de trade.

        Args:
            symbol: Par de trading
            current_price: Preço atual
            agent_analyses: Análises dos agentes anteriores
            capital_available: Capital disponível
            risk_params: Parâmetros de risco

        Returns:
            Plano de execução completo
        """
        if risk_params is None:
            risk_params = {
                "max_risk_per_trade": 0.01,  # 1% do capital
                "min_risk_reward": 1.5,  # 1.5:1 mínimo
                "max_stop_loss": 0.03,  # 3% máximo
            }

        try:
            # Preparar prompt
            prompt = self._build_prompt(
                symbol, current_price, agent_analyses, capital_available, risk_params
            )

            # Consultar LLM
            response = ollama.generate(model=self.model, prompt=prompt, format="json")

            # Parsear e validar resposta
            execution_plan = self._parse_and_validate(
                response["response"], current_price, capital_available, risk_params
            )

            execution_plan["agent"] = "agent_8_daytrade"

            logger.info(
                f"Plano de execução: {execution_plan['decision']} @ ${execution_plan.get('entry_price', 0):.2f}"
            )

            return execution_plan

        except Exception as e:
            logger.error(f"Erro na execução: {e}", exc_info=True)
            return self._get_default_plan(str(e))

    def _build_prompt(
        self,
        symbol: str,
        current_price: float,
        analyses: dict[str, Any],
        capital: float,
        risk_params: dict[str, Any],
    ) -> str:
        """Constrói prompt para o LLM"""

        # Extrair informações dos agentes
        agent4_signal = analyses.get("agent4", {}).get("signal", "HOLD")
        agent4_confidence = analyses.get("agent4", {}).get("confidence", 0.5)
        agent4_patterns = analyses.get("agent4", {}).get("key_patterns", [])

        prompt = f"""
Você é um executor de day-trade profissional com 15 anos de experiência.

DADOS DO TRADE:
- Símbolo: {symbol}
- Preço Atual: ${current_price:.2f}
- Capital Disponível: ${capital:.2f}

ANÁLISES DOS AGENTES:
- Agent 4 (Candlestick): {agent4_signal} (confiança: {agent4_confidence:.0%})
- Padrões Detectados: {', '.join(agent4_patterns) if agent4_patterns else 'Nenhum'}

PARÂMETROS DE RISCO:
- Risco Máximo por Trade: {risk_params.get('max_risk_per_trade', 0.01) * 100:.1f}%
- Risk/Reward Mínimo: {risk_params.get('min_risk_reward', 1.5):.1f}:1
- Stop Loss Máximo: {risk_params.get('max_stop_loss', 0.03) * 100:.1f}%

TAREFA:
Defina um plano de execução completo e preciso para este trade.

REGRAS OBRIGATÓRIAS:
1. Se sinal for BUY: stop_loss < entry_price < take_profit
2. Se sinal for SELL: take_profit < entry_price < stop_loss
3. Respeitar TODOS os limites de risco
4. Calcular quantidade baseada no capital disponível
5. Risk/Reward ratio deve ser >= {risk_params.get('min_risk_reward', 1.5)}

CÁLCULOS NECESSÁRIOS:
- Entry Price: Preço de entrada (pode ser market = current_price ou limit)
- Quantity: Quanto comprar em USD (máximo 50% do capital)
- Stop Loss: Preço de stop (máximo {risk_params.get('max_stop_loss', 0.03) * 100}% de distância)
- Take Profit 1: Primeiro alvo (1.5x o risco)
- Take Profit 2: Segundo alvo (2.5x o risco)

RESPONDA EM JSON (sem markdown, apenas JSON puro):
{{
  "decision": "BUY|SELL|HOLD",
  "entry_price": {current_price},
  "entry_type": "MARKET|LIMIT",
  "quantity_usd": 0.00,
  "stop_loss": {{
    "price": 0.00,
    "percent": 0.00
  }},
  "take_profit_1": {{
    "price": 0.00,
    "percent": 0.00
  }},
  "take_profit_2": {{
    "price": 0.00,
    "percent": 0.00
  }},
  "risk_amount_usd": 0.00,
  "potential_profit_usd": 0.00,
  "risk_reward_ratio": 0.00,
  "max_hold_time": "4h",
  "confidence": 0.XX,
  "reasoning": "Justificativa detalhada em 2-3 frases",
  "validations": {{
    "capital_check": "PASS",
    "risk_check": "PASS",
    "rr_ratio_check": "PASS"
  }}
}}
"""
        return prompt

    def _parse_and_validate(
        self, response: str, current_price: float, capital: float, risk_params: dict[str, Any]
    ) -> dict[str, Any]:
        """Parseia e valida resposta do LLM"""

        try:
            # Remover markdown se presente
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]

            plan = json.loads(response)

            # Validações de segurança
            validations = plan.get("validations", {})

            # 1. Validar capital
            quantity_usd = plan.get("quantity_usd", 0)
            if quantity_usd > capital * 0.5:  # Máximo 50% do capital
                plan["decision"] = "HOLD"
                validations["capital_check"] = "FAIL"
                plan["reasoning"] = "Quantidade excede 50% do capital disponível"
            else:
                validations["capital_check"] = "PASS"

            # 2. Validar risco
            risk_amount = plan.get("risk_amount_usd", 0)
            max_risk = capital * risk_params.get("max_risk_per_trade", 0.01)
            if risk_amount > max_risk:
                plan["decision"] = "HOLD"
                validations["risk_check"] = "FAIL"
                plan["reasoning"] = f"Risco de ${risk_amount:.2f} excede máximo de ${max_risk:.2f}"
            else:
                validations["risk_check"] = "PASS"

            # 3. Validar risk/reward ratio
            rr_ratio = plan.get("risk_reward_ratio", 0)
            min_rr = risk_params.get("min_risk_reward", 1.5)
            if rr_ratio < min_rr:
                plan["decision"] = "HOLD"
                validations["rr_ratio_check"] = "FAIL"
                plan["reasoning"] = f"Risk/Reward de {rr_ratio:.2f} abaixo do mínimo {min_rr:.2f}"
            else:
                validations["rr_ratio_check"] = "PASS"

            # 4. Validar preços (lógica básica)
            if plan["decision"] == "BUY":
                entry = plan.get("entry_price", current_price)
                sl = plan.get("stop_loss", {}).get("price", 0)
                tp1 = plan.get("take_profit_1", {}).get("price", 0)

                if not (sl < entry < tp1):
                    plan["decision"] = "HOLD"
                    validations["price_logic_check"] = "FAIL"
                    plan["reasoning"] = "Lógica de preços inválida para BUY"

            plan["validations"] = validations

            return plan

        except Exception as e:
            logger.error(f"Erro ao parsear/validar: {e}")
            return {
                "decision": "HOLD",
                "entry_price": current_price,
                "entry_type": "MARKET",
                "quantity_usd": 0,
                "stop_loss": {"price": 0, "percent": 0},
                "take_profit_1": {"price": 0, "percent": 0},
                "take_profit_2": {"price": 0, "percent": 0},
                "risk_amount_usd": 0,
                "potential_profit_usd": 0,
                "risk_reward_ratio": 0,
                "max_hold_time": "0h",
                "confidence": 0.0,
                "reasoning": f"Erro ao processar plano: {e!s}",
                "validations": {
                    "capital_check": "FAIL",
                    "risk_check": "FAIL",
                    "rr_ratio_check": "FAIL",
                },
            }

    def _get_default_plan(self, error_msg: str) -> dict[str, Any]:
        """Retorna plano padrão em caso de erro"""
        return {
            "decision": "HOLD",
            "entry_price": 0,
            "entry_type": "MARKET",
            "quantity_usd": 0,
            "stop_loss": {"price": 0, "percent": 0},
            "take_profit_1": {"price": 0, "percent": 0},
            "take_profit_2": {"price": 0, "percent": 0},
            "risk_amount_usd": 0,
            "potential_profit_usd": 0,
            "risk_reward_ratio": 0,
            "max_hold_time": "0h",
            "confidence": 0.0,
            "reasoning": f"Erro no sistema: {error_msg}",
            "validations": {
                "capital_check": "FAIL",
                "risk_check": "FAIL",
                "rr_ratio_check": "FAIL",
            },
            "agent": "agent_8_daytrade",
            "error": error_msg,
        }


# Exemplo de uso
if __name__ == "__main__":
    import asyncio

    async def test_agent():
        agent = DayTradeExecutor()

        # Simular análise do Agent 4
        agent4_analysis = {
            "signal": "BUY",
            "confidence": 0.75,
            "key_patterns": ["Bullish Engulfing", "Hammer"],
        }

        result = await agent.execute(
            symbol="BTC/USD",
            current_price=50000.00,
            agent_analyses={"agent4": agent4_analysis},
            capital_available=10000.00,
        )

        print("\n💼 PLANO DE EXECUÇÃO:")
        print(json.dumps(result, indent=2))

    asyncio.run(test_agent())
