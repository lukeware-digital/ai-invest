"""
CeciAI - Agent 5: Investment Evaluator
Avaliador de oportunidades de investimento

Responsabilidades:
- Consolidar análises dos agentes anteriores (1-4)
- Avaliar qualidade da oportunidade de investimento
- Calcular score de oportunidade (0-100)
- Identificar riscos e recompensas
- Fornecer recomendação final consolidada

Autor: CeciAI Team
Data: 2025-10-08
"""

import json
import logging
from typing import Any

import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InvestmentEvaluator:
    """
    Agent 5: Avaliador de Oportunidades

    Consolida análises anteriores e avalia qualidade da oportunidade.
    """

    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        logger.info(f"Agent 5 (Investment Evaluator) inicializado com modelo {model}")

    async def evaluate(
        self,
        symbol: str,
        agent1_analysis: dict[str, Any],
        agent2_analysis: dict[str, Any],
        agent3_analysis: dict[str, Any],
        agent4_analysis: dict[str, Any],
        capital_available: float,
    ) -> dict[str, Any]:
        """
        Avalia oportunidade de investimento.

        Args:
            symbol: Par de trading
            agent1_analysis: Análise do Market Expert
            agent2_analysis: Análise do Data Analyzer
            agent3_analysis: Análise do Technical Analyst
            agent4_analysis: Análise do Candlestick Specialist
            capital_available: Capital disponível

        Returns:
            Dict com avaliação da oportunidade
        """
        try:
            # Calcular score consolidado
            opportunity_score = self._calculate_opportunity_score(
                agent1_analysis, agent2_analysis, agent3_analysis, agent4_analysis
            )

            # Preparar prompt
            prompt = self._build_prompt(
                symbol,
                agent1_analysis,
                agent2_analysis,
                agent3_analysis,
                agent4_analysis,
                opportunity_score,
                capital_available,
            )

            # Consultar LLM
            logger.info("Agent 5: Avaliando oportunidade de investimento...")
            response = ollama.generate(model=self.model, prompt=prompt, format="json")

            # Parsear resposta
            evaluation = self._parse_response(response["response"])

            # Adicionar dados calculados
            evaluation["opportunity_score"] = opportunity_score
            evaluation["agent"] = "agent_5_investment_evaluator"

            logger.info(f"Agent 5: Score de oportunidade: {opportunity_score}/100")

            return evaluation

        except Exception as e:
            logger.error(f"Agent 5: Erro na avaliação: {e}", exc_info=True)
            return self._get_default_response(str(e))

    def _calculate_opportunity_score(
        self, agent1: dict, agent2: dict, agent3: dict, agent4: dict
    ) -> int:
        """Calcula score de oportunidade (0-100)"""

        score = 0

        # Agent 1: Market Context (25 pontos)
        if agent1.get("market_regime") == "bull":
            score += 15
        elif agent1.get("market_regime") == "sideways":
            score += 8

        if agent1.get("sentiment") == "greed":
            score += 5
        elif agent1.get("sentiment") == "neutral":
            score += 3

        if agent1.get("market_quality") == "high":
            score += 5
        elif agent1.get("market_quality") == "medium":
            score += 3

        # Agent 2: Data Quality (15 pontos)
        if agent2.get("overall_quality") == "excellent":
            score += 15
        elif agent2.get("overall_quality") == "good":
            score += 10
        elif agent2.get("overall_quality") == "fair":
            score += 5

        # Agent 3: Technical Analysis (30 pontos)
        if agent3.get("technical_signal") == "BUY":
            score += 15
        elif agent3.get("technical_signal") == "HOLD":
            score += 5

        confidence = agent3.get("confidence", 0.5)
        score += int(confidence * 15)

        # Agent 4: Candlestick Patterns (30 pontos)
        if agent4.get("signal") == "BUY":
            score += 15
        elif agent4.get("signal") == "HOLD":
            score += 5

        confidence = agent4.get("confidence", 0.5)
        score += int(confidence * 15)

        return min(100, max(0, score))

    def _build_prompt(
        self,
        symbol: str,
        agent1: dict,
        agent2: dict,
        agent3: dict,
        agent4: dict,
        score: int,
        capital: float,
    ) -> str:
        """Constrói prompt para o LLM"""

        prompt = f"""
Você é um avaliador de investimentos profissional.

OPORTUNIDADE: {symbol}
CAPITAL DISPONÍVEL: ${capital:,.2f}

ANÁLISES DOS AGENTES:

1. MARKET EXPERT:
   - Regime: {agent1.get('market_regime', 'N/A')}
   - Sentimento: {agent1.get('sentiment', 'N/A')}
   - Qualidade: {agent1.get('market_quality', 'N/A')}
   - Recomendação: {agent1.get('recommendation', 'N/A')}

2. DATA ANALYZER:
   - Qualidade: {agent2.get('overall_quality', 'N/A')}
   - Confiabilidade: {agent2.get('reliability', 'N/A')}
   - Uso: {agent2.get('recommended_use', 'N/A')}

3. TECHNICAL ANALYST:
   - Sinal: {agent3.get('technical_signal', 'N/A')}
   - Confiança: {agent3.get('confidence', 0):.0%}
   - Força: {agent3.get('signal_strength', 'N/A')}

4. CANDLESTICK SPECIALIST:
   - Sinal: {agent4.get('signal', 'N/A')}
   - Confiança: {agent4.get('confidence', 0):.0%}
   - Padrões: {', '.join(agent4.get('key_patterns', []))}

SCORE CALCULADO: {score}/100

TAREFA:
Avalie esta oportunidade de investimento e forneça uma recomendação consolidada.

RESPONDA EM JSON (sem markdown):
{{
  "recommendation": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
  "confidence": 0.XX,
  "quality": "excellent|good|fair|poor",
  "reasoning": "Justificativa em 2-3 frases",
  "strengths": ["Força 1", "Força 2"],
  "weaknesses": ["Fraqueza 1", "Fraqueza 2"],
  "risk_level": "low|medium|high",
  "reward_potential": "low|medium|high",
  "suggested_allocation": 0.XX,
  "timeframe": "short|medium|long"
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
            logger.error(f"Agent 5: Erro ao parsear: {e}")
            return {
                "recommendation": "HOLD",
                "confidence": 0.5,
                "quality": "fair",
                "reasoning": f"Erro: {e!s}",
                "strengths": [],
                "weaknesses": ["Análise indisponível"],
                "risk_level": "high",
                "reward_potential": "unknown",
                "suggested_allocation": 0.0,
                "timeframe": "medium",
            }

    def _get_default_response(self, error_msg: str) -> dict:
        """Retorna resposta padrão em caso de erro"""
        return {
            "recommendation": "HOLD",
            "confidence": 0.0,
            "quality": "poor",
            "reasoning": f"Erro: {error_msg}",
            "strengths": [],
            "weaknesses": ["Sistema indisponível"],
            "risk_level": "high",
            "reward_potential": "unknown",
            "suggested_allocation": 0.0,
            "timeframe": "unknown",
            "opportunity_score": 0,
            "agent": "agent_5_investment_evaluator",
            "error": error_msg,
        }
