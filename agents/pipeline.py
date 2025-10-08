"""
CeciAI - Pipeline Orchestrator
Orquestrador do pipeline de 9 agentes LLM

Responsabilidades:
- Orquestrar execução dos 9 agentes em sequência
- Executar agentes em paralelo quando possível
- Gerenciar fluxo de dados entre agentes
- Consolidar resultados finais
- Tratamento de erros e fallbacks

Autor: CeciAI Team
Data: 2025-10-08
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

import pandas as pd

# Importar todos os agentes
from agents.agent_1_market_expert import MarketExpert
from agents.agent_2_data_analyzer import DataAnalyzer
from agents.agent_3_ta_strategist import TechnicalAnalystStrategist
from agents.agent_4_candlestick_specialist import CandlestickSpecialist
from agents.agent_5_investment_evaluator import InvestmentEvaluator
from agents.agent_6_time_horizon_advisor import TimeHorizonAdvisor
from agents.agent_7_daytrade_classifier import DayTradeClassifier
from agents.agent_8_daytrade_executor import DayTradeExecutor
from agents.agent_9_longterm_executor import LongTermExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentPipeline:
    """
    Pipeline Orchestrator - Orquestra todos os 9 agentes LLM

    Fluxo:
    1. Agent 1 (Market Expert) - Análise de contexto
    2. Agent 2 (Data Analyzer) - Qualidade dos dados
    3. Agent 3 (Technical Analyst) + Agent 4 (Candlestick) - Paralelo
    4. Agent 5 (Investment Evaluator) - Consolida 1-4
    5. Agent 6 (Time Horizon) + Agent 7 (Classifier) - Paralelo
    6. Agent 8 (Day-Trade) OU Agent 9 (Long-Term) - Baseado em Agent 7
    """

    def __init__(self, model: str = "llama3.2:3b"):
        """
        Inicializa pipeline com todos os agentes.

        Args:
            model: Modelo Ollama a usar em todos os agentes
        """
        logger.info("🚀 Inicializando Agent Pipeline...")

        # Inicializar todos os agentes
        self.agent1 = MarketExpert(model)
        self.agent2 = DataAnalyzer(model)
        self.agent3 = TechnicalAnalystStrategist(model)
        self.agent4 = CandlestickSpecialist(model)
        self.agent5 = InvestmentEvaluator(model)
        self.agent6 = TimeHorizonAdvisor(model)
        self.agent7 = DayTradeClassifier(model)
        self.agent8 = DayTradeExecutor(model)
        self.agent9 = LongTermExecutor(model)

        logger.info("✅ Pipeline inicializado com 9 agentes")

    async def execute(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "1h",
        capital_available: float = 10000.0,
        user_strategy: str = "scalping",
        risk_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Executa pipeline completo de análise.

        Args:
            df: DataFrame com dados OHLCV
            symbol: Par de trading
            timeframe: Timeframe dos dados
            capital_available: Capital disponível
            user_strategy: Estratégia preferida do usuário
            risk_params: Parâmetros de risco

        Returns:
            Dict com resultado completo do pipeline
        """
        if risk_params is None:
            risk_params = {
                "max_risk_per_trade": 0.01,
                "min_risk_reward": 1.5,
                "max_stop_loss": 0.03,
            }

        start_time = datetime.now()
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 INICIANDO PIPELINE PARA {symbol}")
        logger.info(f"{'='*60}\n")

        try:
            # ========== FASE 1: ANÁLISE DE CONTEXTO ==========
            logger.info("📊 FASE 1: Análise de Contexto e Qualidade")

            # Agent 1: Market Expert
            logger.info("  → Agent 1: Market Expert")
            agent1_result = await self.agent1.analyze(df, symbol, timeframe)

            # Agent 2: Data Analyzer
            logger.info("  → Agent 2: Data Analyzer")
            agent2_result = await self.agent2.analyze(df, symbol, timeframe, agent1_result)

            # ========== FASE 2: ANÁLISE TÉCNICA (PARALELO) ==========
            logger.info("\n📈 FASE 2: Análise Técnica (Paralelo)")

            # Executar Agent 3 e Agent 4 em paralelo
            logger.info("  → Agent 3: Technical Analyst")
            logger.info("  → Agent 4: Candlestick Specialist")

            agent3_task = self.agent3.analyze(df, symbol, agent1_result, agent2_result)
            agent4_task = self.agent4.analyze(
                df,
                {
                    "symbol": symbol,
                    "trend": agent1_result.get("market_metrics", {}).get("trend", "neutral"),
                },
            )

            agent3_result, agent4_result = await asyncio.gather(agent3_task, agent4_task)

            # ========== FASE 3: AVALIAÇÃO DE OPORTUNIDADE ==========
            logger.info("\n💎 FASE 3: Avaliação de Oportunidade")
            logger.info("  → Agent 5: Investment Evaluator")

            agent5_result = await self.agent5.evaluate(
                symbol,
                agent1_result,
                agent2_result,
                agent3_result,
                agent4_result,
                capital_available,
            )

            # ========== FASE 4: ESTRATÉGIA E CLASSIFICAÇÃO (PARALELO) ==========
            logger.info("\n🎯 FASE 4: Estratégia e Classificação (Paralelo)")

            logger.info("  → Agent 6: Time Horizon Advisor")
            logger.info("  → Agent 7: Day-Trade Classifier")

            agent6_task = self.agent6.advise(symbol, agent5_result, agent1_result, user_strategy)
            agent7_task = self.agent7.classify(symbol, agent5_result, {})

            agent6_result, agent7_result = await asyncio.gather(agent6_task, agent7_task)

            # ========== FASE 5: EXECUÇÃO ==========
            logger.info("\n🚀 FASE 5: Plano de Execução")

            current_price = float(df["close"].iloc[-1])

            # Rotear para Agent 8 ou Agent 9 baseado na classificação
            if agent7_result.get("trade_type") == "day_trade":
                logger.info("  → Agent 8: Day-Trade Executor")

                executor_result = await self.agent8.execute(
                    symbol=symbol,
                    current_price=current_price,
                    agent_analyses={
                        "agent1": agent1_result,
                        "agent3": agent3_result,
                        "agent4": agent4_result,
                        "agent5": agent5_result,
                    },
                    capital_available=capital_available,
                    risk_params=risk_params,
                )

                executor_used = "agent_8"
            else:
                logger.info("  → Agent 9: Long-Term Executor")

                executor_result = await self.agent9.execute(
                    symbol=symbol,
                    current_price=current_price,
                    investment_evaluation=agent5_result,
                    capital_available=capital_available,
                )

                executor_used = "agent_9"

            # ========== CONSOLIDAR RESULTADOS ==========
            processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"\n{'='*60}")
            logger.info(f"✅ PIPELINE CONCLUÍDO EM {processing_time:.2f}s")
            logger.info(f"{'='*60}\n")

            # Montar resultado final
            final_result = {
                "pipeline_metadata": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat(),
                    "executor_used": executor_used,
                },
                "agent_1_market_expert": agent1_result,
                "agent_2_data_analyzer": agent2_result,
                "agent_3_technical_analyst": agent3_result,
                "agent_4_candlestick_specialist": agent4_result,
                "agent_5_investment_evaluator": agent5_result,
                "agent_6_time_horizon_advisor": agent6_result,
                "agent_7_daytrade_classifier": agent7_result,
                "executor_result": executor_result,
                "final_decision": self._consolidate_decision(
                    agent5_result, agent7_result, executor_result, executor_used
                ),
            }

            return final_result

        except Exception as e:
            logger.error(f"❌ Erro no pipeline: {e}", exc_info=True)
            return self._get_error_response(symbol, str(e))

    def _consolidate_decision(
        self, agent5: dict, agent7: dict, executor: dict, executor_used: str
    ) -> dict[str, Any]:
        """Consolida decisão final do pipeline"""

        # Extrair decisão do executor
        if executor_used == "agent_8":
            decision = executor.get("decision", "HOLD")
            confidence = executor.get("confidence", 0.5)
            entry_price = executor.get("entry_price", 0)
            stop_loss = executor.get("stop_loss", {}).get("price", 0)
            take_profit_1 = executor.get("take_profit_1", {}).get("price", 0)
            take_profit_2 = executor.get("take_profit_2", {}).get("price", 0)
            quantity_usd = executor.get("quantity_usd", 0)
            risk_reward = executor.get("risk_reward_ratio", 0)
        else:
            decision = executor.get("decision", "HOLD")
            confidence = executor.get("confidence", 0.5)
            entry_price = 0
            stop_loss = executor.get("stop_loss", 0)
            take_profit_1 = 0
            take_profit_2 = 0
            quantity_usd = executor.get("accumulation_plan", {}).get("total_allocation_usd", 0)
            risk_reward = 0

        return {
            "decision": decision,
            "confidence": confidence,
            "opportunity_score": agent5.get("opportunity_score", 0),
            "trade_type": agent7.get("trade_type", "unknown"),
            "recommended_strategy": agent7.get("recommended_executor", "unknown"),
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit_1": take_profit_1,
            "take_profit_2": take_profit_2,
            "quantity_usd": quantity_usd,
            "risk_reward_ratio": risk_reward,
            "reasoning": executor.get("reasoning", "N/A"),
            "validations": executor.get("validations", {}),
        }

    def _get_error_response(self, symbol: str, error_msg: str) -> dict[str, Any]:
        """Retorna resposta de erro"""
        return {
            "pipeline_metadata": {
                "symbol": symbol,
                "processing_time": 0,
                "timestamp": datetime.now().isoformat(),
                "error": error_msg,
            },
            "final_decision": {
                "decision": "HOLD",
                "confidence": 0.0,
                "opportunity_score": 0,
                "reasoning": f"Erro no pipeline: {error_msg}",
            },
        }

    async def cleanup(self):
        """Cleanup de recursos"""
        logger.info("🧹 Limpando recursos do pipeline...")
        # Adicionar cleanup se necessário


# Exemplo de uso
if __name__ == "__main__":

    async def test_pipeline():
        # Dados de teste
        data = {
            "open": [50000 + i * 50 for i in range(50)],
            "high": [50000 + i * 50 + 200 for i in range(50)],
            "low": [50000 + i * 50 - 100 for i in range(50)],
            "close": [50000 + i * 50 + 100 for i in range(50)],
            "volume": [1000000 + i * 50000 for i in range(50)],
        }
        df = pd.DataFrame(data)

        # Executar pipeline
        pipeline = AgentPipeline()
        result = await pipeline.execute(
            df=df,
            symbol="BTC/USD",
            timeframe="1h",
            capital_available=10000.0,
            user_strategy="scalping",
        )

        # Exibir resultado
        print("\n" + "=" * 60)
        print("📊 RESULTADO DO PIPELINE")
        print("=" * 60)
        print(f"\nDecisão Final: {result['final_decision']['decision']}")
        print(f"Confiança: {result['final_decision']['confidence']:.0%}")
        print(f"Score de Oportunidade: {result['final_decision']['opportunity_score']}/100")
        print(f"Tipo de Trade: {result['final_decision']['trade_type']}")
        print(f"Tempo de Processamento: {result['pipeline_metadata']['processing_time']:.2f}s")
        print(f"\nJustificativa: {result['final_decision']['reasoning']}")
        print("=" * 60)

    asyncio.run(test_pipeline())
