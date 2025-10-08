"""
CeciAI - Agent 2: Data Analyzer
Especialista em análise de qualidade de dados

Responsabilidades:
- Avaliar qualidade dos dados OHLCV
- Identificar gaps e anomalias
- Validar consistência dos dados
- Recomendar melhor timeframe para análise
- Detectar dados suspeitos ou manipulados

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


class DataAnalyzer:
    """
    Agent 2: Especialista em Qualidade de Dados

    Analisa a qualidade e consistência dos dados de mercado.
    """

    def __init__(self, model: str = "llama3.2:3b"):
        """
        Inicializa o agente.

        Args:
            model: Modelo Ollama a usar
        """
        self.model = model
        logger.info(f"Agent 2 (Data Analyzer) inicializado com modelo {model}")

    async def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        market_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Analisa qualidade dos dados.

        Args:
            df: DataFrame com dados OHLCV
            symbol: Par de trading
            timeframe: Timeframe dos dados
            market_context: Contexto do Agent 1

        Returns:
            Dict com análise de qualidade
        """
        if market_context is None:
            market_context = {}

        try:
            # Validar dados
            data_quality = self._validate_data_quality(df)

            # Detectar anomalias
            anomalies = self._detect_anomalies(df)

            # Avaliar completude
            completeness = self._assess_completeness(df)

            # Preparar prompt para LLM
            prompt = self._build_prompt(
                df, symbol, timeframe, data_quality, anomalies, completeness, market_context
            )

            # Consultar LLM
            logger.info("Agent 2: Consultando LLM para análise de dados...")
            response = ollama.generate(model=self.model, prompt=prompt, format="json")

            # Parsear resposta
            analysis = self._parse_response(response["response"])

            # Adicionar métricas calculadas
            analysis["data_quality"] = data_quality
            analysis["anomalies"] = anomalies
            analysis["completeness"] = completeness
            analysis["agent"] = "agent_2_data_analyzer"

            logger.info(f"Agent 2: Qualidade dos dados: {analysis.get('overall_quality', 'N/A')}")

            return analysis

        except Exception as e:
            logger.error(f"Agent 2: Erro na análise: {e}", exc_info=True)
            return self._get_default_response(str(e))

    def _validate_data_quality(self, df: pd.DataFrame) -> dict[str, Any]:
        """Valida qualidade dos dados"""

        issues = []

        # Verificar valores nulos
        null_counts = df.isnull().sum()
        if null_counts.any():
            issues.append(f"Valores nulos encontrados: {null_counts[null_counts > 0].to_dict()}")

        # Verificar valores negativos
        if (df[["open", "high", "low", "close", "volume"]] < 0).any().any():
            issues.append("Valores negativos encontrados")

        # Verificar consistência OHLC
        invalid_ohlc = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        ).sum()

        if invalid_ohlc > 0:
            issues.append(f"{invalid_ohlc} candles com OHLC inconsistente")

        # Verificar duplicatas
        if "timestamp" in df.columns:
            duplicates = df["timestamp"].duplicated().sum()
            if duplicates > 0:
                issues.append(f"{duplicates} timestamps duplicados")

        # Score de qualidade
        quality_score = 100
        quality_score -= len(null_counts[null_counts > 0]) * 10
        quality_score -= invalid_ohlc * 5
        quality_score = max(0, quality_score)

        return {
            "quality_score": quality_score,
            "issues": issues,
            "total_candles": len(df),
            "null_values": int(null_counts.sum()),
            "invalid_ohlc": int(invalid_ohlc),
        }

    def _detect_anomalies(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Detecta anomalias nos dados"""

        anomalies = []

        # Detectar spikes de preço
        price_changes = df["close"].pct_change().abs()
        spike_threshold = price_changes.mean() + 3 * price_changes.std()
        spikes = price_changes > spike_threshold

        if spikes.sum() > 0:
            anomalies.append(
                {
                    "type": "price_spike",
                    "count": int(spikes.sum()),
                    "severity": "high" if spikes.sum() > 5 else "medium",
                }
            )

        # Detectar volume anormal
        volume_changes = df["volume"].pct_change().abs()
        volume_threshold = volume_changes.mean() + 3 * volume_changes.std()
        volume_spikes = volume_changes > volume_threshold

        if volume_spikes.sum() > 0:
            anomalies.append(
                {"type": "volume_spike", "count": int(volume_spikes.sum()), "severity": "medium"}
            )

        # Detectar gaps
        gaps = []
        for i in range(1, len(df)):
            if df["low"].iloc[i] > df["high"].iloc[i - 1]:
                gaps.append("gap_up")
            elif df["high"].iloc[i] < df["low"].iloc[i - 1]:
                gaps.append("gap_down")

        if gaps:
            anomalies.append({"type": "price_gap", "count": len(gaps), "severity": "low"})

        return anomalies

    def _assess_completeness(self, df: pd.DataFrame) -> dict[str, Any]:
        """Avalia completude dos dados"""

        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        # Calcular cobertura temporal
        if "timestamp" in df.columns and len(df) > 1:
            df_sorted = df.sort_values("timestamp")
            time_diffs = pd.to_datetime(df_sorted["timestamp"]).diff()
            expected_diff = time_diffs.median()
            gaps = (time_diffs > expected_diff * 1.5).sum()
        else:
            gaps = 0

        completeness_score = 100
        completeness_score -= len(missing_columns) * 20
        completeness_score -= min(gaps * 2, 30)
        completeness_score = max(0, completeness_score)

        return {
            "completeness_score": completeness_score,
            "missing_columns": missing_columns,
            "temporal_gaps": int(gaps),
            "data_points": len(df),
        }

    def _build_prompt(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        quality: dict,
        anomalies: list,
        completeness: dict,
        market_context: dict,
    ) -> str:
        """Constrói prompt para o LLM"""

        prompt = f"""
Você é um especialista em análise de qualidade de dados de mercado financeiro.

DADOS ANALISADOS:
- Símbolo: {symbol}
- Timeframe: {timeframe}
- Total de Candles: {len(df)}

QUALIDADE DOS DADOS:
- Score de Qualidade: {quality['quality_score']}/100
- Valores Nulos: {quality['null_values']}
- OHLC Inválidos: {quality['invalid_ohlc']}
- Issues: {', '.join(quality['issues']) if quality['issues'] else 'Nenhum'}

ANOMALIAS DETECTADAS:
{json.dumps(anomalies, indent=2) if anomalies else 'Nenhuma anomalia detectada'}

COMPLETUDE:
- Score: {completeness['completeness_score']}/100
- Gaps Temporais: {completeness['temporal_gaps']}
- Pontos de Dados: {completeness['data_points']}

CONTEXTO DE MERCADO:
- Regime: {market_context.get('market_regime', 'N/A')}
- Volatilidade: {market_context.get('market_metrics', {}).get('volatility', 'N/A')}

TAREFA:
Avalie a qualidade geral dos dados e forneça recomendações:
1. Qualidade geral (excellent/good/fair/poor)
2. Confiabilidade para trading (high/medium/low)
3. Melhor timeframe para análise
4. Recomendações de uso

RESPONDA EM JSON (sem markdown, apenas JSON puro):
{{
  "overall_quality": "excellent|good|fair|poor",
  "reliability": "high|medium|low",
  "confidence": 0.XX,
  "best_timeframe": "1min|5min|15min|1h|4h|1d",
  "recommended_use": "safe|caution|avoid",
  "reasoning": "Explicação detalhada em 2-3 frases",
  "strengths": ["Força 1", "Força 2"],
  "weaknesses": ["Fraqueza 1", "Fraqueza 2"],
  "recommendations": ["Recomendação 1", "Recomendação 2"]
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
            logger.error(f"Agent 2: Erro ao parsear resposta: {e}")
            return {
                "overall_quality": "fair",
                "reliability": "medium",
                "confidence": 0.5,
                "best_timeframe": "1h",
                "recommended_use": "caution",
                "reasoning": f"Erro ao processar análise: {e!s}",
                "strengths": [],
                "weaknesses": ["Análise indisponível"],
                "recommendations": ["Validar dados manualmente"],
            }

    def _get_default_response(self, error_msg: str) -> dict:
        """Retorna resposta padrão em caso de erro"""
        return {
            "overall_quality": "poor",
            "reliability": "low",
            "confidence": 0.0,
            "best_timeframe": "unknown",
            "recommended_use": "avoid",
            "reasoning": f"Erro no sistema: {error_msg}",
            "strengths": [],
            "weaknesses": ["Sistema indisponível"],
            "recommendations": ["Aguardar sistema"],
            "data_quality": {},
            "anomalies": [],
            "completeness": {},
            "agent": "agent_2_data_analyzer",
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

        agent = DataAnalyzer()
        result = await agent.analyze(df, "BTC/USD", "1h")

        print("\n📊 AGENT 2 - DATA ANALYZER:")
        print(f"Qualidade Geral: {result['overall_quality']}")
        print(f"Confiabilidade: {result['reliability']}")
        print(f"Melhor Timeframe: {result['best_timeframe']}")
        print(f"Uso Recomendado: {result['recommended_use']}")
        print(f"Justificativa: {result['reasoning']}")

    asyncio.run(test_agent())
