"""
CeciAI - Testes dos Agentes
Testa os agentes 4 e 8 com mocks das integrações

Autor: CeciAI Team
Data: 2025-10-08
"""

from unittest.mock import patch

import pandas as pd
import pytest

from agents.agent_4_candlestick_specialist import CandlestickSpecialist
from agents.agent_8_daytrade_executor import DayTradeExecutor


@pytest.fixture
def sample_ohlcv_data():
    """Fixture com dados OHLCV de exemplo"""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="1H"),
            "open": [50000 + i * 100 for i in range(20)],
            "high": [50500 + i * 100 for i in range(20)],
            "low": [49500 + i * 100 for i in range(20)],
            "close": [50300 + i * 100 for i in range(20)],
            "volume": [1000000 + i * 10000 for i in range(20)],
        }
    )


@pytest.fixture
def sample_context():
    """Fixture com contexto de exemplo"""
    return {
        "symbol": "BTC/USD",
        "timeframe": "1h",
        "trend": "uptrend",
        "market_conditions": "normal",
    }


class TestCandlestickSpecialist:
    """Testes para Agent 4 - Candlestick Specialist"""

    def test_init(self):
        """Testa inicialização do agente"""
        agent = CandlestickSpecialist()
        assert agent.model == "llama3.2:3b"
        assert hasattr(agent, "detector")
        assert agent.detector is not None

    @pytest.mark.asyncio
    async def test_analyze_basic(self, sample_ohlcv_data, sample_context):
        """Testa análise básica"""
        agent = CandlestickSpecialist()

        # Mock Ollama
        with patch("agents.agent_4_candlestick_specialist.ollama.chat") as mock_ollama:
            mock_ollama.return_value = {
                "message": {
                    "content": '{"signal": "BUY", "confidence": 0.8, "reasoning": "Padrões bullish detectados"}'
                }
            }

            # Mock detector
            with patch.object(agent.detector, "detect_all_patterns") as mock_detect:
                mock_detect.return_value = [
                    {"pattern": "hammer", "confidence": 0.8, "signal": "bullish"}
                ]

                result = await agent.analyze(sample_ohlcv_data, sample_context)

                assert "signal" in result
                assert "confidence" in result
                assert "reasoning" in result
                assert result["signal"] in ["BUY", "SELL", "HOLD"]

    @pytest.mark.asyncio
    async def test_analyze_with_mock_ollama(self, sample_ohlcv_data, sample_context):
        """Testa análise com Ollama mockado"""
        agent = CandlestickSpecialist()

        # Mock Ollama para retornar HOLD
        with patch("agents.agent_4_candlestick_specialist.ollama.generate") as mock_ollama:
            mock_ollama.return_value = {
                "response": '{"signal": "HOLD", "confidence": 0.5, "reasoning": "Nenhum padrão significativo detectado", "key_patterns": [], "confirmation_needed": true, "confirmation_criteria": "Aguardar próximo candle", "risk_level": "LOW", "timeframe_recommendation": "HOLD"}'
            }

            # Mock detector
            with patch.object(agent.detector, "detect_all_patterns") as mock_detect:
                mock_detect.return_value = []

                result = await agent.analyze(sample_ohlcv_data, sample_context)

                assert result["signal"] == "HOLD"
                assert "confidence" in result
                assert "reasoning" in result

    @pytest.mark.asyncio
    async def test_analyze_with_none_context(self, sample_ohlcv_data):
        """Testa análise com contexto None"""
        agent = CandlestickSpecialist()

        with patch("agents.agent_4_candlestick_specialist.ollama.generate") as mock_ollama:
            mock_ollama.return_value = {
                "response": '{"signal": "BUY", "confidence": 0.8, "reasoning": "Padrões bullish detectados", "key_patterns": ["hammer"], "confirmation_needed": false, "confirmation_criteria": "Confirmado", "risk_level": "MEDIUM", "timeframe_recommendation": "SCALPING"}'
            }

            with patch.object(agent.detector, "detect_all_patterns") as mock_detect:
                from utils.candlestick_patterns import CandlePattern, PatternType, SignalStrength

                mock_detect.return_value = [
                    CandlePattern(
                        name="Hammer",
                        pattern_type=PatternType.BULLISH_REVERSAL,
                        signal="BUY",
                        confidence=0.8,
                        significance=80,
                        strength=SignalStrength.STRONG,
                        candle_index=0,
                        description="Test",
                        confirmation_needed=False,
                        confirmation_criteria="Test",
                    )
                ]

                result = await agent.analyze(sample_ohlcv_data, None)  # None context

                assert result["signal"] == "BUY"
                assert "confidence" in result
                assert "reasoning" in result

    @pytest.mark.asyncio
    async def test_analyze_with_exception(self, sample_ohlcv_data, sample_context):
        """Testa análise com exceção"""
        agent = CandlestickSpecialist()

        # Mock para gerar exceção
        with patch.object(
            agent.detector, "detect_all_patterns", side_effect=Exception("Test error")
        ):
            result = await agent.analyze(sample_ohlcv_data, sample_context)

            assert result["signal"] == "HOLD"
            assert result["confidence"] == 0.0
            assert "Erro na análise" in result["reasoning"]
            assert "error" in result

    @pytest.mark.asyncio
    async def test_parse_response_with_markdown(self, sample_ohlcv_data, sample_context):
        """Testa parsing de resposta com markdown"""
        agent = CandlestickSpecialist()

        # Mock Ollama para retornar resposta com markdown
        with patch("agents.agent_4_candlestick_specialist.ollama.generate") as mock_ollama:
            mock_ollama.return_value = {
                "response": '```json\n{"signal": "BUY", "confidence": 0.8, "reasoning": "Test", "key_patterns": [], "confirmation_needed": false, "confirmation_criteria": "Test", "risk_level": "LOW", "timeframe_recommendation": "HOLD"}\n```'
            }

            with patch.object(agent.detector, "detect_all_patterns") as mock_detect:
                mock_detect.return_value = []

                result = await agent.analyze(sample_ohlcv_data, sample_context)

                assert result["signal"] == "BUY"
                assert result["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_parse_response_invalid_json(self, sample_ohlcv_data, sample_context):
        """Testa parsing de resposta com JSON inválido"""
        agent = CandlestickSpecialist()

        # Mock Ollama para retornar JSON inválido
        with patch("agents.agent_4_candlestick_specialist.ollama.generate") as mock_ollama:
            mock_ollama.return_value = {"response": "invalid json response"}

            with patch.object(agent.detector, "detect_all_patterns") as mock_detect:
                mock_detect.return_value = []

                result = await agent.analyze(sample_ohlcv_data, sample_context)

                # Should return default response due to JSON parsing error
                assert result["signal"] == "HOLD"
                assert result["confidence"] == 0.5
                assert "Erro ao parsear resposta do LLM" in result["reasoning"]


class TestDayTradeExecutor:
    """Testes para Agent 8 - Day Trade Executor"""

    def test_init(self):
        """Testa inicialização do agente"""
        agent = DayTradeExecutor()
        assert agent.model == "llama3.2:3b"

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Testa execução básica"""
        agent = DayTradeExecutor()

        agent_analyses = {
            "agent4": {
                "signal": "BUY",
                "confidence": 0.8,
                "patterns": ["hammer", "bullish_engulfing"],
            }
        }

        # Mock Ollama
        with patch("agents.agent_8_daytrade_executor.ollama.chat") as mock_ollama:
            mock_ollama.return_value = {
                "message": {
                    "content": '{"decision": "BUY", "confidence": 0.8, "entry_price": 50000.0, "stop_loss": 49000.0, "take_profit_1": 51000.0, "quantity_usd": 1000.0, "reasoning": "Sinal de compra confirmado"}'
                }
            }

            result = await agent.execute(
                symbol="BTC/USD",
                current_price=50000.0,
                agent_analyses=agent_analyses,
                capital_available=10000.0,
            )

            assert "decision" in result
            assert "confidence" in result
            assert "reasoning" in result
            assert result["decision"] in ["BUY", "SELL", "HOLD"]

    @pytest.mark.asyncio
    async def test_execute_with_different_signals(self):
        """Testa execução com diferentes sinais"""
        agent = DayTradeExecutor()

        # Teste com sinal BUY
        agent_analyses = {"agent4": {"signal": "BUY", "confidence": 0.8, "patterns": ["hammer"]}}

        with patch("agents.agent_8_daytrade_executor.ollama.generate") as mock_ollama:
            mock_ollama.return_value = {
                "response": '{"decision": "BUY", "confidence": 0.8, "entry_price": 50000.0, "entry_type": "MARKET", "quantity_usd": 1000.0, "stop_loss": {"price": 48000.0, "percent": 0.04}, "take_profit_1": {"price": 52000.0, "percent": 0.04}, "take_profit_2": {"price": 54000.0, "percent": 0.08}, "risk_amount_usd": 50.0, "potential_profit_usd": 80.0, "risk_reward_ratio": 1.6, "max_hold_time": "4h", "reasoning": "Sinal confirmado", "validations": {"capital_check": "PASS", "risk_check": "PASS", "rr_ratio_check": "PASS"}}'
            }

            result = await agent.execute(
                symbol="BTC/USD",
                current_price=50000.0,
                agent_analyses=agent_analyses,
                capital_available=10000.0,
            )

            assert result["decision"] == "BUY"
            assert "confidence" in result
            assert "reasoning" in result

    @pytest.mark.asyncio
    async def test_execute_with_markdown_response(self):
        """Testa parsing de resposta com markdown"""
        agent = DayTradeExecutor()

        agent_analyses = {"agent4": {"signal": "BUY", "confidence": 0.8, "patterns": ["hammer"]}}

        # Mock resposta com markdown
        with patch("agents.agent_8_daytrade_executor.ollama.generate") as mock_ollama:
            mock_ollama.return_value = {
                "response": '```json\n{"decision": "BUY", "confidence": 0.8, "entry_price": 50000.0, "entry_type": "MARKET", "quantity_usd": 1000.0, "stop_loss": {"price": 48000.0, "percent": 0.04}, "take_profit_1": {"price": 52000.0, "percent": 0.04}, "take_profit_2": {"price": 54000.0, "percent": 0.08}, "risk_amount_usd": 50.0, "potential_profit_usd": 80.0, "risk_reward_ratio": 1.6, "max_hold_time": "4h", "reasoning": "Sinal confirmado", "validations": {"capital_check": "PASS", "risk_check": "PASS", "rr_ratio_check": "PASS"}}\n```'
            }

            result = await agent.execute(
                symbol="BTC/USD",
                current_price=50000.0,
                agent_analyses=agent_analyses,
                capital_available=10000.0,
            )

            assert result["decision"] == "BUY"
            assert "confidence" in result

    @pytest.mark.asyncio
    async def test_execute_capital_validation_fail(self):
        """Testa falha na validação de capital"""
        agent = DayTradeExecutor()

        agent_analyses = {"agent4": {"signal": "BUY", "confidence": 0.8, "patterns": ["hammer"]}}

        # Mock resposta com quantidade excessiva (>50% do capital)
        with patch("agents.agent_8_daytrade_executor.ollama.generate") as mock_ollama:
            mock_ollama.return_value = {
                "response": '{"decision": "BUY", "confidence": 0.8, "entry_price": 50000.0, "entry_type": "MARKET", "quantity_usd": 6000.0, "stop_loss": {"price": 48000.0, "percent": 0.04}, "take_profit_1": {"price": 52000.0, "percent": 0.04}, "take_profit_2": {"price": 54000.0, "percent": 0.08}, "risk_amount_usd": 50.0, "potential_profit_usd": 80.0, "risk_reward_ratio": 1.6, "max_hold_time": "4h", "reasoning": "Sinal confirmado", "validations": {"capital_check": "PASS", "risk_check": "PASS", "rr_ratio_check": "PASS"}}'
            }

            result = await agent.execute(
                symbol="BTC/USD",
                current_price=50000.0,
                agent_analyses=agent_analyses,
                capital_available=10000.0,
            )

            assert result["decision"] == "HOLD"
            assert result["validations"]["capital_check"] == "FAIL"
            assert "excede 50% do capital" in result["reasoning"]

    @pytest.mark.asyncio
    async def test_execute_risk_validation_fail(self):
        """Testa falha na validação de risco"""
        agent = DayTradeExecutor()

        agent_analyses = {"agent4": {"signal": "BUY", "confidence": 0.8, "patterns": ["hammer"]}}

        # Mock resposta com risco excessivo
        with patch("agents.agent_8_daytrade_executor.ollama.generate") as mock_ollama:
            mock_ollama.return_value = {
                "response": '{"decision": "BUY", "confidence": 0.8, "entry_price": 50000.0, "entry_type": "MARKET", "quantity_usd": 1000.0, "stop_loss": {"price": 48000.0, "percent": 0.04}, "take_profit_1": {"price": 52000.0, "percent": 0.04}, "take_profit_2": {"price": 54000.0, "percent": 0.08}, "risk_amount_usd": 500.0, "potential_profit_usd": 80.0, "risk_reward_ratio": 1.6, "max_hold_time": "4h", "reasoning": "Sinal confirmado", "validations": {"capital_check": "PASS", "risk_check": "PASS", "rr_ratio_check": "PASS"}}'
            }

            result = await agent.execute(
                symbol="BTC/USD",
                current_price=50000.0,
                agent_analyses=agent_analyses,
                capital_available=10000.0,
            )

            assert result["decision"] == "HOLD"
            assert result["validations"]["risk_check"] == "FAIL"
            assert "excede máximo" in result["reasoning"]

    @pytest.mark.asyncio
    async def test_execute_risk_reward_validation_fail(self):
        """Testa falha na validação de risk/reward ratio"""
        agent = DayTradeExecutor()

        agent_analyses = {"agent4": {"signal": "BUY", "confidence": 0.8, "patterns": ["hammer"]}}

        # Mock resposta com risk/reward ratio baixo
        with patch("agents.agent_8_daytrade_executor.ollama.generate") as mock_ollama:
            mock_ollama.return_value = {
                "response": '{"decision": "BUY", "confidence": 0.8, "entry_price": 50000.0, "entry_type": "MARKET", "quantity_usd": 1000.0, "stop_loss": {"price": 48000.0, "percent": 0.04}, "take_profit_1": {"price": 52000.0, "percent": 0.04}, "take_profit_2": {"price": 54000.0, "percent": 0.08}, "risk_amount_usd": 50.0, "potential_profit_usd": 80.0, "risk_reward_ratio": 1.0, "max_hold_time": "4h", "reasoning": "Sinal confirmado", "validations": {"capital_check": "PASS", "risk_check": "PASS", "rr_ratio_check": "PASS"}}'
            }

            result = await agent.execute(
                symbol="BTC/USD",
                current_price=50000.0,
                agent_analyses=agent_analyses,
                capital_available=10000.0,
            )

            assert result["decision"] == "HOLD"
            assert result["validations"]["rr_ratio_check"] == "FAIL"
            assert "abaixo do mínimo" in result["reasoning"]

    @pytest.mark.asyncio
    async def test_execute_price_logic_validation_fail(self):
        """Testa falha na validação de lógica de preços"""
        agent = DayTradeExecutor()

        agent_analyses = {"agent4": {"signal": "BUY", "confidence": 0.8, "patterns": ["hammer"]}}

        # Mock resposta com lógica de preços inválida (stop_loss > entry_price)
        with patch("agents.agent_8_daytrade_executor.ollama.generate") as mock_ollama:
            mock_ollama.return_value = {
                "response": '{"decision": "BUY", "confidence": 0.8, "entry_price": 50000.0, "entry_type": "MARKET", "quantity_usd": 1000.0, "stop_loss": {"price": 52000.0, "percent": 0.04}, "take_profit_1": {"price": 48000.0, "percent": 0.04}, "take_profit_2": {"price": 54000.0, "percent": 0.08}, "risk_amount_usd": 50.0, "potential_profit_usd": 80.0, "risk_reward_ratio": 1.6, "max_hold_time": "4h", "reasoning": "Sinal confirmado", "validations": {"capital_check": "PASS", "risk_check": "PASS", "rr_ratio_check": "PASS"}}'
            }

            result = await agent.execute(
                symbol="BTC/USD",
                current_price=50000.0,
                agent_analyses=agent_analyses,
                capital_available=10000.0,
            )

            assert result["decision"] == "HOLD"
            assert result["validations"]["price_logic_check"] == "FAIL"
            assert "Lógica de preços inválida" in result["reasoning"]

    @pytest.mark.asyncio
    async def test_execute_json_parsing_error(self):
        """Testa erro no parsing do JSON"""
        agent = DayTradeExecutor()

        agent_analyses = {"agent4": {"signal": "BUY", "confidence": 0.8, "patterns": ["hammer"]}}

        # Mock resposta com JSON inválido
        with patch("agents.agent_8_daytrade_executor.ollama.generate") as mock_ollama:
            mock_ollama.return_value = {"response": "invalid json response"}

            result = await agent.execute(
                symbol="BTC/USD",
                current_price=50000.0,
                agent_analyses=agent_analyses,
                capital_available=10000.0,
            )

            assert result["decision"] == "HOLD"
            assert result["confidence"] == 0.0
            assert "Erro ao processar plano" in result["reasoning"]
            assert result["validations"]["capital_check"] == "FAIL"
