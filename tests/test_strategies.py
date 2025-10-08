"""
Tests for trading strategies
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from strategies.arbitrage import ArbitrageStrategy
from strategies.base_strategy import BaseStrategy
from strategies.scalping import ScalpingStrategy
from strategies.swing_trading import SwingTradingStrategy


class TestBaseStrategy:
    """Test BaseStrategy abstract class"""

    def test_base_strategy_init(self):
        """Test BaseStrategy initialization"""

        # Create a concrete implementation for testing
        class TestStrategy(BaseStrategy):
            async def analyze(self, df, symbol, current_price, agent_analyses):
                return {"signal": "HOLD"}

            async def execute(self, analysis, capital_available):
                return {"action": "executed"}

        strategy = TestStrategy(
            name="Test",
            timeframe="1h",
            max_risk_per_trade=0.02,
            min_risk_reward=2.0,
            max_stop_loss=0.05,
        )

        assert strategy.name == "Test"
        assert strategy.timeframe == "1h"
        assert strategy.max_risk_per_trade == 0.02
        assert strategy.min_risk_reward == 2.0
        assert strategy.max_stop_loss == 0.05
        assert strategy.performance_metrics["total_trades"] == 0
        assert strategy.performance_metrics["win_rate"] == 0.0

    def test_base_strategy_calculate_position_size(self):
        """Test position size calculation"""

        class TestStrategy(BaseStrategy):
            async def analyze(self, df, symbol, current_price, agent_analyses):
                return {"signal": "HOLD"}

            async def execute(self, analysis, capital_available):
                return {"action": "executed"}

        strategy = TestStrategy("Test", "1h")

        # Test with valid inputs
        size = strategy.calculate_position_size(
            entry_price=50000, stop_loss=49000, capital_available=10000
        )

        # Should calculate based on risk management
        assert size > 0
        assert isinstance(size, float)

    def test_base_strategy_calculate_position_size_no_risk(self):
        """Test position size calculation with no risk (same entry and stop)"""

        class TestStrategy(BaseStrategy):
            async def analyze(self, df, symbol, current_price, agent_analyses):
                return {"signal": "HOLD"}

            async def execute(self, analysis, capital_available):
                return {"action": "executed"}

        strategy = TestStrategy("Test", "1h")

        # Test with no risk (same entry and stop loss)
        size = strategy.calculate_position_size(
            entry_price=50000,
            stop_loss=50000,  # Same as entry
            capital_available=10000,
        )

        # Should return fallback (10% of capital)
        assert size == 1000  # 10% of 10000

    def test_base_strategy_validate_trade(self):
        """Test trade validation"""

        class TestStrategy(BaseStrategy):
            async def analyze(self, df, symbol, current_price, agent_analyses):
                return {"signal": "HOLD"}

            async def execute(self, analysis, capital_available):
                return {"action": "executed"}

        strategy = TestStrategy("Test", "1h")

        # Test valid trade
        validation = strategy.validate_trade(
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000,
            quantity_usd=1000,
            capital_available=10000,
        )

        assert validation["is_valid"] is True
        assert validation["capital_check"] == "PASS"
        assert validation["risk_check"] == "PASS"
        assert validation["rr_ratio_check"] == "PASS"
        assert validation["stop_loss_check"] == "PASS"

    def test_base_strategy_validate_trade_invalid(self):
        """Test trade validation with invalid parameters"""

        class TestStrategy(BaseStrategy):
            async def analyze(self, df, symbol, current_price, agent_analyses):
                return {"signal": "HOLD"}

            async def execute(self, analysis, capital_available):
                return {"action": "executed"}

        strategy = TestStrategy("Test", "1h")

        # Test with excessive quantity (>50% of capital)
        validation = strategy.validate_trade(
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000,
            quantity_usd=6000,  # 60% of capital
            capital_available=10000,
        )

        assert validation["is_valid"] is False
        assert validation["capital_check"] == "FAIL"

    def test_base_strategy_record_trade(self):
        """Test trade recording"""

        class TestStrategy(BaseStrategy):
            async def analyze(self, df, symbol, current_price, agent_analyses):
                return {"signal": "HOLD"}

            async def execute(self, analysis, capital_available):
                return {"action": "executed"}

        strategy = TestStrategy("Test", "1h")

        # Record winning trade
        trade = {"symbol": "BTC/USD", "side": "BUY", "pnl": 100}

        strategy.record_trade(trade)

        assert strategy.performance_metrics["total_trades"] == 1
        assert strategy.performance_metrics["winning_trades"] == 1
        assert strategy.performance_metrics["total_pnl"] == 100
        assert strategy.performance_metrics["win_rate"] == 1.0
        assert len(strategy.trades_history) == 1

    def test_base_strategy_get_performance_metrics(self):
        """Test getting performance metrics"""

        class TestStrategy(BaseStrategy):
            async def analyze(self, df, symbol, current_price, agent_analyses):
                return {"signal": "HOLD"}

            async def execute(self, analysis, capital_available):
                return {"action": "executed"}

        strategy = TestStrategy("Test", "1h")

        metrics = strategy.get_performance_metrics()

        assert "strategy" in metrics
        assert "timeframe" in metrics
        assert "total_trades" in metrics
        assert "recent_trades" in metrics
        assert metrics["strategy"] == "Test"
        assert metrics["timeframe"] == "1h"

    def test_base_strategy_reset_performance(self):
        """Test resetting performance"""

        class TestStrategy(BaseStrategy):
            async def analyze(self, df, symbol, current_price, agent_analyses):
                return {"signal": "HOLD"}

            async def execute(self, analysis, capital_available):
                return {"action": "executed"}

        strategy = TestStrategy("Test", "1h")

        # Add a trade first
        strategy.record_trade({"pnl": 100})
        assert strategy.performance_metrics["total_trades"] == 1

        # Reset
        strategy.reset_performance()

        assert strategy.performance_metrics["total_trades"] == 0
        assert strategy.performance_metrics["total_pnl"] == 0.0
        assert len(strategy.trades_history) == 0


class TestScalpingStrategy:
    """Test ScalpingStrategy"""

    def test_scalping_strategy_init(self):
        """Test ScalpingStrategy initialization"""
        strategy = ScalpingStrategy()

        assert strategy.name == "Scalping"
        assert strategy.timeframe == "1min"
        assert strategy.max_risk_per_trade == 0.005
        assert strategy.min_risk_reward == 1.5
        assert strategy.max_stop_loss == 0.003
        assert strategy.target_profit_pct == 0.008
        assert strategy.max_hold_time_minutes == 30

    @pytest.mark.asyncio
    async def test_scalping_strategy_analyze(self):
        """Test scalping analysis"""
        strategy = ScalpingStrategy()

        # Create sample data
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="1min"),
                "open": np.linspace(50000, 50500, 100),
                "high": np.linspace(50100, 50600, 100),
                "low": np.linspace(49900, 50400, 100),
                "close": np.linspace(50050, 50550, 100),
                "volume": np.random.uniform(1000, 5000, 100),
            }
        )

        # Mock agent analyses
        agent_analyses = {
            "agent1": {"market_metrics": {"volume_ratio": 1.5, "volatility": 1.2}},
            "agent3": {"indicators": {"rsi": 45, "macd_crossover": "bullish"}},
            "agent4": {"signal": "BUY", "confidence": 0.75},
        }

        result = await strategy.analyze(df, "BTC/USD", 50000, agent_analyses)

        assert isinstance(result, dict)
        assert "signal" in result
        assert "confidence" in result
        assert "strategy" in result
        assert "symbol" in result
        assert result["signal"] in ["BUY", "SELL", "HOLD"]
        assert result["strategy"] == "Scalping"

    @pytest.mark.asyncio
    async def test_scalping_strategy_analyze_hold_signal(self):
        """Test scalping analysis with HOLD signal"""
        strategy = ScalpingStrategy()

        df = pd.DataFrame({"close": np.random.uniform(50000, 51000, 50)})

        # Mock agent analyses with low confidence
        agent_analyses = {
            "agent1": {"market_metrics": {"volume_ratio": 0.8}},
            "agent3": {"indicators": {"rsi": 50}},
            "agent4": {"signal": "HOLD", "confidence": 0.3},
        }

        result = await strategy.analyze(df, "BTC/USD", 50000, agent_analyses)

        assert result["signal"] == "HOLD"
        assert result["confidence"] < 0.7

    @pytest.mark.asyncio
    async def test_scalping_strategy_execute_hold(self):
        """Test scalping execution with HOLD signal"""
        strategy = ScalpingStrategy()

        analysis = {"signal": "HOLD", "confidence": 0.3, "current_price": 50000}

        result = await strategy.execute(analysis, 10000)

        assert result["decision"] == "HOLD"
        assert "reasoning" in result

    @pytest.mark.asyncio
    async def test_scalping_strategy_execute_buy(self):
        """Test scalping execution with BUY signal"""
        strategy = ScalpingStrategy()

        analysis = {
            "signal": "BUY",
            "confidence": 0.8,
            "current_price": 50000,
            "reasons": ["RSI oversold", "Volume adequate"],
            "warnings": [],
        }

        result = await strategy.execute(analysis, 10000)

        assert isinstance(result, dict)
        assert "decision" in result
        assert "entry_price" in result
        assert "stop_loss" in result
        assert "take_profit_1" in result
        assert "quantity_usd" in result
        assert "validations" in result

    @pytest.mark.asyncio
    async def test_scalping_strategy_analyze_rsi_ideal(self):
        """Test scalping analysis with ideal RSI"""
        strategy = ScalpingStrategy()

        df = pd.DataFrame({"close": np.random.uniform(50000, 51000, 50)})

        # Mock agent analyses with ideal RSI
        agent_analyses = {
            "agent1": {"market_metrics": {"volume_ratio": 1.5, "volatility": 1.0}},
            "agent3": {"indicators": {"rsi": 50}},  # Ideal RSI
            "agent4": {"signal": "HOLD", "confidence": 0.3},
        }

        result = await strategy.analyze(df, "BTC/USD", 50000, agent_analyses)

        assert "RSI ideal para scalping" in str(result["reasons"])

    @pytest.mark.asyncio
    async def test_scalping_strategy_analyze_rsi_oversold(self):
        """Test scalping analysis with oversold RSI"""
        strategy = ScalpingStrategy()

        df = pd.DataFrame({"close": np.random.uniform(50000, 51000, 50)})

        # Mock agent analyses with oversold RSI
        agent_analyses = {
            "agent1": {"market_metrics": {"volume_ratio": 1.5, "volatility": 1.0}},
            "agent3": {"indicators": {"rsi": 25}},  # Oversold
            "agent4": {"signal": "HOLD", "confidence": 0.3},
        }

        result = await strategy.analyze(df, "BTC/USD", 50000, agent_analyses)

        assert result["signal"] == "BUY"
        assert "RSI oversold" in str(result["reasons"])

    @pytest.mark.asyncio
    async def test_scalping_strategy_analyze_rsi_overbought(self):
        """Test scalping analysis with overbought RSI"""
        strategy = ScalpingStrategy()

        df = pd.DataFrame({"close": np.random.uniform(50000, 51000, 50)})

        # Mock agent analyses with overbought RSI
        agent_analyses = {
            "agent1": {"market_metrics": {"volume_ratio": 1.5, "volatility": 1.0}},
            "agent3": {"indicators": {"rsi": 75}},  # Overbought
            "agent4": {"signal": "HOLD", "confidence": 0.3},
        }

        result = await strategy.analyze(df, "BTC/USD", 50000, agent_analyses)

        assert result["signal"] == "SELL"
        assert "RSI overbought" in str(result["reasons"])

    @pytest.mark.asyncio
    async def test_scalping_strategy_analyze_macd_bullish(self):
        """Test scalping analysis with bullish MACD"""
        strategy = ScalpingStrategy()

        df = pd.DataFrame({"close": np.random.uniform(50000, 51000, 50)})

        # Mock agent analyses with bullish MACD
        agent_analyses = {
            "agent1": {"market_metrics": {"volume_ratio": 1.5, "volatility": 1.0}},
            "agent3": {"indicators": {"rsi": 50, "macd_crossover": "bullish"}},
            "agent4": {"signal": "HOLD", "confidence": 0.3},
        }

        result = await strategy.analyze(df, "BTC/USD", 50000, agent_analyses)

        assert result["signal"] == "BUY"
        assert "MACD bullish crossover" in result["reasons"]

    @pytest.mark.asyncio
    async def test_scalping_strategy_analyze_macd_bearish(self):
        """Test scalping analysis with bearish MACD"""
        strategy = ScalpingStrategy()

        df = pd.DataFrame({"close": np.random.uniform(50000, 51000, 50)})

        # Mock agent analyses with bearish MACD
        agent_analyses = {
            "agent1": {"market_metrics": {"volume_ratio": 1.5, "volatility": 1.0}},
            "agent3": {"indicators": {"rsi": 50, "macd_crossover": "bearish"}},
            "agent4": {"signal": "HOLD", "confidence": 0.3},
        }

        result = await strategy.analyze(df, "BTC/USD", 50000, agent_analyses)

        assert result["signal"] == "SELL"
        assert "MACD bearish crossover" in result["reasons"]

    @pytest.mark.asyncio
    async def test_scalping_strategy_analyze_strong_pattern(self):
        """Test scalping analysis with strong pattern"""
        strategy = ScalpingStrategy()

        df = pd.DataFrame({"close": np.random.uniform(50000, 51000, 50)})

        # Mock agent analyses with strong pattern
        agent_analyses = {
            "agent1": {"market_metrics": {"volume_ratio": 1.5, "volatility": 1.0}},
            "agent3": {"indicators": {"rsi": 50}},
            "agent4": {"signal": "BUY", "confidence": 0.8},  # Strong pattern
        }

        result = await strategy.analyze(df, "BTC/USD", 50000, agent_analyses)

        assert result["signal"] == "BUY"
        assert "Padrão forte detectado" in str(result["reasons"])

    @pytest.mark.asyncio
    async def test_scalping_strategy_analyze_high_volatility(self):
        """Test scalping analysis with high volatility"""
        strategy = ScalpingStrategy()

        df = pd.DataFrame({"close": np.random.uniform(50000, 51000, 50)})

        # Mock agent analyses with high volatility
        agent_analyses = {
            "agent1": {
                "market_metrics": {"volume_ratio": 1.5, "volatility": 4.0}
            },  # High volatility
            "agent3": {"indicators": {"rsi": 50}},
            "agent4": {"signal": "HOLD", "confidence": 0.3},
        }

        result = await strategy.analyze(df, "BTC/USD", 50000, agent_analyses)

        assert any("Alta volatilidade" in warning for warning in result["warnings"])

    @pytest.mark.asyncio
    async def test_scalping_strategy_analyze_low_volatility(self):
        """Test scalping analysis with low volatility"""
        strategy = ScalpingStrategy()

        df = pd.DataFrame({"close": np.random.uniform(50000, 51000, 50)})

        # Mock agent analyses with low volatility
        agent_analyses = {
            "agent1": {
                "market_metrics": {"volume_ratio": 1.5, "volatility": 0.3}
            },  # Low volatility
            "agent3": {"indicators": {"rsi": 50}},
            "agent4": {"signal": "HOLD", "confidence": 0.3},
        }

        result = await strategy.analyze(df, "BTC/USD", 50000, agent_analyses)

        assert any("Baixa volatilidade" in warning for warning in result["warnings"])

    @pytest.mark.asyncio
    async def test_scalping_strategy_analyze_exception(self):
        """Test scalping analysis with exception"""
        strategy = ScalpingStrategy()

        # Mock agent_analyses to cause exception during processing
        agent_analyses = None  # This will cause an exception when accessing .get()

        df = pd.DataFrame({"close": np.random.uniform(50000, 51000, 50)})

        result = await strategy.analyze(df, "BTC/USD", 50000, agent_analyses)

        assert result["signal"] == "HOLD"
        assert any("Erro na análise" in warning for warning in result["warnings"])

    @pytest.mark.asyncio
    async def test_scalping_strategy_execute_sell(self):
        """Test scalping execution with SELL signal"""
        strategy = ScalpingStrategy()

        analysis = {
            "signal": "SELL",
            "confidence": 0.8,
            "current_price": 50000,
            "reasons": ["RSI overbought", "Volume adequate"],
            "warnings": [],
        }

        result = await strategy.execute(analysis, 10000)

        assert isinstance(result, dict)
        assert "decision" in result
        assert "entry_price" in result
        assert "stop_loss" in result
        assert "take_profit_1" in result
        assert "quantity_usd" in result
        assert "validations" in result

    @pytest.mark.asyncio
    async def test_scalping_strategy_execute_zero_price(self):
        """Test scalping execution with zero current price"""
        strategy = ScalpingStrategy()

        analysis = {
            "signal": "BUY",
            "confidence": 0.8,
            "current_price": 0,  # Zero price
            "reasons": ["RSI oversold"],
            "warnings": [],
        }

        result = await strategy.execute(analysis, 10000)

        assert result["decision"] == "HOLD"
        assert "Nenhum sinal claro" in result["reasoning"]

    @pytest.mark.asyncio
    async def test_scalping_strategy_execute_invalid_validation(self):
        """Test scalping execution with invalid validation"""
        strategy = ScalpingStrategy()

        # Mock validate_trade to return invalid
        with patch.object(strategy, "validate_trade") as mock_validate:
            mock_validate.return_value = {
                "is_valid": False,
                "errors": ["Risk too high", "Capital insufficient"],
            }

            analysis = {
                "signal": "BUY",
                "confidence": 0.8,
                "current_price": 50000,
                "reasons": ["RSI oversold"],
                "warnings": [],
            }

            result = await strategy.execute(analysis, 10000)

            assert result["decision"] == "HOLD"
            assert "Validação falhou" in result["reasoning"]


class TestSwingTradingStrategy:
    """Test SwingTradingStrategy"""

    def test_swing_trading_strategy_init(self):
        """Test SwingTradingStrategy initialization"""
        strategy = SwingTradingStrategy()

        assert strategy.name == "Swing Trading"
        assert strategy.timeframe == "4h"
        # Check actual values from implementation
        assert strategy.max_risk_per_trade == 0.01  # 1% per trade
        assert strategy.min_risk_reward == 2.0
        assert strategy.max_stop_loss == 0.02  # 2%

    @pytest.mark.asyncio
    async def test_swing_trading_strategy_analyze(self):
        """Test swing trading analysis"""
        strategy = SwingTradingStrategy()

        # Create sample data
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="4h"),
                "open": np.random.uniform(50000, 51000, 100),
                "high": np.random.uniform(50500, 51500, 100),
                "low": np.random.uniform(49500, 50500, 100),
                "close": np.random.uniform(50000, 51000, 100),
                "volume": np.random.uniform(1000, 5000, 100),
            }
        )

        # Mock agent analyses
        agent_analyses = {
            "agent1": {"market_metrics": {"trend": "bullish"}},
            "agent3": {"indicators": {"rsi": 35}},
            "agent4": {"signal": "BUY", "confidence": 0.8},
        }

        result = await strategy.analyze(df, "BTC/USD", 50000, agent_analyses)

        assert isinstance(result, dict)
        assert "signal" in result
        assert result["signal"] in ["BUY", "SELL", "HOLD"]

    @pytest.mark.asyncio
    async def test_swing_trading_strategy_analyze_uptrend(self):
        """Test swing trading analysis with uptrend"""
        strategy = SwingTradingStrategy()

        df = pd.DataFrame({"close": np.random.uniform(50000, 51000, 50)})

        # Test uptrend signal
        agent_analyses = {
            "agent1": {"market_metrics": {"trend": "uptrend"}},
            "agent3": {"indicators": {"rsi": 50}},
        }

        result = await strategy.analyze(df, "BTC/USD", 50000, agent_analyses)

        assert result["signal"] == "BUY"
        assert "Tendência de alta confirmada" in result["reasons"]

    @pytest.mark.asyncio
    async def test_swing_trading_strategy_analyze_downtrend(self):
        """Test swing trading analysis with downtrend"""
        strategy = SwingTradingStrategy()

        df = pd.DataFrame({"close": np.random.uniform(50000, 51000, 50)})

        # Test downtrend signal
        agent_analyses = {
            "agent1": {"market_metrics": {"trend": "downtrend"}},
            "agent3": {"indicators": {"rsi": 50}},
        }

        result = await strategy.analyze(df, "BTC/USD", 50000, agent_analyses)

        assert result["signal"] == "SELL"
        assert "Tendência de baixa confirmada" in result["reasons"]

    @pytest.mark.asyncio
    async def test_swing_trading_strategy_analyze_rsi_oversold(self):
        """Test swing trading analysis with RSI oversold"""
        strategy = SwingTradingStrategy()

        df = pd.DataFrame({"close": np.random.uniform(50000, 51000, 50)})

        # Test RSI oversold
        agent_analyses = {
            "agent1": {"market_metrics": {"trend": "neutral"}},
            "agent3": {"indicators": {"rsi": 30}},
        }

        result = await strategy.analyze(df, "BTC/USD", 50000, agent_analyses)

        assert result["signal"] == "BUY"
        assert any("RSI oversold" in reason for reason in result["reasons"])

    @pytest.mark.asyncio
    async def test_swing_trading_strategy_analyze_rsi_overbought(self):
        """Test swing trading analysis with RSI overbought"""
        strategy = SwingTradingStrategy()

        df = pd.DataFrame({"close": np.random.uniform(50000, 51000, 50)})

        # Test RSI overbought
        agent_analyses = {
            "agent1": {"market_metrics": {"trend": "neutral"}},
            "agent3": {"indicators": {"rsi": 70}},
        }

        result = await strategy.analyze(df, "BTC/USD", 50000, agent_analyses)

        assert result["signal"] == "SELL"
        assert any("RSI overbought" in reason for reason in result["reasons"])

    @pytest.mark.asyncio
    async def test_swing_trading_strategy_execute_buy(self):
        """Test swing trading execution with BUY signal"""
        strategy = SwingTradingStrategy()

        analysis = {
            "signal": "BUY",
            "confidence": 0.8,
            "current_price": 50000,
            "reasons": ["Uptrend confirmed", "RSI oversold"],
        }

        result = await strategy.execute(analysis, 10000)

        assert result["decision"] == "BUY" or result["decision"] == "HOLD"  # Depends on validation
        assert "entry_price" in result
        assert "stop_loss" in result
        assert "take_profit" in result
        assert "quantity_usd" in result
        assert "validations" in result

    @pytest.mark.asyncio
    async def test_swing_trading_strategy_execute_sell(self):
        """Test swing trading execution with SELL signal"""
        strategy = SwingTradingStrategy()

        analysis = {
            "signal": "SELL",
            "confidence": 0.8,
            "current_price": 50000,
            "reasons": ["Downtrend confirmed", "RSI overbought"],
        }

        result = await strategy.execute(analysis, 10000)

        assert result["decision"] == "SELL" or result["decision"] == "HOLD"  # Depends on validation
        assert "entry_price" in result
        assert "stop_loss" in result
        assert "take_profit" in result
        assert "quantity_usd" in result
        assert "validations" in result


class TestArbitrageStrategy:
    """Test ArbitrageStrategy"""

    def test_arbitrage_strategy_init(self):
        """Test ArbitrageStrategy initialization"""
        strategy = ArbitrageStrategy()

        assert strategy.name == "Arbitrage"
        assert strategy.timeframe == "1min"
        assert strategy.max_risk_per_trade == 0.002
        # Check actual value from implementation
        assert strategy.min_risk_reward == 1.2  # Overridden in ArbitrageStrategy
        assert strategy.max_stop_loss == 0.005  # 0.5%

    @pytest.mark.asyncio
    async def test_arbitrage_strategy_analyze(self):
        """Test arbitrage analysis"""
        strategy = ArbitrageStrategy()

        # Create sample data
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=50, freq="1min"),
                "open": np.random.uniform(50000, 51000, 50),
                "high": np.random.uniform(50500, 51500, 50),
                "low": np.random.uniform(49500, 50500, 50),
                "close": np.random.uniform(50000, 51000, 50),
                "volume": np.random.uniform(1000, 5000, 50),
            }
        )

        # Mock agent analyses
        agent_analyses = {
            "agent1": {"market_metrics": {"spread": 0.5}},
            "agent3": {"indicators": {"volatility": 0.8}},
            "agent4": {"signal": "BUY", "confidence": 0.6},
        }

        result = await strategy.analyze(df, "BTC/USD", 50000, agent_analyses)

        assert isinstance(result, dict)
        assert "signal" in result
        assert result["signal"] in ["BUY", "SELL", "HOLD"]
