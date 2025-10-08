"""
Tests for backtesting module
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest_engine import BacktestEngine
from backtesting.paper_trading import PaperTradingEngine


class TestBacktestEngine:
    """Test BacktestEngine"""

    def test_backtest_engine_init(self):
        """Test BacktestEngine initialization"""
        engine = BacktestEngine(initial_capital=20000.0)

        assert engine.initial_capital == 20000.0
        assert engine.capital == 20000.0
        assert engine.trades == []
        assert engine.equity_curve == []

    def test_backtest_engine_init_default(self):
        """Test BacktestEngine with default capital"""
        engine = BacktestEngine()

        assert engine.initial_capital == 10000.0
        assert engine.capital == 10000.0

    @pytest.mark.asyncio
    async def test_backtest_engine_run(self):
        """Test running a backtest"""
        engine = BacktestEngine()

        # Create sample data with timestamp column
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="1h"),
                "open": np.random.uniform(50000, 51000, 100),
                "high": np.random.uniform(50500, 51500, 100),
                "low": np.random.uniform(49500, 50500, 100),
                "close": np.random.uniform(50000, 51000, 100),
                "volume": np.random.uniform(1000, 5000, 100),
            }
        )

        # Mock strategy and agent pipeline
        mock_strategy = Mock()
        mock_agent_pipeline = Mock()

        result = await engine.run(df, mock_strategy, mock_agent_pipeline)

        assert isinstance(result, dict)
        assert "initial_capital" in result
        assert "final_capital" in result
        assert "total_return" in result
        assert "trades" in result
        assert "metrics" in result
        assert "equity_curve" in result
        assert result["initial_capital"] == 10000.0
        assert len(result["equity_curve"]) > 0

    @pytest.mark.asyncio
    async def test_backtest_engine_run_no_timestamp(self):
        """Test running backtest without timestamp column"""
        engine = BacktestEngine()

        # Create sample data without timestamp column
        df = pd.DataFrame(
            {
                "open": np.random.uniform(50000, 51000, 100),
                "high": np.random.uniform(50500, 51500, 100),
                "low": np.random.uniform(49500, 50500, 100),
                "close": np.random.uniform(50000, 51000, 100),
                "volume": np.random.uniform(1000, 5000, 100),
            }
        )

        mock_strategy = Mock()
        mock_agent_pipeline = Mock()

        result = await engine.run(df, mock_strategy, mock_agent_pipeline)

        assert isinstance(result, dict)
        assert len(result["equity_curve"]) > 0
        # Should use index when no timestamp column
        assert isinstance(result["equity_curve"][0]["timestamp"], int | np.integer)

    def test_backtest_engine_calculate_metrics_no_trades(self):
        """Test metrics calculation with no trades"""
        engine = BacktestEngine()

        metrics = engine._calculate_metrics()

        assert metrics == {}

    def test_backtest_engine_calculate_metrics_with_trades(self):
        """Test metrics calculation with trades"""
        engine = BacktestEngine(initial_capital=10000.0)

        # Add some trades manually
        engine.trades = [{"pnl": 100}, {"pnl": 150}, {"pnl": -50}]

        metrics = engine._calculate_metrics()

        assert metrics["total_trades"] == 3
        assert metrics["winning_trades"] == 2
        assert metrics["losing_trades"] == 1
        assert metrics["win_rate"] == 2 / 3
        assert isinstance(metrics["sharpe_ratio"], int | float)
        assert isinstance(metrics["max_drawdown"], int | float)

    def test_backtest_engine_calculate_max_drawdown_no_equity(self):
        """Test max drawdown calculation with no equity curve"""
        engine = BacktestEngine()

        max_dd = engine._calculate_max_drawdown()

        assert max_dd == 0.0

    def test_backtest_engine_calculate_max_drawdown_with_equity(self):
        """Test max drawdown calculation with equity curve"""
        engine = BacktestEngine()

        # Create equity curve with drawdown
        engine.equity_curve = [
            {"equity": 10000},
            {"equity": 11000},  # Peak
            {"equity": 10500},  # Drawdown
            {"equity": 9500},  # Bigger drawdown
            {"equity": 10200},  # Recovery
        ]

        max_dd = engine._calculate_max_drawdown()

        # Max drawdown should be (11000 - 9500) / 11000 = 0.136...
        assert max_dd > 0
        assert max_dd < 1.0


class TestPaperTradingEngine:
    """Test PaperTradingEngine"""

    @patch("backtesting.paper_trading.CapitalManager")
    def test_paper_trading_engine_init(self, mock_capital_manager):
        """Test PaperTradingEngine initialization"""
        engine = PaperTradingEngine(initial_capital=15000.0)

        mock_capital_manager.assert_called_once_with(initial_capital=15000.0)
        assert engine.is_running is False
        assert hasattr(engine, "capital_manager")

    @patch("backtesting.paper_trading.CapitalManager")
    def test_paper_trading_engine_init_default(self, mock_capital_manager):
        """Test PaperTradingEngine with default capital"""
        _ = PaperTradingEngine()

        mock_capital_manager.assert_called_once_with(initial_capital=10000.0)

    @patch("backtesting.paper_trading.CapitalManager")
    @pytest.mark.asyncio
    async def test_paper_trading_engine_start(self, mock_capital_manager):
        """Test starting paper trading"""
        engine = PaperTradingEngine()

        await engine.start()

        assert engine.is_running is True

    @patch("backtesting.paper_trading.CapitalManager")
    @pytest.mark.asyncio
    async def test_paper_trading_engine_stop(self, mock_capital_manager):
        """Test stopping paper trading"""
        engine = PaperTradingEngine()
        engine.is_running = True

        await engine.stop()

        assert engine.is_running is False

    @patch("backtesting.paper_trading.CapitalManager")
    @pytest.mark.asyncio
    async def test_paper_trading_engine_execute_trade_not_running(self, mock_capital_manager):
        """Test executing trade when not running"""
        engine = PaperTradingEngine()
        engine.is_running = False

        trade_request = {"symbol": "BTC/USD", "side": "BUY", "quantity_usd": 1000, "price": 50000}

        result = await engine.execute_trade(trade_request)

        assert result["success"] is False
        assert "error" in result
        assert "não está rodando" in result["error"]

    @patch("backtesting.paper_trading.CapitalManager")
    @pytest.mark.asyncio
    async def test_paper_trading_engine_execute_trade_success(self, mock_capital_manager):
        """Test successful trade execution"""
        mock_capital_manager_instance = Mock()
        mock_capital_manager.return_value = mock_capital_manager_instance
        mock_capital_manager_instance.can_open_position.return_value = {
            "can_open": True,
            "reasons": [],
        }
        mock_capital_manager_instance.open_position.return_value = True
        mock_capital_manager_instance.get_status.return_value = {
            "total_capital": 10000,
            "available_capital": 9000,
        }

        engine = PaperTradingEngine()
        engine.is_running = True

        trade_request = {"symbol": "BTC/USD", "side": "BUY", "quantity_usd": 1000, "price": 50000}

        result = await engine.execute_trade(trade_request)

        assert result["success"] is True
        assert "trade" in result
        assert "capital_status" in result
        mock_capital_manager_instance.can_open_position.assert_called_once_with(1000)
        mock_capital_manager_instance.open_position.assert_called_once_with(trade_request)

    @patch("backtesting.paper_trading.CapitalManager")
    @pytest.mark.asyncio
    async def test_paper_trading_engine_execute_trade_insufficient_capital(
        self, mock_capital_manager
    ):
        """Test executing trade with insufficient capital"""
        mock_capital_manager_instance = Mock()
        mock_capital_manager.return_value = mock_capital_manager_instance
        mock_capital_manager_instance.can_open_position.return_value = {
            "can_open": False,
            "reasons": ["Insufficient capital"],
        }

        engine = PaperTradingEngine()
        engine.is_running = True

        trade_request = {
            "symbol": "BTC/USD",
            "side": "BUY",
            "quantity_usd": 15000,  # Large quantity
            "price": 50000,
        }

        result = await engine.execute_trade(trade_request)

        assert result["success"] is False
        assert "reasons" in result
        assert "Insufficient capital" in result["reasons"]
        mock_capital_manager_instance.can_open_position.assert_called_once_with(15000)

    @patch("backtesting.paper_trading.CapitalManager")
    def test_paper_trading_engine_get_status(self, mock_capital_manager):
        """Test getting paper trading status"""
        mock_capital_manager_instance = Mock()
        mock_capital_manager.return_value = mock_capital_manager_instance
        mock_capital_manager_instance.get_status.return_value = {
            "total_capital": 10500.0,
            "available_capital": 8000.0,
        }
        mock_capital_manager_instance.get_performance_metrics.return_value = {
            "total_trades": 5,
            "win_rate": 0.6,
            "total_profit": 500.0,
        }

        engine = PaperTradingEngine()
        engine.is_running = True

        status = engine.get_status()

        assert status["is_running"] is True
        assert "capital" in status
        assert "performance" in status
        assert status["capital"]["total_capital"] == 10500.0
        assert status["performance"]["total_trades"] == 5
        mock_capital_manager_instance.get_status.assert_called_once()
        mock_capital_manager_instance.get_performance_metrics.assert_called_once()
