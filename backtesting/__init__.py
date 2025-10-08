"""
CeciAI - Backtesting Module
Sistema de backtesting para validação de estratégias

Autor: CeciAI Team
Data: 2025-10-08
"""

from .backtest_engine import BacktestEngine
from .paper_trading import PaperTradingEngine

__all__ = ["BacktestEngine", "PaperTradingEngine"]
