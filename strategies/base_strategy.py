"""
CeciAI - Base Strategy
Classe base para todas as estratégias de trading

Responsabilidades:
- Interface comum para todas as estratégias
- Validações básicas
- Gestão de risco padrão
- Logging e tracking

Autor: CeciAI Team
Data: 2025-10-08
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Classe base abstrata para estratégias de trading

    Todas as estratégias devem herdar desta classe e implementar:
    - analyze(): Análise da oportunidade
    - execute(): Execução do trade
    - validate(): Validação das condições
    """

    def __init__(
        self,
        name: str,
        timeframe: str,
        max_risk_per_trade: float = 0.01,
        min_risk_reward: float = 1.5,
        max_stop_loss: float = 0.03,
    ):
        """
        Inicializa a estratégia.

        Args:
            name: Nome da estratégia
            timeframe: Timeframe principal
            max_risk_per_trade: Risco máximo por trade (% do capital)
            min_risk_reward: Risk/Reward mínimo
            max_stop_loss: Stop loss máximo (% do preço)
        """
        self.name = name
        self.timeframe = timeframe
        self.max_risk_per_trade = max_risk_per_trade
        self.min_risk_reward = min_risk_reward
        self.max_stop_loss = max_stop_loss

        self.trades_history = []
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
        }

        logger.info(f"Estratégia '{name}' inicializada (timeframe: {timeframe})")

    @abstractmethod
    async def analyze(
        self, df: pd.DataFrame, symbol: str, current_price: float, agent_analyses: dict[str, object]
    ) -> dict[str, object]:
        """
        Analisa oportunidade de trade.

        Args:
            df: DataFrame com dados OHLCV
            symbol: Par de trading
            current_price: Preço atual
            agent_analyses: Análises dos agentes LLM

        Returns:
            Dict com análise da estratégia
        """

    @abstractmethod
    async def execute(self, analysis: dict[str, object], capital_available: float) -> dict[str, object]:
        """
        Executa trade baseado na análise.

        Args:
            analysis: Resultado da análise
            capital_available: Capital disponível

        Returns:
            Dict com plano de execução
        """

    def validate_trade(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        quantity_usd: float,
        capital_available: float,
    ) -> dict[str, Any]:
        """
        Valida trade antes da execução.

        Args:
            entry_price: Preço de entrada
            stop_loss: Stop loss
            take_profit: Take profit
            quantity_usd: Quantidade em USD
            capital_available: Capital disponível

        Returns:
            Dict com validações
        """
        validations = {
            "capital_check": "PASS",
            "risk_check": "PASS",
            "rr_ratio_check": "PASS",
            "stop_loss_check": "PASS",
            "is_valid": True,
            "errors": [],
        }

        # 1. Validar capital
        if quantity_usd > capital_available * 0.5:
            validations["capital_check"] = "FAIL"
            validations["errors"].append(f"Quantidade ({quantity_usd:.2f}) excede 50% do capital")

        # 2. Validar risco
        risk_amount = abs(entry_price - stop_loss) / entry_price * quantity_usd
        max_risk = capital_available * self.max_risk_per_trade

        if risk_amount > max_risk:
            validations["risk_check"] = "FAIL"
            validations["errors"].append(
                f"Risco ({risk_amount:.2f}) excede máximo ({max_risk:.2f})"
            )

        # 3. Validar risk/reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0

        if rr_ratio < self.min_risk_reward:
            validations["rr_ratio_check"] = "FAIL"
            validations["errors"].append(
                f"R/R ({rr_ratio:.2f}) abaixo do mínimo ({self.min_risk_reward})"
            )

        # 4. Validar stop loss
        stop_loss_pct = abs(entry_price - stop_loss) / entry_price

        if stop_loss_pct > self.max_stop_loss:
            validations["stop_loss_check"] = "FAIL"
            validations["errors"].append(
                f"Stop loss ({stop_loss_pct:.2%}) excede máximo ({self.max_stop_loss:.0%})"
            )

        # Resultado final
        validations["is_valid"] = all(
            [
                validations["capital_check"] == "PASS",
                validations["risk_check"] == "PASS",
                validations["rr_ratio_check"] == "PASS",
                validations["stop_loss_check"] == "PASS",
            ]
        )

        return validations

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        capital_available: float,
        risk_percent: float | None = None,
    ) -> float:
        """
        Calcula tamanho da posição baseado no risco.

        Args:
            entry_price: Preço de entrada
            stop_loss: Stop loss
            capital_available: Capital disponível
            risk_percent: % de risco (usa self.max_risk_per_trade se None)

        Returns:
            Quantidade em USD
        """
        if risk_percent is None:
            risk_percent = self.max_risk_per_trade

        # Risco máximo em USD
        max_risk_usd = capital_available * risk_percent

        # Risco por unidade
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit == 0:
            return capital_available * 0.1  # Fallback: 10% do capital

        # Quantidade baseada no risco
        quantity_usd = (max_risk_usd / risk_per_unit) * entry_price

        # Limitar a 50% do capital
        quantity_usd = min(quantity_usd, capital_available * 0.5)

        return quantity_usd

    def record_trade(self, trade: dict[str, Any]):
        """Registra trade no histórico"""
        trade["timestamp"] = datetime.now().isoformat()
        trade["strategy"] = self.name

        self.trades_history.append(trade)

        # Atualizar métricas
        self.performance_metrics["total_trades"] += 1

        if trade.get("pnl", 0) > 0:
            self.performance_metrics["winning_trades"] += 1
        elif trade.get("pnl", 0) < 0:
            self.performance_metrics["losing_trades"] += 1

        self.performance_metrics["total_pnl"] += trade.get("pnl", 0)

        if self.performance_metrics["total_trades"] > 0:
            self.performance_metrics["win_rate"] = (
                self.performance_metrics["winning_trades"]
                / self.performance_metrics["total_trades"]
            )

    def get_performance_metrics(self) -> dict[str, Any]:
        """Retorna métricas de performance"""
        return {
            **self.performance_metrics,
            "strategy": self.name,
            "timeframe": self.timeframe,
            "recent_trades": self.trades_history[-10:] if self.trades_history else [],
        }

    def reset_performance(self):
        """Reseta métricas de performance"""
        self.trades_history = []
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
        }

        logger.info(f"Performance da estratégia '{self.name}' resetada")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"timeframe='{self.timeframe}', "
            f"trades={self.performance_metrics['total_trades']}, "
            f"win_rate={self.performance_metrics['win_rate']:.1%})"
        )
