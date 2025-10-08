"""
CeciAI - Backtest Engine
Motor de backtesting para validação histórica

Autor: CeciAI Team
Data: 2025-10-08
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Motor de Backtesting

    Features:
    - Simula trades em dados históricos
    - Calcula métricas de performance
    - Gera relatórios detalhados
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades = []
        self.equity_curve = []

    async def run(
        self,
        df: pd.DataFrame,
        strategy,
        agent_pipeline,
        symbol: str = "BTC/USD",
        risk_percent: float = 0.01,
    ) -> dict[str, Any]:
        """
        Executa backtest.

        Args:
            df: Dados históricos
            strategy: Estratégia a testar
            agent_pipeline: Pipeline de agentes
            symbol: Par de trading
            risk_percent: Percentual de risco por trade

        Returns:
            Resultados do backtest
        """
        logger.info(f"🚀 Iniciando backtest com {len(df)} candles para {symbol}")

        # Resetar capital e trades
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = []

        # Posição aberta atual
        open_position = None

        # Começar após 60 candles para ter dados suficientes para LSTM e indicadores
        for i in range(60, len(df)):
            try:
                # Dados até o candle atual
                window_df = df.iloc[: i + 1].copy()
                current_price = float(df["close"].iloc[i])
                current_timestamp = df["timestamp"].iloc[i] if "timestamp" in df.columns else i

                # Verificar se há posição aberta para fechar
                if open_position:
                    # Verificar stop loss
                    if open_position["signal"] == "BUY":
                        if current_price <= open_position["stop_loss"]:
                            # Stop loss atingido
                            pnl = (
                                current_price - open_position["entry_price"]
                            ) / open_position["entry_price"] * open_position["quantity_usd"]
                            self.capital += open_position["quantity_usd"] + pnl

                            trade_result = {
                                **open_position,
                                "exit_price": current_price,
                                "exit_timestamp": current_timestamp,
                                "exit_reason": "stop_loss",
                                "pnl": pnl,
                                "pnl_pct": pnl / open_position["quantity_usd"],
                            }
                            self.trades.append(trade_result)
                            logger.info(
                                f"❌ Stop Loss: {symbol} @ ${current_price:,.2f} | P&L: ${pnl:,.2f}"
                            )
                            open_position = None

                        # Verificar take profit
                        elif (
                            "take_profit_1" in open_position
                            and current_price >= open_position["take_profit_1"]
                        ):
                            # Take profit atingido
                            pnl = (
                                current_price - open_position["entry_price"]
                            ) / open_position["entry_price"] * open_position["quantity_usd"]
                            self.capital += open_position["quantity_usd"] + pnl

                            trade_result = {
                                **open_position,
                                "exit_price": current_price,
                                "exit_timestamp": current_timestamp,
                                "exit_reason": "take_profit",
                                "pnl": pnl,
                                "pnl_pct": pnl / open_position["quantity_usd"],
                            }
                            self.trades.append(trade_result)
                            logger.info(
                                f"✅ Take Profit: {symbol} @ ${current_price:,.2f} | P&L: ${pnl:,.2f}"
                            )
                            open_position = None

                # Se não há posição aberta, verificar novas oportunidades
                if not open_position and self.capital > 0:
                    try:
                        # Executar pipeline de análise
                        pipeline_result = await agent_pipeline.execute(
                            df=window_df,
                            symbol=symbol,
                            timeframe="1h",
                            capital_available=self.capital,
                            user_strategy=strategy.name if hasattr(strategy, "name") else "scalping",
                            risk_params={
                                "max_risk_per_trade": risk_percent,
                                "min_risk_reward": 1.5,
                                "max_stop_loss": 0.03,
                            },
                        )

                        # Verificar decisão
                        decision = pipeline_result.get("final_decision", {})

                        if decision.get("decision") == "BUY" and decision.get("confidence", 0) > 0.6:
                            # Abrir posição de compra
                            quantity_usd = min(
                                decision.get("quantity_usd", self.capital * 0.1), self.capital * 0.2
                            )

                            if quantity_usd > 100:  # Mínimo de $100 por trade
                                open_position = {
                                    "symbol": symbol,
                                    "signal": "BUY",
                                    "entry_price": current_price,
                                    "entry_timestamp": current_timestamp,
                                    "quantity_usd": quantity_usd,
                                    "stop_loss": decision.get("stop_loss", current_price * 0.98),
                                    "take_profit_1": decision.get(
                                        "take_profit_1", current_price * 1.02
                                    ),
                                    "take_profit_2": decision.get(
                                        "take_profit_2", current_price * 1.05
                                    ),
                                    "confidence": decision.get("confidence", 0),
                                    "opportunity_score": decision.get("opportunity_score", 0),
                                }

                                self.capital -= quantity_usd

                                logger.info(
                                    f"🟢 Abertura BUY: {symbol} @ ${current_price:,.2f} | "
                                    f"Quantidade: ${quantity_usd:,.2f} | "
                                    f"Score: {decision.get('opportunity_score', 0)}/100"
                                )

                    except Exception as e:
                        logger.error(f"Erro ao processar candle {i}: {e}")

                # Registrar equity
                equity = self.capital
                if open_position:
                    # Adicionar P&L não realizado
                    unrealized_pnl = (
                        current_price - open_position["entry_price"]
                    ) / open_position["entry_price"] * open_position["quantity_usd"]
                    equity += open_position["quantity_usd"] + unrealized_pnl

                self.equity_curve.append({"timestamp": current_timestamp, "equity": equity})

            except Exception as e:
                logger.error(f"Erro no backtest no candle {i}: {e}")
                continue

        # Fechar posição aberta no final
        if open_position:
            final_price = float(df["close"].iloc[-1])
            pnl = (
                final_price - open_position["entry_price"]
            ) / open_position["entry_price"] * open_position["quantity_usd"]
            self.capital += open_position["quantity_usd"] + pnl

            trade_result = {
                **open_position,
                "exit_price": final_price,
                "exit_timestamp": df["timestamp"].iloc[-1] if "timestamp" in df.columns else len(df),
                "exit_reason": "end_of_period",
                "pnl": pnl,
                "pnl_pct": pnl / open_position["quantity_usd"],
            }
            self.trades.append(trade_result)

        # Calcular métricas
        metrics = self._calculate_metrics()

        logger.info(f"✅ Backtest concluído: {len(self.trades)} trades executados")
        logger.info(f"Capital inicial: ${self.initial_capital:,.2f}")
        logger.info(f"Capital final: ${self.capital:,.2f}")
        logger.info(
            f"Retorno total: {((self.capital - self.initial_capital) / self.initial_capital) * 100:.2f}%"
        )
        if metrics:
            logger.info(f"Win rate: {metrics.get('win_rate', 0):.1%}")
            logger.info(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")

        return {
            "initial_capital": self.initial_capital,
            "final_capital": self.capital,
            "total_return": (self.capital - self.initial_capital) / self.initial_capital,
            "total_return_pct": ((self.capital - self.initial_capital) / self.initial_capital) * 100,
            "trades": self.trades,
            "metrics": metrics,
            "equity_curve": self.equity_curve,
        }

    def _calculate_metrics(self) -> dict[str, Any]:
        """Calcula métricas de performance"""
        if not self.trades:
            return {}

        winning_trades = [t for t in self.trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in self.trades if t.get("pnl", 0) < 0]

        returns = [t.get("pnl", 0) / self.initial_capital for t in self.trades]

        return {
            "total_trades": len(self.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(self.trades) if self.trades else 0,
            "sharpe_ratio": np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            "max_drawdown": self._calculate_max_drawdown(),
        }

    def _calculate_max_drawdown(self) -> float:
        """Calcula maximum drawdown"""
        if not self.equity_curve:
            return 0.0

        equity_values = [e["equity"] for e in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0.0

        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd
