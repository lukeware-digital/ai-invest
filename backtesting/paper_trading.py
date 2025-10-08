"""
CeciAI - Paper Trading Engine
Sistema de paper trading para testes em tempo real

Autor: CeciAI Team
Data: 2025-10-08
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from config.capital_management import CapitalManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """
    Motor de Paper Trading

    Features:
    - Simula trades em tempo real
    - Usa dados reais mas sem dinheiro real
    - Tracking de performance
    - Monitoramento de posições abertas
    - Execução automática de stop loss e take profit
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.capital_manager = CapitalManager(initial_capital=initial_capital)
        self.is_running = False
        self.monitoring_task = None
        logger.info(f"Paper Trading Engine inicializado com ${initial_capital:,.2f}")

    async def start(self, coinapi_client=None, check_interval: int = 60):
        """
        Inicia paper trading com monitoramento automático.

        Args:
            coinapi_client: Cliente CoinAPI para buscar preços atuais
            check_interval: Intervalo de verificação em segundos (padrão: 60s)
        """
        self.is_running = True
        self.coinapi_client = coinapi_client
        logger.info("📈 Paper Trading iniciado")

        # Iniciar task de monitoramento
        if coinapi_client:
            self.monitoring_task = asyncio.create_task(
                self._monitor_positions(check_interval)
            )
            logger.info(f"🔍 Monitoramento de posições ativo (intervalo: {check_interval}s)")

    async def stop(self):
        """Para paper trading e cancela monitoramento"""
        self.is_running = False

        # Cancelar task de monitoramento
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("⏸️  Paper Trading pausado")

    async def _monitor_positions(self, check_interval: int):
        """
        Monitora posições abertas e executa stop loss / take profit automaticamente.

        Args:
            check_interval: Intervalo de verificação em segundos
        """
        while self.is_running:
            try:
                open_positions = self.capital_manager.get_open_positions()

                for position in open_positions:
                    try:
                        # Buscar preço atual
                        symbol = position.get("symbol")
                        if not symbol or not self.coinapi_client:
                            continue

                        current_price = await self.coinapi_client.get_latest_price(symbol)

                        # Verificar stop loss
                        stop_loss = position.get("stop_loss", {})
                        if isinstance(stop_loss, dict):
                            stop_price = stop_loss.get("price", 0)
                        else:
                            stop_price = stop_loss

                        if current_price <= stop_price:
                            logger.warning(
                                f"🛑 Stop Loss ativado: {symbol} @ ${current_price:,.2f}"
                            )
                            result = self.capital_manager.close_position(
                                position_id=position.get("id", ""),
                                exit_price=current_price,
                                reason="stop_loss",
                            )
                            if result.get("success"):
                                logger.info(
                                    f"Posição fechada: P&L ${result['pnl']:,.2f} ({result['pnl_pct']:+.2%})"
                                )
                            continue

                        # Verificar take profit
                        take_profit_1 = position.get("take_profit_1", {})
                        if isinstance(take_profit_1, dict):
                            tp1_price = take_profit_1.get("price", float("inf"))
                        else:
                            tp1_price = take_profit_1 if take_profit_1 else float("inf")

                        if current_price >= tp1_price:
                            logger.info(
                                f"🎯 Take Profit ativado: {symbol} @ ${current_price:,.2f}"
                            )
                            result = self.capital_manager.close_position(
                                position_id=position.get("id", ""),
                                exit_price=current_price,
                                reason="take_profit",
                            )
                            if result.get("success"):
                                logger.info(
                                    f"Posição fechada: P&L ${result['pnl']:,.2f} ({result['pnl_pct']:+.2%})"
                                )

                    except Exception as e:
                        logger.error(f"Erro ao monitorar posição {position.get('symbol')}: {e}")

                # Aguardar próxima verificação
                await asyncio.sleep(check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro no monitoramento: {e}")
                await asyncio.sleep(check_interval)

    async def execute_trade(self, trade: dict[str, Any]) -> dict[str, Any]:
        """
        Executa trade simulado.

        Args:
            trade: Dados do trade (deve conter: symbol, signal, entry_price, quantity_usd, stop_loss, take_profit_1)

        Returns:
            Resultado da execução
        """
        if not self.is_running:
            return {"success": False, "error": "Paper trading não está rodando"}

        # Adicionar ID único se não houver
        if "id" not in trade:
            import uuid
            trade["id"] = str(uuid.uuid4())[:8]

        # Verificar se pode abrir
        validation = self.capital_manager.can_open_position(trade["quantity_usd"])

        if not validation["can_open"]:
            logger.warning(f"Trade rejeitado: {validation['reasons']}")
            return {"success": False, "reasons": validation["reasons"]}

        # Abrir posição
        success = self.capital_manager.open_position(trade)

        if success:
            logger.info(
                f"✅ Trade executado: {trade['signal']} {trade['symbol']} @ ${trade['entry_price']:,.2f} | "
                f"Quantidade: ${trade['quantity_usd']:,.2f}"
            )

        return {
            "success": success,
            "trade": trade,
            "capital_status": self.capital_manager.get_status(),
        }

    async def close_position(self, position_id: str, exit_price: float, reason: str = "manual") -> dict[str, Any]:
        """
        Fecha uma posição aberta.

        Args:
            position_id: ID da posição
            exit_price: Preço de saída
            reason: Razão do fechamento

        Returns:
            Resultado do fechamento
        """
        if not self.is_running:
            return {"success": False, "error": "Paper trading não está rodando"}

        result = self.capital_manager.close_position(position_id, exit_price, reason)

        if result.get("success"):
            position = result.get("position", {})
            logger.info(
                f"✅ Posição fechada: {position.get('symbol')} | "
                f"P&L: ${result['pnl']:,.2f} ({result['pnl_pct']:+.2%})"
            )

        return result

    def get_status(self) -> dict[str, Any]:
        """Retorna status completo do paper trading"""
        capital_status = self.capital_manager.get_status()
        performance = self.capital_manager.get_performance_metrics()

        return {
            "is_running": self.is_running,
            "timestamp": datetime.now().isoformat(),
            "capital": capital_status,
            "performance": performance,
            "open_positions": self.capital_manager.get_open_positions(),
            "recent_trades": self.capital_manager.get_closed_positions(limit=10),
        }

    def get_open_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Retorna posições abertas, opcionalmente filtradas por símbolo"""
        return self.capital_manager.get_open_positions(symbol)

    def get_performance_metrics(self) -> dict[str, Any]:
        """Retorna métricas de performance"""
        return self.capital_manager.get_performance_metrics()
