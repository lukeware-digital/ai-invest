"""
CeciAI - Capital Management System
Sistema completo de gestão de capital e risco

Responsabilidades:
- Tracking de capital disponível
- Alocação por trade
- Circuit breaker
- Gestão de risco

Autor: CeciAI Team
Data: 2025-10-08
"""

import logging
from datetime import datetime, timedelta
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CapitalManager:
    """
    Gerenciador de Capital

    Features:
    - Tracking de capital total e disponível
    - Histórico de trades
    - Circuit breaker automático
    - Limites de risco
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_daily_loss: float = 0.03,
        max_position_size: float = 0.20,
        max_concurrent_positions: int = 5,
    ):
        """
        Inicializa o gerenciador.

        Args:
            initial_capital: Capital inicial
            max_daily_loss: Perda máxima diária (% do capital)
            max_position_size: Tamanho máximo de posição (% do capital)
            max_concurrent_positions: Número máximo de posições simultâneas
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_capital = initial_capital

        self.max_daily_loss = max_daily_loss
        self.max_position_size = max_position_size
        self.max_concurrent_positions = max_concurrent_positions

        self.open_positions = []
        self.closed_positions = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0

        self.circuit_breaker_active = False
        self.circuit_breaker_until = None

        self.consecutive_losses = 0
        self.max_consecutive_losses = 3

        logger.info(f"Capital Manager inicializado com ${initial_capital:,.2f}")

    def can_open_position(self, quantity_usd: float) -> dict[str, Any]:
        """
        Verifica se pode abrir nova posição.

        Args:
            quantity_usd: Quantidade em USD

        Returns:
            Dict com validação
        """
        result = {"can_open": True, "reasons": [], "warnings": []}

        # 1. Circuit breaker
        if self.circuit_breaker_active:
            if datetime.now() < self.circuit_breaker_until:
                result["can_open"] = False
                result["reasons"].append(f"Circuit breaker ativo até {self.circuit_breaker_until}")
                return result
            else:
                self.deactivate_circuit_breaker()

        # 2. Perda diária máxima
        if abs(self.daily_pnl) >= self.current_capital * self.max_daily_loss:
            result["can_open"] = False
            result["reasons"].append(f"Perda diária máxima atingida ({self.daily_pnl:.2f})")
            return result

        # 3. Posições simultâneas
        if len(self.open_positions) >= self.max_concurrent_positions:
            result["can_open"] = False
            result["reasons"].append(
                f"Máximo de posições simultâneas atingido ({len(self.open_positions)})"
            )
            return result

        # 4. Capital disponível
        if quantity_usd > self.available_capital:
            result["can_open"] = False
            result["reasons"].append(
                f"Capital insuficiente (disponível: ${self.available_capital:,.2f})"
            )
            return result

        # 5. Tamanho da posição
        if quantity_usd > self.current_capital * self.max_position_size:
            result["can_open"] = False
            result["reasons"].append(f"Posição excede {self.max_position_size:.0%} do capital")
            return result

        # Warnings
        if quantity_usd > self.current_capital * 0.15:
            result["warnings"].append(
                f"Posição grande: {quantity_usd/self.current_capital:.1%} do capital"
            )

        return result

    def open_position(self, trade: dict[str, Any]) -> bool:
        """
        Abre nova posição.

        Args:
            trade: Dados do trade

        Returns:
            True se aberto com sucesso
        """
        quantity_usd = trade.get("quantity_usd", 0)

        # Validar
        validation = self.can_open_position(quantity_usd)
        if not validation["can_open"]:
            logger.warning(f"Não pode abrir posição: {validation['reasons']}")
            return False

        # Reservar capital
        self.available_capital -= quantity_usd

        # Adicionar à lista
        trade["opened_at"] = datetime.now()
        trade["status"] = "open"
        self.open_positions.append(trade)

        logger.info(f"✅ Posição aberta: {trade['symbol']} ${quantity_usd:,.2f}")
        logger.info(f"Capital disponível: ${self.available_capital:,.2f}")

        return True

    def close_position(
        self, position_id: str, exit_price: float, reason: str = "manual"
    ) -> dict[str, Any]:
        """
        Fecha posição.

        Args:
            position_id: ID da posição
            exit_price: Preço de saída
            reason: Razão do fechamento

        Returns:
            Dict com resultado
        """
        # Encontrar posição
        position = None
        for i, pos in enumerate(self.open_positions):
            if pos.get("id") == position_id:
                position = self.open_positions.pop(i)
                break

        if position is None:
            logger.error(f"Posição {position_id} não encontrada")
            return {"success": False, "error": "Position not found"}

        # Calcular P&L
        entry_price = position["entry_price"]
        quantity_usd = position["quantity_usd"]

        if position["signal"] == "BUY":
            pnl = (exit_price - entry_price) / entry_price * quantity_usd
        else:  # SELL
            pnl = (entry_price - exit_price) / entry_price * quantity_usd

        pnl_pct = pnl / quantity_usd

        # Liberar capital
        self.available_capital += quantity_usd + pnl
        self.current_capital += pnl

        # Atualizar P&L
        self.daily_pnl += pnl
        self.total_pnl += pnl

        # Atualizar posição
        position["closed_at"] = datetime.now()
        position["exit_price"] = exit_price
        position["pnl"] = pnl
        position["pnl_pct"] = pnl_pct
        position["close_reason"] = reason
        position["status"] = "closed"

        self.closed_positions.append(position)

        # Verificar circuit breaker
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.activate_circuit_breaker()
        else:
            self.consecutive_losses = 0

        logger.info(f"✅ Posição fechada: {position['symbol']} P&L: ${pnl:,.2f} ({pnl_pct:+.2%})")

        return {"success": True, "position": position, "pnl": pnl, "pnl_pct": pnl_pct}

    def activate_circuit_breaker(self, duration_hours: int = 1):
        """Ativa circuit breaker"""
        self.circuit_breaker_active = True
        self.circuit_breaker_until = datetime.now() + timedelta(hours=duration_hours)

        logger.warning(f"⚠️  CIRCUIT BREAKER ATIVADO até {self.circuit_breaker_until}")
        logger.warning(f"Perdas consecutivas: {self.consecutive_losses}")

    def deactivate_circuit_breaker(self):
        """Desativa circuit breaker"""
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None
        self.consecutive_losses = 0

        logger.info("✅ Circuit breaker desativado")

    def reset_daily_pnl(self):
        """Reseta P&L diário (chamar a cada novo dia)"""
        logger.info(f"Resetando P&L diário (anterior: ${self.daily_pnl:,.2f})")
        self.daily_pnl = 0.0

    def reset_daily(self):
        """Alias para reset_daily_pnl - reseta P&L e perdas consecutivas"""
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        logger.info("Reset diário completo")

    def get_status(self) -> dict[str, Any]:
        """Retorna status do capital"""
        return {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "available_capital": self.available_capital,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl / self.initial_capital,
            "daily_pnl": self.daily_pnl,
            "open_positions": len(self.open_positions),
            "closed_positions": len(self.closed_positions),
            "circuit_breaker_active": self.circuit_breaker_active,
            "consecutive_losses": self.consecutive_losses,
        }

    def get_performance_metrics(self) -> dict[str, Any]:
        """Calcula métricas de performance"""
        if not self.closed_positions:
            return {"total_trades": 0}

        winning_trades = [p for p in self.closed_positions if p["pnl"] > 0]
        losing_trades = [p for p in self.closed_positions if p["pnl"] < 0]

        total_wins = sum(p["pnl"] for p in winning_trades)
        total_losses = sum(abs(p["pnl"]) for p in losing_trades)

        return {
            "total_trades": len(self.closed_positions),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(self.closed_positions),
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl / self.initial_capital,
            "avg_win": total_wins / len(winning_trades) if winning_trades else 0,
            "avg_loss": total_losses / len(losing_trades) if losing_trades else 0,
            "profit_factor": total_wins / total_losses if total_losses > 0 else 0,
            "largest_win": max((p["pnl"] for p in winning_trades), default=0),
            "largest_loss": min((p["pnl"] for p in losing_trades), default=0),
        }

    def get_open_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """
        Retorna posições abertas.

        Args:
            symbol: Filtrar por símbolo (opcional)

        Returns:
            Lista de posições abertas
        """
        if symbol:
            return [p for p in self.open_positions if p.get("symbol") == symbol]
        return self.open_positions.copy()

    def get_closed_positions(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Retorna posições fechadas.

        Args:
            limit: Limite de resultados (opcional)

        Returns:
            Lista de posições fechadas
        """
        positions = self.closed_positions.copy()
        if limit:
            return positions[-limit:]
        return positions

    def calculate_position_size(
        self, entry_price: float, stop_loss: float, risk_percent: float = 0.01
    ) -> float:
        """
        Calcula tamanho de posição baseado em risco.

        Args:
            entry_price: Preço de entrada
            stop_loss: Preço de stop loss
            risk_percent: Percentual de risco (padrão 1%)

        Returns:
            Tamanho da posição em USD
        """
        # Validar stop loss
        if stop_loss >= entry_price:
            logger.warning("Stop loss inválido (deve ser menor que entry)")
            return 0.0

        # Calcular risco por unidade
        risk_per_unit = entry_price - stop_loss
        risk_pct = risk_per_unit / entry_price

        # Capital a arriscar
        capital_at_risk = self.current_capital * risk_percent

        # Tamanho da posição
        position_size = capital_at_risk / risk_pct

        # Limitar ao máximo permitido
        max_size = self.current_capital * self.max_position_size
        position_size = min(position_size, max_size)

        # Limitar ao capital disponível
        position_size = min(position_size, self.available_capital)

        return position_size
