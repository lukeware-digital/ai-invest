"""
CeciAI - Testes do Capital Management
Testa sistema de gestão de capital e risco

Autor: CeciAI Team
Data: 2025-10-08
"""

from datetime import datetime, timedelta

import pytest

from config.capital_management import CapitalManager


@pytest.fixture
def capital_manager():
    """Fixture que retorna um CapitalManager"""
    return CapitalManager(
        initial_capital=10000.0,
        max_daily_loss=0.03,
        max_position_size=0.20,
        max_concurrent_positions=5,
    )


class TestCapitalManagerInit:
    """Testes de inicialização"""

    def test_init_default_values(self):
        """Testa inicialização com valores padrão"""
        cm = CapitalManager()
        assert cm.initial_capital == 10000.0
        assert cm.current_capital == 10000.0
        assert cm.available_capital == 10000.0
        assert cm.max_daily_loss == 0.03
        assert cm.max_position_size == 0.20
        assert cm.max_concurrent_positions == 5

    def test_init_custom_values(self):
        """Testa inicialização com valores customizados"""
        cm = CapitalManager(
            initial_capital=50000.0,
            max_daily_loss=0.05,
            max_position_size=0.15,
            max_concurrent_positions=10,
        )
        assert cm.initial_capital == 50000.0
        assert cm.max_daily_loss == 0.05
        assert cm.max_position_size == 0.15
        assert cm.max_concurrent_positions == 10

    def test_init_state(self, capital_manager):
        """Testa estado inicial"""
        assert capital_manager.open_positions == []
        assert capital_manager.closed_positions == []
        assert capital_manager.daily_pnl == 0.0
        assert capital_manager.total_pnl == 0.0
        assert capital_manager.circuit_breaker_active is False
        assert capital_manager.consecutive_losses == 0


class TestCanOpenPosition:
    """Testes para verificação de abertura de posição"""

    def test_can_open_position_valid(self, capital_manager):
        """Testa abertura de posição válida"""
        result = capital_manager.can_open_position(1000.0)
        assert result["can_open"] is True
        assert len(result["reasons"]) == 0

    def test_can_open_position_insufficient_capital(self, capital_manager):
        """Testa com capital insuficiente"""
        result = capital_manager.can_open_position(15000.0)
        assert result["can_open"] is False
        assert any("Capital insuficiente" in r for r in result["reasons"])

    def test_can_open_position_exceeds_max_size(self, capital_manager):
        """Testa com tamanho excedendo máximo"""
        # max_position_size = 20% de 10000 = 2000
        result = capital_manager.can_open_position(3000.0)
        assert result["can_open"] is False
        assert any("excede" in r for r in result["reasons"])

    def test_can_open_position_max_concurrent(self, capital_manager):
        """Testa limite de posições simultâneas"""
        # Abrir 5 posições (máximo)
        for i in range(5):
            capital_manager.open_positions.append({"id": f"pos_{i}", "quantity_usd": 1000})

        result = capital_manager.can_open_position(1000.0)
        assert result["can_open"] is False
        assert any("Máximo de posições" in r for r in result["reasons"])

    def test_can_open_position_circuit_breaker_active(self, capital_manager):
        """Testa com circuit breaker ativo"""
        capital_manager.activate_circuit_breaker(duration_hours=1)

        result = capital_manager.can_open_position(1000.0)
        assert result["can_open"] is False
        assert any("Circuit breaker" in r for r in result["reasons"])

    def test_can_open_position_max_daily_loss(self, capital_manager):
        """Testa com perda diária máxima atingida"""
        # max_daily_loss = 3% de 10000 = 300
        capital_manager.daily_pnl = -300

        result = capital_manager.can_open_position(1000.0)
        assert result["can_open"] is False
        assert any("Perda diária máxima" in r for r in result["reasons"])


class TestOpenPosition:
    """Testes para abertura de posição"""

    def test_open_position_success(self, capital_manager):
        """Testa abertura de posição com sucesso"""
        trade = {
            "id": "test_trade_1",
            "symbol": "BTC/USD",
            "quantity_usd": 1000.0,
            "entry_price": 50000.0,
            "stop_loss": 49000.0,
            "take_profit": 52000.0,
            "strategy": "scalping",
            "signal": "BUY",
        }
        result = capital_manager.open_position(trade)

        assert result is True
        assert len(capital_manager.open_positions) == 1
        assert capital_manager.open_positions[0]["symbol"] == "BTC/USD"
        assert capital_manager.available_capital == 9000.0

    def test_open_position_invalid(self, capital_manager):
        """Testa abertura de posição inválida"""
        trade = {
            "id": "test_trade_invalid",
            "symbol": "BTC/USD",
            "quantity_usd": 15000.0,  # Excede capital disponível
            "entry_price": 50000.0,
            "signal": "BUY",
        }
        result = capital_manager.open_position(trade)

        assert result is False
        assert len(capital_manager.open_positions) == 0

    def test_open_position_updates_available_capital(self, capital_manager):
        """Testa que capital disponível é atualizado"""
        initial_available = capital_manager.available_capital

        trade = {
            "id": "test_trade_2",
            "symbol": "BTC/USD",
            "quantity_usd": 2000.0,
            "entry_price": 50000.0,
            "signal": "BUY",
        }
        capital_manager.open_position(trade)

        assert capital_manager.available_capital == initial_available - 2000.0


class TestClosePosition:
    """Testes para fechamento de posição"""

    def test_close_position_profit(self, capital_manager):
        """Testa fechamento com lucro"""
        # Abrir posição
        trade = {
            "id": "test_trade_profit",
            "symbol": "BTC/USD",
            "quantity_usd": 1000.0,
            "entry_price": 50000.0,
            "stop_loss": 49000.0,
            "take_profit": 52000.0,
            "signal": "BUY",
        }
        capital_manager.open_position(trade)

        # Fechar com lucro
        result = capital_manager.close_position(position_id="test_trade_profit", exit_price=52000.0)

        assert result["pnl"] > 0
        assert len(capital_manager.open_positions) == 0
        assert len(capital_manager.closed_positions) == 1
        assert capital_manager.current_capital > 10000.0
        assert capital_manager.consecutive_losses == 0

    def test_close_position_loss(self, capital_manager):
        """Testa fechamento com perda"""
        # Abrir posição
        trade = {
            "id": "test_trade_loss",
            "symbol": "BTC/USD",
            "quantity_usd": 1000.0,
            "entry_price": 50000.0,
            "signal": "BUY",
        }
        capital_manager.open_position(trade)

        # Fechar com perda
        result = capital_manager.close_position(position_id="test_trade_loss", exit_price=48000.0)

        assert result["pnl"] < 0
        assert capital_manager.current_capital < 10000.0
        assert capital_manager.consecutive_losses == 1

    def test_close_position_invalid_id(self, capital_manager):
        """Testa fechamento com ID inválido"""
        result = capital_manager.close_position(position_id="invalid_id", exit_price=50000.0)

        assert result["success"] is False

    def test_close_position_updates_pnl(self, capital_manager):
        """Testa que P&L é atualizado"""
        trade = {
            "id": "test_trade_pnl",
            "symbol": "BTC/USD",
            "quantity_usd": 1000.0,
            "entry_price": 50000.0,
            "signal": "BUY",
        }
        capital_manager.open_position(trade)

        initial_total_pnl = capital_manager.total_pnl

        capital_manager.close_position(position_id="test_trade_pnl", exit_price=51000.0)

        assert capital_manager.total_pnl != initial_total_pnl


class TestCircuitBreaker:
    """Testes para circuit breaker"""

    def test_activate_circuit_breaker(self, capital_manager):
        """Testa ativação do circuit breaker"""
        capital_manager.activate_circuit_breaker(duration_hours=2)

        assert capital_manager.circuit_breaker_active is True
        assert capital_manager.circuit_breaker_until is not None
        assert capital_manager.circuit_breaker_until > datetime.now()

    def test_deactivate_circuit_breaker(self, capital_manager):
        """Testa desativação do circuit breaker"""
        capital_manager.activate_circuit_breaker(duration_hours=1)
        capital_manager.deactivate_circuit_breaker()

        assert capital_manager.circuit_breaker_active is False
        assert capital_manager.circuit_breaker_until is None

    def test_circuit_breaker_auto_deactivate(self, capital_manager):
        """Testa desativação automática após expirar"""
        # Ativar com tempo negativo (já expirado)
        capital_manager.circuit_breaker_active = True
        capital_manager.circuit_breaker_until = datetime.now() - timedelta(hours=1)

        # Tentar abrir posição deve desativar automaticamente
        _ = capital_manager.can_open_position(1000.0)

        assert capital_manager.circuit_breaker_active is False

    def test_circuit_breaker_on_consecutive_losses(self, capital_manager):
        """Testa ativação automática após perdas consecutivas"""
        # Simular 3 perdas consecutivas
        for i in range(3):
            trade = {
                "id": f"test_trade_loss_{i}",
                "symbol": "BTC/USD",
                "quantity_usd": 1000.0,
                "entry_price": 50000.0,
                "signal": "BUY",
            }
            capital_manager.open_position(trade)
            capital_manager.close_position(
                position_id=f"test_trade_loss_{i}",
                exit_price=48000.0,  # Perda
            )

        # Circuit breaker deve estar ativo
        assert capital_manager.circuit_breaker_active is True


class TestGetStatus:
    """Testes para obtenção de status"""

    def test_get_status(self, capital_manager):
        """Testa obtenção de status"""
        status = capital_manager.get_status()

        assert "current_capital" in status
        assert "available_capital" in status
        assert "initial_capital" in status
        assert "open_positions" in status
        assert "daily_pnl" in status
        assert "total_pnl" in status
        assert "circuit_breaker_active" in status
        assert "consecutive_losses" in status

    def test_get_status_with_positions(self, capital_manager):
        """Testa status com posições abertas"""
        trade = {
            "id": "test_status",
            "symbol": "BTC/USD",
            "quantity_usd": 2000.0,
            "entry_price": 50000.0,
            "signal": "BUY",
        }
        capital_manager.open_position(trade)

        status = capital_manager.get_status()

        assert status["open_positions"] == 1
        assert status["available_capital"] == 8000.0


class TestGetOpenPositions:
    """Testes para obtenção de posições abertas"""

    def test_get_open_positions_empty(self, capital_manager):
        """Testa com nenhuma posição aberta"""
        positions = capital_manager.get_open_positions()
        assert len(positions) == 0

    def test_get_open_positions_with_data(self, capital_manager):
        """Testa com posições abertas"""
        trade1 = {
            "id": "test_btc",
            "symbol": "BTC/USD",
            "quantity_usd": 1000.0,
            "entry_price": 50000.0,
            "signal": "BUY",
        }
        trade2 = {
            "id": "test_eth",
            "symbol": "ETH/USD",
            "quantity_usd": 1500.0,
            "entry_price": 3000.0,
            "signal": "BUY",
        }
        capital_manager.open_position(trade1)
        capital_manager.open_position(trade2)

        positions = capital_manager.get_open_positions()
        assert len(positions) == 2

    def test_get_open_positions_by_symbol(self, capital_manager):
        """Testa filtragem por símbolo"""
        trade1 = {
            "id": "test_btc_filter",
            "symbol": "BTC/USD",
            "quantity_usd": 1000.0,
            "entry_price": 50000.0,
            "signal": "BUY",
        }
        trade2 = {
            "id": "test_eth_filter",
            "symbol": "ETH/USD",
            "quantity_usd": 1500.0,
            "entry_price": 3000.0,
            "signal": "BUY",
        }
        capital_manager.open_position(trade1)
        capital_manager.open_position(trade2)

        btc_positions = capital_manager.get_open_positions(symbol="BTC/USD")
        assert len(btc_positions) == 1
        assert btc_positions[0]["symbol"] == "BTC/USD"


class TestGetClosedPositions:
    """Testes para obtenção de posições fechadas"""

    def test_get_closed_positions_empty(self, capital_manager):
        """Testa com nenhuma posição fechada"""
        positions = capital_manager.get_closed_positions()
        assert len(positions) == 0

    def test_get_closed_positions_with_data(self, capital_manager):
        """Testa com posições fechadas"""
        # Abrir e fechar posições
        for i in range(3):
            trade = {
                "id": f"test_closed_{i}",
                "symbol": "BTC/USD",
                "quantity_usd": 1000.0,
                "entry_price": 50000.0,
                "signal": "BUY",
            }
            capital_manager.open_position(trade)
            capital_manager.close_position(position_id=f"test_closed_{i}", exit_price=51000.0)

        positions = capital_manager.get_closed_positions()
        assert len(positions) == 3

    def test_get_closed_positions_with_limit(self, capital_manager):
        """Testa com limite de resultados"""
        # Abrir e fechar 5 posições
        for i in range(5):
            trade = {
                "id": f"test_limit_{i}",
                "symbol": "BTC/USD",
                "quantity_usd": 1000.0,
                "entry_price": 50000.0,
                "signal": "BUY",
            }
            capital_manager.open_position(trade)
            capital_manager.close_position(position_id=f"test_limit_{i}", exit_price=51000.0)

        positions = capital_manager.get_closed_positions(limit=3)
        assert len(positions) == 3


class TestResetDaily:
    """Testes para reset diário"""

    def test_reset_daily(self, capital_manager):
        """Testa reset diário"""
        # Simular P&L diário
        capital_manager.daily_pnl = 150.0
        capital_manager.consecutive_losses = 2

        capital_manager.reset_daily()

        assert capital_manager.daily_pnl == 0.0
        assert capital_manager.consecutive_losses == 0


class TestCalculatePositionSize:
    """Testes para cálculo de tamanho de posição"""

    def test_calculate_position_size(self, capital_manager):
        """Testa cálculo de tamanho de posição"""
        size = capital_manager.calculate_position_size(
            entry_price=50000.0, stop_loss=49000.0, risk_percent=0.01
        )

        assert size > 0
        assert size <= capital_manager.current_capital * capital_manager.max_position_size

    def test_calculate_position_size_high_risk(self, capital_manager):
        """Testa com risco alto"""
        size = capital_manager.calculate_position_size(
            entry_price=50000.0,
            stop_loss=45000.0,  # 10% de stop loss
            risk_percent=0.01,
        )

        # Tamanho deve ser menor devido ao stop loss largo
        assert size > 0

    def test_calculate_position_size_invalid_stop(self, capital_manager):
        """Testa com stop loss inválido"""
        size = capital_manager.calculate_position_size(
            entry_price=50000.0,
            stop_loss=55000.0,  # Stop acima do entry (inválido)
            risk_percent=0.01,
        )

        assert size == 0


class TestEdgeCases:
    """Testes para casos extremos"""

    def test_zero_capital(self):
        """Testa com capital zero"""
        cm = CapitalManager(initial_capital=0.0)

        result = cm.can_open_position(1000.0)
        assert result["can_open"] is False

    def test_negative_pnl_extreme(self, capital_manager):
        """Testa com P&L negativo extremo"""
        # Simular perda total
        capital_manager.current_capital = 0.0
        capital_manager.available_capital = 0.0

        result = capital_manager.can_open_position(100.0)
        assert result["can_open"] is False

    def test_multiple_positions_same_symbol(self, capital_manager):
        """Testa múltiplas posições no mesmo símbolo"""
        trade1 = {
            "id": "test_multi_1",
            "symbol": "BTC/USD",
            "quantity_usd": 1000.0,
            "entry_price": 50000.0,
            "signal": "BUY",
        }
        trade2 = {
            "id": "test_multi_2",
            "symbol": "BTC/USD",
            "quantity_usd": 1000.0,
            "entry_price": 50100.0,
            "signal": "BUY",
        }
        capital_manager.open_position(trade1)
        capital_manager.open_position(trade2)

        positions = capital_manager.get_open_positions(symbol="BTC/USD")
        assert len(positions) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=config.capital_management", "--cov-report=term-missing"])
