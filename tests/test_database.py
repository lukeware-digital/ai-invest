"""
CeciAI - Testes do Database Manager
Testa gerenciamento de banco de dados com mocks

Autor: CeciAI Team
Data: 2025-10-08
"""

import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from utils.database import AsyncDatabaseManager, DatabaseManager


@pytest.fixture
def temp_db_path():
    """Fixture que cria um banco temporário"""
    # Criar um path temporário sem criar o arquivo
    fd, db_path = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)
    os.unlink(db_path)  # Remover o arquivo vazio para DuckDB criar corretamente
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        try:
            os.unlink(db_path)
        except Exception:
            pass


@pytest.fixture
def db_manager(temp_db_path):
    """Fixture que retorna um DatabaseManager"""
    with patch("redis.Redis") as mock_redis:
        mock_redis.return_value.ping.side_effect = Exception("Redis not available")
        db = DatabaseManager(db_path=temp_db_path, use_redis=False)
        yield db
        db.close()


@pytest.fixture
def sample_ohlcv_data():
    """Fixture com dados OHLCV de exemplo"""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1H"),
            "open": [50000 + i * 100 for i in range(10)],
            "high": [50500 + i * 100 for i in range(10)],
            "low": [49500 + i * 100 for i in range(10)],
            "close": [50300 + i * 100 for i in range(10)],
            "volume": [1000000 + i * 10000 for i in range(10)],
        }
    )


class TestDatabaseManagerInit:
    """Testes de inicialização"""

    def test_init_creates_tables(self, temp_db_path):
        """Testa que inicialização cria tabelas"""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value.ping.side_effect = Exception("Redis not available")
            db = DatabaseManager(db_path=temp_db_path, use_redis=False)

            # Verificar que tabelas existem
            tables = db.conn.execute("SHOW TABLES").fetchall()
            table_names = [t[0] for t in tables]

            assert "ohlcv_data" in table_names
            assert "trades" in table_names
            assert "analyses" in table_names
            assert "capital_history" in table_names

            db.close()

    def test_init_with_redis_unavailable(self, temp_db_path):
        """Testa inicialização quando Redis não está disponível"""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value.ping.side_effect = Exception("Connection refused")
            db = DatabaseManager(db_path=temp_db_path, use_redis=True)

            assert db.use_redis is False
            assert db.redis_client is None

            db.close()

    def test_init_with_redis_available(self, temp_db_path):
        """Testa inicialização quando Redis está disponível"""
        with patch("redis.Redis") as mock_redis:
            mock_redis_instance = MagicMock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True

            db = DatabaseManager(db_path=temp_db_path, use_redis=True)

            assert db.use_redis is True
            assert db.redis_client is not None

            db.close()


class TestDatabaseManagerOHLCV:
    """Testes para operações OHLCV"""

    def test_save_ohlcv_data(self, db_manager, sample_ohlcv_data):
        """Testa salvamento de dados OHLCV"""
        rows = db_manager.save_ohlcv_data("BTC/USD", "1h", sample_ohlcv_data)

        assert rows == len(sample_ohlcv_data)

        # Verificar que dados foram salvos
        result = db_manager.conn.execute(
            "SELECT COUNT(*) FROM ohlcv_data WHERE symbol = 'BTC/USD'"
        ).fetchone()
        assert result[0] == len(sample_ohlcv_data)

    def test_get_ohlcv_data(self, db_manager, sample_ohlcv_data):
        """Testa busca de dados OHLCV"""
        # Salvar dados primeiro
        db_manager.save_ohlcv_data("BTC/USD", "1h", sample_ohlcv_data)

        # Buscar dados
        result = db_manager.get_ohlcv_data("BTC/USD", "1h", limit=100)

        assert not result.empty
        assert len(result) == len(sample_ohlcv_data)
        assert "timestamp" in result.columns
        assert "close" in result.columns

    def test_get_ohlcv_data_with_date_range(self, db_manager, sample_ohlcv_data):
        """Testa busca com filtro de data"""
        db_manager.save_ohlcv_data("BTC/USD", "1h", sample_ohlcv_data)

        start_date = sample_ohlcv_data["timestamp"].iloc[2]
        end_date = sample_ohlcv_data["timestamp"].iloc[7]

        result = db_manager.get_ohlcv_data(
            "BTC/USD", "1h", start_date=start_date, end_date=end_date
        )

        assert not result.empty
        assert len(result) <= 6  # 5 candles no range

    def test_get_latest_ohlcv(self, db_manager, sample_ohlcv_data):
        """Testa busca dos últimos candles"""
        db_manager.save_ohlcv_data("BTC/USD", "1h", sample_ohlcv_data)

        result = db_manager.get_latest_ohlcv("BTC/USD", "1h", count=5)

        assert not result.empty
        assert len(result) <= 5

    def test_save_ohlcv_data_upsert(self, db_manager, sample_ohlcv_data):
        """Testa que dados duplicados são substituídos (upsert)"""
        # Salvar primeira vez
        db_manager.save_ohlcv_data("BTC/USD", "1h", sample_ohlcv_data)

        # Modificar dados e salvar novamente
        modified_data = sample_ohlcv_data.copy()
        modified_data["close"] = modified_data["close"] + 1000
        db_manager.save_ohlcv_data("BTC/USD", "1h", modified_data)

        # Verificar que não duplicou
        result = db_manager.conn.execute(
            "SELECT COUNT(*) FROM ohlcv_data WHERE symbol = 'BTC/USD'"
        ).fetchone()
        assert result[0] == len(sample_ohlcv_data)


class TestDatabaseManagerTrades:
    """Testes para operações de trades"""

    def test_save_trade(self, db_manager):
        """Testa salvamento de trade"""
        trade_data = {
            "trade_id": "trade_001",
            "symbol": "BTC/USD",
            "action": "BUY",
            "entry_price": 50000,
            "quantity": 0.1,
            "stop_loss": 49000,
            "take_profit": 52000,
            "status": "open",
            "strategy": "scalping",
            "entry_time": datetime.utcnow(),
        }

        trade_id = db_manager.save_trade(trade_data)
        assert trade_id == "trade_001"

        # Verificar que foi salvo
        result = db_manager.conn.execute(
            "SELECT * FROM trades WHERE trade_id = ?", ["trade_001"]
        ).fetchone()
        assert result is not None

    def test_update_trade(self, db_manager):
        """Testa atualização de trade"""
        # Criar trade
        trade_data = {
            "trade_id": "trade_002",
            "symbol": "BTC/USD",
            "action": "BUY",
            "entry_price": 50000,
            "quantity": 0.1,
            "status": "open",
            "strategy": "scalping",
            "entry_time": datetime.utcnow(),
        }
        db_manager.save_trade(trade_data)

        # Atualizar trade
        db_manager.update_trade(
            trade_id="trade_002", exit_price=52000, pnl=200, pnl_percent=4.0, status="closed"
        )

        # Verificar atualização
        result = db_manager.conn.execute(
            "SELECT status, exit_price, pnl FROM trades WHERE trade_id = ?", ["trade_002"]
        ).fetchone()
        assert result[0] == "closed"
        assert result[1] == 52000
        assert result[2] == 200

    def test_get_trades(self, db_manager):
        """Testa busca de trades"""
        # Criar múltiplos trades
        for i in range(5):
            trade_data = {
                "trade_id": f"trade_{i:03d}",
                "symbol": "BTC/USD",
                "action": "BUY",
                "entry_price": 50000 + i * 100,
                "quantity": 0.1,
                "status": "open" if i % 2 == 0 else "closed",
                "strategy": "scalping",
                "entry_time": datetime.utcnow(),
            }
            db_manager.save_trade(trade_data)

        # Buscar todos
        all_trades = db_manager.get_trades(limit=100)
        assert len(all_trades) == 5

        # Buscar por símbolo
        btc_trades = db_manager.get_trades(symbol="BTC/USD")
        assert len(btc_trades) == 5

        # Buscar por status
        open_trades = db_manager.get_trades(status="open")
        assert len(open_trades) == 3


class TestDatabaseManagerAnalyses:
    """Testes para operações de análises"""

    def test_save_analysis(self, db_manager):
        """Testa salvamento de análise"""
        analysis_data = {
            "request_id": "req_001",
            "symbol": "BTC/USD",
            "timeframe": "1h",
            "strategy": "scalping",
            "decision": "BUY",
            "confidence": 0.85,
            "opportunity_score": 78,
            "technical_analysis": {"rsi": 45, "macd": 120},
        }

        db_manager.save_analysis(analysis_data)

        # Verificar que foi salvo
        result = db_manager.conn.execute(
            "SELECT * FROM analyses WHERE request_id = ?", ["req_001"]
        ).fetchone()
        assert result is not None

    def test_get_recent_analyses(self, db_manager):
        """Testa busca de análises recentes"""
        # Criar múltiplas análises
        for i in range(5):
            analysis_data = {
                "request_id": f"req_{i:03d}",
                "symbol": "BTC/USD",
                "timeframe": "1h",
                "strategy": "scalping",
                "decision": "BUY",
                "confidence": 0.85,
                "opportunity_score": 70 + i,
                "technical_analysis": {},
            }
            db_manager.save_analysis(analysis_data)

        # Buscar recentes
        recent = db_manager.get_recent_analyses(hours=24, limit=10)
        assert len(recent) == 5

        # Buscar por símbolo
        btc_analyses = db_manager.get_recent_analyses(symbol="BTC/USD", hours=24)
        assert len(btc_analyses) == 5


class TestDatabaseManagerCapital:
    """Testes para operações de capital"""

    def test_save_capital_snapshot(self, db_manager):
        """Testa salvamento de snapshot de capital"""
        capital_data = {
            "total_capital": 10000,
            "available_capital": 7500,
            "allocated_capital": 2500,
            "pnl_daily": 150,
            "pnl_total": 500,
        }

        db_manager.save_capital_snapshot(capital_data)

        # Verificar que foi salvo
        result = db_manager.conn.execute("SELECT COUNT(*) FROM capital_history").fetchone()
        assert result[0] == 1

    def test_get_capital_history(self, db_manager):
        """Testa busca de histórico de capital"""
        # Criar múltiplos snapshots
        for i in range(5):
            capital_data = {
                "total_capital": 10000 + i * 100,
                "available_capital": 7500 + i * 50,
                "allocated_capital": 2500 + i * 50,
                "pnl_daily": 150,
                "pnl_total": 500 + i * 100,
            }
            db_manager.save_capital_snapshot(capital_data)

        # Buscar histórico
        history = db_manager.get_capital_history(days=30)
        assert len(history) == 5


class TestDatabaseManagerStatistics:
    """Testes para estatísticas"""

    def test_get_statistics_empty(self, db_manager):
        """Testa estatísticas com banco vazio"""
        stats = db_manager.get_statistics()

        assert "total_candles" in stats
        assert "total_trades" in stats
        assert "win_rate" in stats
        assert "total_pnl" in stats
        assert "db_size_mb" in stats

        assert stats["total_candles"] == 0
        assert stats["total_trades"] == 0

    def test_get_statistics_with_data(self, db_manager, sample_ohlcv_data):
        """Testa estatísticas com dados"""
        # Adicionar dados
        db_manager.save_ohlcv_data("BTC/USD", "1h", sample_ohlcv_data)

        # Adicionar trades
        for i in range(3):
            trade_data = {
                "trade_id": f"trade_{i:03d}",
                "symbol": "BTC/USD",
                "action": "BUY",
                "entry_price": 50000,
                "quantity": 0.1,
                "status": "closed",
                "strategy": "scalping",
                "entry_time": datetime.utcnow(),
            }
            db_manager.save_trade(trade_data)
            db_manager.update_trade(
                f"trade_{i:03d}",
                exit_price=51000 if i % 2 == 0 else 49000,
                pnl=100 if i % 2 == 0 else -100,
                pnl_percent=2.0 if i % 2 == 0 else -2.0,
                status="closed",
            )

        stats = db_manager.get_statistics()

        assert stats["total_candles"] == len(sample_ohlcv_data)
        assert stats["total_trades"] == 3
        assert stats["win_rate"] > 0


class TestDatabaseManagerCleanup:
    """Testes para limpeza de dados"""

    def test_cleanup_old_data(self, db_manager, sample_ohlcv_data):
        """Testa remoção de dados antigos"""
        # Adicionar dados antigos
        old_data = sample_ohlcv_data.copy()
        old_data["timestamp"] = pd.date_range("2020-01-01", periods=10, freq="1H")
        db_manager.save_ohlcv_data("BTC/USD", "1h", old_data)

        # Adicionar dados recentes
        db_manager.save_ohlcv_data("BTC/USD", "1h", sample_ohlcv_data)

        # Limpar dados com mais de 90 dias
        result = db_manager.cleanup_old_data(days=90)

        # Verificar que resultado foi retornado
        assert "deleted_candles" in result
        assert "deleted_analyses" in result

    def test_optimize(self, db_manager):
        """Testa otimização do banco"""
        # Não deve lançar exceção
        db_manager.optimize()


class TestDatabaseManagerContextManager:
    """Testes para context manager"""

    def test_context_manager(self, temp_db_path):
        """Testa uso como context manager"""
        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value.ping.side_effect = Exception("Redis not available")

            with DatabaseManager(db_path=temp_db_path, use_redis=False) as db:
                assert db.conn is not None

            # Verificar que conexão foi fechada
            # (não podemos verificar diretamente, mas não deve lançar exceção)


class TestAsyncDatabaseManager:
    """Testes para AsyncDatabaseManager"""

    @pytest.mark.asyncio
    async def test_async_save_ohlcv_data(self, temp_db_path, sample_ohlcv_data):
        """Testa salvamento assíncrono"""
        with patch("redis.asyncio.Redis"):
            async_db = AsyncDatabaseManager(db_path=temp_db_path, use_redis=False)

            rows = await async_db.save_ohlcv_data("BTC/USD", "1h", sample_ohlcv_data)
            assert rows == len(sample_ohlcv_data)

            await async_db.close()

    @pytest.mark.asyncio
    async def test_async_get_ohlcv_data(self, temp_db_path, sample_ohlcv_data):
        """Testa busca assíncrona"""
        with patch("redis.asyncio.Redis"):
            async_db = AsyncDatabaseManager(db_path=temp_db_path, use_redis=False)

            # Salvar dados primeiro
            await async_db.save_ohlcv_data("BTC/USD", "1h", sample_ohlcv_data)

            # Buscar dados
            result = await async_db.get_ohlcv_data("BTC/USD", "1h", limit=100)
            assert not result.empty

            await async_db.close()


class TestDatabaseManagerRedisCache:
    """Testes para cache Redis"""

    def test_get_ohlcv_with_redis_cache(self, temp_db_path, sample_ohlcv_data):
        """Testa busca com cache Redis"""
        with patch("redis.Redis") as mock_redis:
            mock_redis_instance = MagicMock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            mock_redis_instance.get.return_value = None

            db = DatabaseManager(db_path=temp_db_path, use_redis=True)

            # Salvar e buscar dados
            db.save_ohlcv_data("BTC/USD", "1h", sample_ohlcv_data)
            result = db.get_ohlcv_data("BTC/USD", "1h", limit=100)

            assert not result.empty

            # Verificar que tentou cachear
            assert mock_redis_instance.setex.called

            db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=utils.database", "--cov-report=term-missing"])
