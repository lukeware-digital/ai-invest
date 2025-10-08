"""
CeciAI - Database Manager
Sistema de cache local ultra-performático e barato

Usa:
- DuckDB: Banco principal (arquivo local, zero custo, alta performance)
- Redis: Cache em memória (opcional, para dados hot)
- Parquet: Formato de armazenamento comprimido

Autor: CeciAI Team
Data: 2025-10-07
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import redis
from redis.asyncio import Redis as AsyncRedis


class DatabaseManager:
    """
    Gerenciador de banco de dados local de alta performance.

    Features:
    - DuckDB para armazenamento persistente (zero custo)
    - Redis para cache em memória (opcional)
    - Compressão automática com Parquet
    - Queries SQL rápidas
    - TTL automático para dados antigos
    """

    def __init__(
        self,
        db_path: str = "data/ceciai.duckdb",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        use_redis: bool = True,
    ):
        """
        Inicializa o gerenciador de banco de dados.

        Args:
            db_path: Caminho para arquivo DuckDB
            redis_host: Host do Redis
            redis_port: Porta do Redis
            use_redis: Se True, usa Redis para cache
        """
        self.db_path = db_path
        self.use_redis = use_redis

        # Criar diretório se não existir
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Conectar DuckDB
        self.conn = duckdb.connect(db_path)

        # Configurar DuckDB para performance
        self.conn.execute("SET memory_limit='4GB'")
        self.conn.execute("SET threads=8")

        # Redis (opcional)
        self.redis_client = None
        if use_redis:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host, port=redis_port, db=0, decode_responses=True
                )
                self.redis_client.ping()
            except Exception as e:
                print(f"⚠️  Redis não disponível, usando apenas DuckDB: {e}")
                self.use_redis = False
                self.redis_client = None

        # Criar tabelas
        self._create_tables()

    def _create_tables(self):
        """Cria tabelas necessárias"""

        # Tabela de dados OHLCV
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume DOUBLE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(symbol, timeframe, timestamp)
            )
        """
        )

        # Índices para performance
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time
            ON ohlcv_data(symbol, timeframe, timestamp DESC)
        """
        )

        # Tabela de trades executados
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                trade_id VARCHAR PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                action VARCHAR NOT NULL,
                entry_price DOUBLE NOT NULL,
                exit_price DOUBLE,
                quantity DOUBLE NOT NULL,
                stop_loss DOUBLE,
                take_profit DOUBLE,
                pnl DOUBLE,
                pnl_percent DOUBLE,
                status VARCHAR NOT NULL,
                strategy VARCHAR NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Tabela de análises (cache de decisões)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                request_id VARCHAR PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                strategy VARCHAR NOT NULL,
                decision VARCHAR NOT NULL,
                confidence DOUBLE NOT NULL,
                opportunity_score INTEGER NOT NULL,
                analysis_data JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Tabela de capital histórico
        self.conn.execute(
            """
            CREATE SEQUENCE IF NOT EXISTS capital_history_seq;
            CREATE TABLE IF NOT EXISTS capital_history (
                id INTEGER PRIMARY KEY DEFAULT nextval('capital_history_seq'),
                total_capital DOUBLE NOT NULL,
                available_capital DOUBLE NOT NULL,
                allocated_capital DOUBLE NOT NULL,
                pnl_daily DOUBLE,
                pnl_total DOUBLE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

    # ==================== OHLCV DATA ====================

    def save_ohlcv_data(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        """
        Salva dados OHLCV no banco.

        Args:
            symbol: Par de trading (ex: BTC/USD)
            timeframe: Timeframe (ex: 1h)
            df: DataFrame com colunas [timestamp, open, high, low, close, volume]

        Returns:
            Número de linhas inseridas
        """
        # Adicionar colunas de metadados
        df = df.copy()
        df["symbol"] = symbol
        df["timeframe"] = timeframe

        # Convert timestamp to string to avoid precision issues
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        # Inserir no DuckDB usando INSERT OR REPLACE
        try:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO ohlcv_data
                (symbol, timeframe, timestamp, open, high, low, close, volume)
                SELECT symbol, timeframe, CAST(timestamp AS TIMESTAMP), open, high, low, close, volume
                FROM df
            """
            )
        except Exception:
            # Fallback: delete then insert
            for _, row in df.iterrows():
                self.conn.execute(
                    """
                    DELETE FROM ohlcv_data
                    WHERE symbol = ? AND timeframe = ? AND timestamp = CAST(? AS TIMESTAMP)
                """,
                    [symbol, timeframe, row["timestamp"]],
                )

            self.conn.execute(
                """
                INSERT INTO ohlcv_data
                (symbol, timeframe, timestamp, open, high, low, close, volume)
                SELECT symbol, timeframe, CAST(timestamp AS TIMESTAMP), open, high, low, close, volume
                FROM df
            """
            )

        rows_inserted = len(df)

        # Salvar também em Parquet para backup comprimido
        parquet_path = f"data/historical/{symbol.replace('/', '_')}_{timeframe}.parquet"
        Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, compression="snappy", index=False)

        return rows_inserted

    def get_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Busca dados OHLCV do banco.

        Args:
            symbol: Par de trading
            timeframe: Timeframe
            start_date: Data inicial (opcional)
            end_date: Data final (opcional)
            limit: Limite de registros

        Returns:
            DataFrame com dados OHLCV
        """
        # Tentar cache Redis primeiro
        if self.use_redis and self.redis_client:
            cache_key = f"ohlcv:{symbol}:{timeframe}:{start_date}:{end_date}:{limit}"
            cached = self.redis_client.get(cache_key)
            if cached:
                return pd.read_json(cached)

        # Query DuckDB
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_data
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        df = self.conn.execute(query, params).df()

        # Cachear no Redis (TTL 5 minutos)
        if self.use_redis and self.redis_client and not df.empty:
            cache_key = f"ohlcv:{symbol}:{timeframe}:{start_date}:{end_date}:{limit}"
            self.redis_client.setex(
                cache_key,
                300,  # 5 minutos
                df.to_json(),
            )

        return df

    def get_latest_ohlcv(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """
        Busca os últimos N candles.

        Args:
            symbol: Par de trading
            timeframe: Timeframe
            count: Número de candles

        Returns:
            DataFrame com últimos candles
        """
        return self.get_ohlcv_data(symbol=symbol, timeframe=timeframe, limit=count)

    # ==================== TRADES ====================

    def save_trade(self, trade_data: dict[str, Any]) -> str:
        """
        Salva trade executado.

        Args:
            trade_data: Dados do trade

        Returns:
            ID do trade
        """
        self.conn.execute(
            """
            INSERT INTO trades (
                trade_id, symbol, action, entry_price, quantity,
                stop_loss, take_profit, status, strategy, entry_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                trade_data["trade_id"],
                trade_data["symbol"],
                trade_data["action"],
                trade_data["entry_price"],
                trade_data["quantity"],
                trade_data.get("stop_loss"),
                trade_data.get("take_profit"),
                trade_data.get("status", "open"),
                trade_data.get("strategy", "scalping"),
                trade_data.get("entry_time", datetime.utcnow()),
            ],
        )

        return trade_data["trade_id"]

    def update_trade(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float,
        pnl_percent: float,
        status: str = "closed",
    ):
        """Atualiza trade quando fechado"""
        self.conn.execute(
            """
            UPDATE trades
            SET exit_price = ?, pnl = ?, pnl_percent = ?,
                status = ?, exit_time = CURRENT_TIMESTAMP
            WHERE trade_id = ?
        """,
            [exit_price, pnl, pnl_percent, status, trade_id],
        )

    def get_trades(
        self, symbol: str | None = None, status: str | None = None, limit: int = 100
    ) -> pd.DataFrame:
        """Busca histórico de trades"""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY entry_time DESC LIMIT ?"
        params.append(limit)

        return self.conn.execute(query, params).df()

    # ==================== ANALYSES ====================

    def save_analysis(self, analysis_data: dict[str, Any]):
        """Salva resultado de análise"""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO analyses (
                request_id, symbol, timeframe, strategy,
                decision, confidence, opportunity_score, analysis_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                analysis_data["request_id"],
                analysis_data["symbol"],
                analysis_data["timeframe"],
                analysis_data["strategy"],
                analysis_data["decision"],
                analysis_data["confidence"],
                analysis_data["opportunity_score"],
                json.dumps(analysis_data),
            ],
        )

    def get_recent_analyses(
        self, symbol: str | None = None, hours: int = 24, limit: int = 50
    ) -> pd.DataFrame:
        """Busca análises recentes"""
        # DuckDB interval syntax
        query = f"""
            SELECT * FROM analyses
            WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '{hours}' HOUR
        """
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        return self.conn.execute(query, params).df()

    # ==================== CAPITAL HISTORY ====================

    def save_capital_snapshot(self, capital_data: dict[str, Any]):
        """Salva snapshot do capital"""
        self.conn.execute(
            """
            INSERT INTO capital_history (
                total_capital, available_capital, allocated_capital,
                pnl_daily, pnl_total
            ) VALUES (?, ?, ?, ?, ?)
        """,
            [
                capital_data["total_capital"],
                capital_data["available_capital"],
                capital_data["allocated_capital"],
                capital_data.get("pnl_daily", 0),
                capital_data.get("pnl_total", 0),
            ],
        )

    def get_capital_history(self, days: int = 30) -> pd.DataFrame:
        """Busca histórico de capital"""
        # DuckDB interval syntax
        return self.conn.execute(
            f"""
            SELECT * FROM capital_history
            WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '{days}' DAY
            ORDER BY timestamp DESC
        """
        ).df()

    # ==================== STATISTICS ====================

    def get_statistics(self) -> dict[str, Any]:
        """Retorna estatísticas do banco"""
        stats = {}

        # Total de candles armazenados
        stats["total_candles"] = self.conn.execute(
            "SELECT COUNT(*) as count FROM ohlcv_data"
        ).fetchone()[0]

        # Total de trades
        stats["total_trades"] = self.conn.execute(
            "SELECT COUNT(*) as count FROM trades"
        ).fetchone()[0]

        # Win rate
        win_rate = self.conn.execute(
            """
            SELECT
                COUNT(CASE WHEN pnl > 0 THEN 1 END) * 100.0 / COUNT(*) as win_rate
            FROM trades
            WHERE status = 'closed'
        """
        ).fetchone()
        stats["win_rate"] = win_rate[0] if win_rate[0] else 0

        # Total P&L
        total_pnl = self.conn.execute(
            """
            SELECT SUM(pnl) as total_pnl
            FROM trades
            WHERE status = 'closed'
        """
        ).fetchone()
        stats["total_pnl"] = total_pnl[0] if total_pnl[0] else 0

        # Tamanho do banco
        stats["db_size_mb"] = os.path.getsize(self.db_path) / (1024 * 1024)

        return stats

    # ==================== CLEANUP ====================

    def cleanup_old_data(self, days: int = 90):
        """Remove dados antigos para economizar espaço"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")

        # Remover candles antigos
        self.conn.execute(
            f"""
            DELETE FROM ohlcv_data
            WHERE timestamp < CAST('{cutoff_str}' AS TIMESTAMP)
        """
        )

        # Remover análises antigas
        self.conn.execute(
            f"""
            DELETE FROM analyses
            WHERE created_at < CAST('{cutoff_str}' AS TIMESTAMP)
        """
        )

        # Vacuum para recuperar espaço
        self.conn.execute("VACUUM")

        return {"deleted_candles": "success", "deleted_analyses": "success"}

    def optimize(self):
        """Otimiza banco de dados"""
        self.conn.execute("ANALYZE")
        self.conn.execute("VACUUM")

    def close(self):
        """Fecha conexões"""
        if self.conn:
            self.conn.close()

        if self.redis_client:
            self.redis_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==================== ASYNC VERSION ====================


class AsyncDatabaseManager:
    """Versão assíncrona do DatabaseManager"""

    def __init__(
        self,
        db_path: str = "data/ceciai.duckdb",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        use_redis: bool = True,
    ):
        self.db_manager = DatabaseManager(db_path, redis_host, redis_port, use_redis)
        self.redis_async = None

        if use_redis:
            self.redis_async = AsyncRedis(
                host=redis_host, port=redis_port, db=0, decode_responses=True
            )

    async def save_ohlcv_data(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        """Versão assíncrona de save_ohlcv_data"""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.db_manager.save_ohlcv_data, symbol, timeframe, df
        )

    async def get_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Versão assíncrona de get_ohlcv_data"""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.db_manager.get_ohlcv_data, symbol, timeframe, start_date, end_date, limit
        )

    async def close(self):
        """Fecha conexões assíncronas"""
        if self.redis_async:
            await self.redis_async.aclose()
        self.db_manager.close()


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Exemplo de uso
    print("CeciAI - Database Manager")
    print("=" * 50)

    with DatabaseManager() as db:
        # Estatísticas
        stats = db.get_statistics()
        print("\n📊 Estatísticas:")
        print(f"  Total de candles: {stats['total_candles']:,}")
        print(f"  Total de trades: {stats['total_trades']}")
        print(f"  Win rate: {stats['win_rate']:.2f}%")
        print(f"  Total P&L: ${stats['total_pnl']:,.2f}")
        print(f"  Tamanho do banco: {stats['db_size_mb']:.2f} MB")

        print("\n✅ Database Manager funcionando!")
