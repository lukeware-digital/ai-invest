"""
CeciAI - CoinAPI Client
Cliente assíncrono com cache local automático (DuckDB)

Features:
- 100% assíncrono
- Cache automático de todos os dados
- Modo desenvolvimento (usa cache, economiza API calls)
- Modo produção (dados real-time + cache)
- Zero custo de armazenamento

Autor: CeciAI Team
Data: 2025-10-07
"""

import asyncio
import os
from datetime import datetime, timedelta
from enum import Enum

import aiohttp
import pandas as pd
from loguru import logger

from utils.database import AsyncDatabaseManager


class CoinAPIMode(Enum):
    """Modos de operação"""

    DEVELOPMENT = "development"  # Usa apenas cache
    PRODUCTION = "production"  # Real-time + cache


class CoinAPIClient:
    """
    Cliente assíncrono para CoinAPI com cache local.

    Todos os dados baixados são salvos automaticamente no DuckDB.
    """

    BASE_URL = "https://rest.coinapi.io/v1"

    def __init__(
        self,
        api_key: str | None = None,
        mode: CoinAPIMode = CoinAPIMode.DEVELOPMENT,
        db_path: str = "data/ceciai.duckdb",
    ):
        """
        Inicializa cliente CoinAPI.

        Args:
            api_key: Chave da API (obrigatória em produção)
            mode: Modo de operação (development ou production)
            db_path: Caminho para banco DuckDB
        """
        self.api_key = api_key or os.getenv("COINAPI_KEY")
        self.mode = mode

        # Validar API key em produção
        if mode == CoinAPIMode.PRODUCTION and not self.api_key:
            raise ValueError("API key é obrigatória em modo produção")

        # Database para cache
        self.db = AsyncDatabaseManager(db_path=db_path)

        # Session HTTP
        self.session: aiohttp.ClientSession | None = None

        # Headers
        self.headers = {"X-CoinAPI-Key": self.api_key or "", "Accept": "application/json"}

        logger.info(f"CoinAPI Client inicializado em modo: {mode.value}")

    async def __aenter__(self):
        """Context manager entry"""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()

    async def close(self):
        """Fecha conexões"""
        if self.session:
            await self.session.close()
        await self.db.close()

    # ==================== OHLCV DATA ====================

    async def get_ohlcv_data(
        self, symbol: str, timeframe: str = "1h", limit: int = 100, force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Busca dados OHLCV (Open, High, Low, Close, Volume).

        Em modo development: Usa apenas cache local
        Em modo production: Busca da API e salva no cache

        Args:
            symbol: Par de trading (ex: BTC/USD)
            timeframe: Timeframe (1min, 5min, 1h, 4h, 1d)
            limit: Número de candles
            force_refresh: Se True, força busca da API

        Returns:
            DataFrame com dados OHLCV
        """
        # Converter símbolo para formato CoinAPI
        coinapi_symbol = self._convert_symbol(symbol)

        # Tentar cache primeiro
        if not force_refresh:
            cached_data = await self._get_from_cache(symbol, timeframe, limit)

            if not cached_data.empty and self._is_data_fresh(cached_data, hours=24):
                logger.info(f"✅ Usando cache para {symbol} {timeframe}")
                return cached_data

        # Modo development: Retornar cache ou erro
        if self.mode == CoinAPIMode.DEVELOPMENT:
            if not cached_data.empty:
                logger.warning(f"⚠️  Cache antigo para {symbol}, mas em modo dev")
                return cached_data
            else:
                raise ValueError(
                    f"Sem dados em cache para {symbol} {timeframe}. "
                    f"Execute download_historical_data.py primeiro."
                )

        # Modo production: Buscar da API
        logger.info(f"🌐 Buscando {symbol} {timeframe} da CoinAPI...")

        data = await self._fetch_from_api(
            coinapi_symbol=coinapi_symbol, timeframe=timeframe, limit=limit
        )

        # Salvar no cache
        if not data.empty:
            await self._save_to_cache(symbol, timeframe, data)
            logger.info(f"💾 Dados salvos no cache: {len(data)} candles")

        return data

    async def get_latest_price(self, symbol: str) -> float:
        """
        Busca preço atual.

        Args:
            symbol: Par de trading

        Returns:
            Preço atual
        """
        # Tentar cache primeiro (últimos 5 minutos)
        cached = await self._get_from_cache(symbol, "1min", limit=1)

        if not cached.empty:
            last_update = pd.to_datetime(cached["timestamp"].iloc[0])
            if datetime.utcnow() - last_update < timedelta(minutes=5):
                return float(cached["close"].iloc[0])

        # Buscar da API
        if self.mode == CoinAPIMode.PRODUCTION:
            coinapi_symbol = self._convert_symbol(symbol)

            url = f"{self.BASE_URL}/quotes/{coinapi_symbol}/current"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data["ask_price"])
                else:
                    raise Exception(f"Erro ao buscar preço: {response.status}")

        # Fallback: último preço do cache
        if not cached.empty:
            return float(cached["close"].iloc[0])

        raise ValueError(f"Sem dados disponíveis para {symbol}")

    async def get_orderbook(self, symbol: str, depth: int = 20) -> dict[str, object]:
        """
        Busca order book (livro de ofertas).

        Args:
            symbol: Par de trading
            depth: Profundidade do order book

        Returns:
            Dict com bids e asks
        """
        if self.mode == CoinAPIMode.DEVELOPMENT:
            # Retornar mock em desenvolvimento
            return {"bids": [[50000, 0.5], [49900, 1.0]], "asks": [[50100, 0.5], [50200, 1.0]]}

        coinapi_symbol = self._convert_symbol(symbol)
        url = f"{self.BASE_URL}/orderbooks/{coinapi_symbol}/current"

        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "bids": [[b["price"], b["size"]] for b in data["bids"][:depth]],
                    "asks": [[a["price"], a["size"]] for a in data["asks"][:depth]],
                }
            else:
                raise Exception(f"Erro ao buscar orderbook: {response.status}")

    # ==================== PRIVATE METHODS ====================

    async def _get_from_cache(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Busca dados do cache (DuckDB)"""
        try:
            return await self.db.get_ohlcv_data(symbol=symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            logger.error(f"Erro ao buscar cache: {e}")
            return pd.DataFrame()

    async def _save_to_cache(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Salva dados no cache (DuckDB)"""
        try:
            await self.db.save_ohlcv_data(symbol, timeframe, df)
        except Exception as e:
            logger.error(f"Erro ao salvar cache: {e}")

    async def _fetch_from_api(
        self, coinapi_symbol: str, timeframe: str, limit: int
    ) -> pd.DataFrame:
        """Busca dados da API CoinAPI"""

        # Converter timeframe para formato CoinAPI
        period_id = self._convert_timeframe(timeframe)

        # Calcular data inicial
        end_time = datetime.utcnow()
        start_time = self._calculate_start_time(end_time, timeframe, limit)

        url = f"{self.BASE_URL}/ohlcv/{coinapi_symbol}/history"
        params = {
            "period_id": period_id,
            "time_start": start_time.isoformat(),
            "time_end": end_time.isoformat(),
            "limit": limit,
        }

        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()

                # Converter para DataFrame
                df = pd.DataFrame(data)

                if not df.empty:
                    df = df.rename(
                        columns={
                            "time_period_start": "timestamp",
                            "price_open": "open",
                            "price_high": "high",
                            "price_low": "low",
                            "price_close": "close",
                            "volume_traded": "volume",
                        }
                    )

                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

                return df

            elif response.status == 429:
                raise Exception("Rate limit excedido. Aguarde antes de tentar novamente.")

            else:
                error_text = await response.text()
                raise Exception(f"Erro {response.status}: {error_text}")

    def _is_data_fresh(self, df: pd.DataFrame, hours: int = 24) -> bool:
        """Verifica se dados são recentes"""
        if df.empty:
            return False

        last_timestamp = pd.to_datetime(df["timestamp"].iloc[0])
        age = datetime.utcnow() - last_timestamp

        return age < timedelta(hours=hours)

    def _convert_symbol(self, symbol: str) -> str:
        """
        Converte símbolo para formato CoinAPI.

        BTC/USD -> BINANCE_SPOT_BTC_USDT
        """
        # Mapeamento de símbolos
        mapping = {
            "BTC/USD": "BINANCE_SPOT_BTC_USDT",
            "ETH/USD": "BINANCE_SPOT_ETH_USDT",
            "BTC/USDT": "BINANCE_SPOT_BTC_USDT",
            "ETH/USDT": "BINANCE_SPOT_ETH_USDT",
        }

        return mapping.get(symbol, symbol)

    def _convert_timeframe(self, timeframe: str) -> str:
        """
        Converte timeframe para formato CoinAPI.

        1min -> 1MIN
        1h -> 1HRS
        """
        mapping = {
            "1min": "1MIN",
            "5min": "5MIN",
            "15min": "15MIN",
            "1h": "1HRS",
            "4h": "4HRS",
            "1d": "1DAY",
        }

        return mapping.get(timeframe, timeframe)

    def _calculate_start_time(self, end_time: datetime, timeframe: str, limit: int) -> datetime:
        """Calcula data inicial baseado no timeframe e limite"""

        # Duração de cada candle em minutos
        durations = {"1min": 1, "5min": 5, "15min": 15, "1h": 60, "4h": 240, "1d": 1440}

        minutes = durations.get(timeframe, 60) * limit
        return end_time - timedelta(minutes=minutes)


# ==================== USAGE EXAMPLE ====================


async def main():
    """Exemplo de uso"""
    print("CeciAI - CoinAPI Client")
    print("=" * 50)

    async with CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT) as client:
        # Buscar dados OHLCV
        try:
            df = await client.get_ohlcv_data("BTC/USD", "1h", limit=100)
            print(f"\n📊 Dados OHLCV: {len(df)} candles")
            print(df.head())

            # Preço atual
            price = await client.get_latest_price("BTC/USD")
            print(f"\n💰 Preço atual: ${price:,.2f}")

        except Exception as e:
            print(f"\n❌ Erro: {e}")
            print("💡 Execute 'python utils/download_historical_data.py' primeiro")


if __name__ == "__main__":
    asyncio.run(main())
