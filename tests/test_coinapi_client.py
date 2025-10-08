"""
CeciAI - Testes do CoinAPI Client
Testa cliente com mocks (sem chamadas reais à API)

Autor: CeciAI Team
Data: 2025-10-08
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from utils.coinapi_client import CoinAPIClient, CoinAPIMode


class TestCoinAPIClientInit:
    """Testes de inicialização do cliente"""

    def test_init_development_mode(self):
        """Testa inicialização em modo development"""
        client = CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT)
        assert client.mode == CoinAPIMode.DEVELOPMENT
        assert client.BASE_URL == "https://rest.coinapi.io/v1"

    def test_init_production_mode_without_key(self):
        """Testa que modo production requer API key"""
        with pytest.raises(ValueError, match="API key é obrigatória"):
            CoinAPIClient(mode=CoinAPIMode.PRODUCTION, api_key=None)

    def test_init_production_mode_with_key(self):
        """Testa inicialização em modo production com key"""
        client = CoinAPIClient(mode=CoinAPIMode.PRODUCTION, api_key="test-key")
        assert client.mode == CoinAPIMode.PRODUCTION
        assert client.api_key == "test-key"

    def test_init_with_env_var(self):
        """Testa inicialização com variável de ambiente"""
        with patch.dict("os.environ", {"COINAPI_KEY": "env-key"}):
            client = CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT)
            assert client.api_key == "env-key"


class TestCoinAPIClientHelpers:
    """Testes para métodos auxiliares"""

    def test_convert_symbol(self):
        """Testa conversão de símbolos"""
        client = CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT)

        assert client._convert_symbol("BTC/USD") == "BINANCE_SPOT_BTC_USDT"
        assert client._convert_symbol("ETH/USD") == "BINANCE_SPOT_ETH_USDT"
        assert client._convert_symbol("BTC/USDT") == "BINANCE_SPOT_BTC_USDT"
        assert client._convert_symbol("ETH/USDT") == "BINANCE_SPOT_ETH_USDT"

    def test_convert_timeframe(self):
        """Testa conversão de timeframes"""
        client = CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT)

        assert client._convert_timeframe("1min") == "1MIN"
        assert client._convert_timeframe("5min") == "5MIN"
        assert client._convert_timeframe("15min") == "15MIN"
        assert client._convert_timeframe("1h") == "1HRS"
        assert client._convert_timeframe("4h") == "4HRS"
        assert client._convert_timeframe("1d") == "1DAY"

    def test_calculate_start_time(self):
        """Testa cálculo de data inicial"""
        client = CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT)
        end_time = datetime(2024, 1, 1, 12, 0, 0)

        # 1h com 10 candles = 10 horas atrás
        start_time = client._calculate_start_time(end_time, "1h", 10)
        expected = end_time - timedelta(hours=10)
        assert start_time == expected

        # 1d com 7 candles = 7 dias atrás
        start_time = client._calculate_start_time(end_time, "1d", 7)
        expected = end_time - timedelta(days=7)
        assert start_time == expected

    def test_is_data_fresh(self):
        """Testa verificação de dados recentes"""
        client = CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT)

        # Dados recentes (1 hora atrás)
        df_fresh = pd.DataFrame({"timestamp": [datetime.utcnow() - timedelta(hours=1)]})
        assert client._is_data_fresh(df_fresh, hours=24)

        # Dados antigos (48 horas atrás)
        df_old = pd.DataFrame({"timestamp": [datetime.utcnow() - timedelta(hours=48)]})
        assert not client._is_data_fresh(df_old, hours=24)

        # DataFrame vazio
        df_empty = pd.DataFrame()
        assert not client._is_data_fresh(df_empty, hours=24)


class TestCoinAPIClientOHLCV:
    """Testes para métodos de OHLCV"""

    @pytest.mark.asyncio
    async def test_get_ohlcv_data_from_cache(self):
        """Testa busca de dados do cache"""
        client = CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT)

        # Mock do database
        mock_df = pd.DataFrame(
            {
                "timestamp": [datetime.utcnow()],
                "open": [50000],
                "high": [51000],
                "low": [49000],
                "close": [50500],
                "volume": [1000000],
            }
        )

        with patch.object(client.db, "get_ohlcv_data", return_value=mock_df):
            result = await client.get_ohlcv_data("BTC/USD", "1h", limit=100)
            assert not result.empty
            assert len(result) == 1
            assert result["close"].iloc[0] == 50500

    @pytest.mark.asyncio
    async def test_get_ohlcv_data_development_no_cache(self):
        """Testa erro quando não há cache em modo development"""
        client = CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT)

        with patch.object(client.db, "get_ohlcv_data", return_value=pd.DataFrame()), pytest.raises(ValueError, match="Sem dados em cache"):
            await client.get_ohlcv_data("BTC/USD", "1h", limit=100)

    @pytest.mark.asyncio
    async def test_get_ohlcv_data_development_old_cache(self):
        """Testa uso de cache antigo em modo development"""
        client = CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT)

        # Mock old data (48 hours ago)
        old_df = pd.DataFrame(
            {
                "timestamp": [datetime.utcnow() - timedelta(hours=48)],
                "open": [50000],
                "high": [51000],
                "low": [49000],
                "close": [50500],
                "volume": [1000000],
            }
        )

        with patch.object(client.db, "get_ohlcv_data", return_value=old_df):
            result = await client.get_ohlcv_data("BTC/USD", "1h", limit=100)
            assert not result.empty
            # Should use old cache in development mode

    @pytest.mark.asyncio
    async def test_get_ohlcv_data_production_from_api(self):
        """Testa busca de dados da API em modo production"""
        client = CoinAPIClient(mode=CoinAPIMode.PRODUCTION, api_key="test-key")

        # Mock da API response
        mock_api_data = [
            {
                "time_period_start": "2024-01-01T00:00:00",
                "price_open": 50000,
                "price_high": 51000,
                "price_low": 49000,
                "price_close": 50500,
                "volume_traded": 1000000,
            }
        ]

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_api_data)

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        client.session = AsyncMock()
        client.session.get = MagicMock(return_value=mock_context)

        with patch.object(client.db, "get_ohlcv_data", return_value=pd.DataFrame()), patch.object(
            client.db, "save_ohlcv_data", return_value=None
        ):
            result = await client.get_ohlcv_data("BTC/USD", "1h", limit=100, force_refresh=True)
            assert not result.empty
            assert "timestamp" in result.columns
            assert "close" in result.columns

    @pytest.mark.asyncio
    async def test_get_latest_price_from_cache(self):
        """Testa busca de preço atual do cache"""
        client = CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT)

        mock_df = pd.DataFrame({"timestamp": [datetime.utcnow()], "close": [50500]})

        with patch.object(client.db, "get_ohlcv_data", return_value=mock_df):
            price = await client.get_latest_price("BTC/USD")
            assert price == 50500

    @pytest.mark.asyncio
    async def test_get_latest_price_no_data(self):
        """Testa erro quando não há dados de preço"""
        client = CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT)

        with patch.object(client.db, "get_ohlcv_data", return_value=pd.DataFrame()), pytest.raises(ValueError, match="Sem dados disponíveis"):
            await client.get_latest_price("BTC/USD")

    @pytest.mark.asyncio
    async def test_get_latest_price_old_cache_fallback(self):
        """Testa fallback para cache antigo quando não há dados recentes"""
        client = CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT)

        # Mock old data (1 hour ago)
        old_df = pd.DataFrame(
            {"timestamp": [datetime.utcnow() - timedelta(hours=1)], "close": [50500]}
        )

        with patch.object(client.db, "get_ohlcv_data", return_value=old_df):
            price = await client.get_latest_price("BTC/USD")
            assert price == 50500

    @pytest.mark.asyncio
    async def test_get_latest_price_production_api(self):
        """Testa busca de preço da API em modo production"""
        client = CoinAPIClient(mode=CoinAPIMode.PRODUCTION, api_key="test-key")

        # Mock API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"ask_price": 50500.0})

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        client.session = AsyncMock()
        client.session.get = MagicMock(return_value=mock_context)

        with patch.object(client.db, "get_ohlcv_data", return_value=pd.DataFrame()):
            price = await client.get_latest_price("BTC/USD")
            assert price == 50500.0

    @pytest.mark.asyncio
    async def test_get_orderbook_production(self):
        """Testa orderbook em modo production"""
        client = CoinAPIClient(mode=CoinAPIMode.PRODUCTION, api_key="test-key")

        # Mock API response
        mock_api_data = {
            "bids": [{"price": 50000, "size": 0.5}, {"price": 49900, "size": 1.0}],
            "asks": [{"price": 50100, "size": 0.5}, {"price": 50200, "size": 1.0}],
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_api_data)

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        client.session = AsyncMock()
        client.session.get = MagicMock(return_value=mock_context)

        orderbook = await client.get_orderbook("BTC/USD", depth=2)

        assert "bids" in orderbook
        assert "asks" in orderbook
        assert len(orderbook["bids"]) == 2
        assert len(orderbook["asks"]) == 2
        assert orderbook["bids"][0] == [50000, 0.5]
        assert orderbook["asks"][0] == [50100, 0.5]

    @pytest.mark.asyncio
    async def test_get_orderbook_production_error(self):
        """Testa erro no orderbook em modo production"""
        client = CoinAPIClient(mode=CoinAPIMode.PRODUCTION, api_key="test-key")

        mock_response = AsyncMock()
        mock_response.status = 500

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        client.session = AsyncMock()
        client.session.get = MagicMock(return_value=mock_context)

        with pytest.raises(Exception, match="Erro ao buscar orderbook"):
            await client.get_orderbook("BTC/USD")

    @pytest.mark.asyncio
    async def test_get_orderbook_development(self):
        """Testa orderbook em modo development (mock)"""
        client = CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT)

        orderbook = await client.get_orderbook("BTC/USD")
        assert "bids" in orderbook
        assert "asks" in orderbook
        assert len(orderbook["bids"]) > 0
        assert len(orderbook["asks"]) > 0


class TestCoinAPIClientContextManager:
    """Testes para context manager"""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Testa uso como context manager"""
        async with CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT) as client:
            assert client.session is not None

    @pytest.mark.asyncio
    async def test_close_method(self):
        """Testa método close"""
        client = CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT)
        client.session = AsyncMock()

        await client.close()
        client.session.close.assert_called_once()


class TestCoinAPIClientErrors:
    """Testes para tratamento de erros"""

    @pytest.mark.asyncio
    async def test_api_rate_limit_error(self):
        """Testa erro de rate limit"""
        client = CoinAPIClient(mode=CoinAPIMode.PRODUCTION, api_key="test-key")

        mock_response = AsyncMock()
        mock_response.status = 429

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        client.session = AsyncMock()
        client.session.get = MagicMock(return_value=mock_context)

        with patch.object(client.db, "get_ohlcv_data", return_value=pd.DataFrame()), pytest.raises(Exception, match="Rate limit excedido"):
            await client._fetch_from_api("BINANCE_SPOT_BTC_USDT", "1h", 100)

    @pytest.mark.asyncio
    async def test_api_generic_error(self):
        """Testa erro genérico da API"""
        client = CoinAPIClient(mode=CoinAPIMode.PRODUCTION, api_key="test-key")

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        client.session = AsyncMock()
        client.session.get = MagicMock(return_value=mock_context)

        with pytest.raises(Exception, match="Erro 500"):
            await client._fetch_from_api("BINANCE_SPOT_BTC_USDT", "1h", 100)


class TestCoinAPIClientCache:
    """Testes para funcionalidade de cache"""

    @pytest.mark.asyncio
    async def test_cache_save(self):
        """Testa salvamento no cache"""
        client = CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT)

        df = pd.DataFrame(
            {
                "timestamp": [datetime.utcnow()],
                "open": [50000],
                "high": [51000],
                "low": [49000],
                "close": [50500],
                "volume": [1000000],
            }
        )

        with patch.object(client.db, "save_ohlcv_data", return_value=1) as mock_save:
            await client._save_to_cache("BTC/USD", "1h", df)
            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_get(self):
        """Testa busca do cache"""
        client = CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT)

        mock_df = pd.DataFrame({"timestamp": [datetime.utcnow()], "close": [50500]})

        with patch.object(client.db, "get_ohlcv_data", return_value=mock_df) as mock_get:
            result = await client._get_from_cache("BTC/USD", "1h", 100)
            assert not result.empty
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_get_error(self):
        """Testa erro ao buscar do cache"""
        client = CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT)

        with patch.object(client.db, "get_ohlcv_data", side_effect=Exception("Database error")):
            result = await client._get_from_cache("BTC/USD", "1h", 100)
            assert result.empty

    @pytest.mark.asyncio
    async def test_cache_save_error(self):
        """Testa erro ao salvar no cache"""
        client = CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT)

        df = pd.DataFrame({"timestamp": [datetime.utcnow()], "close": [50500]})

        with patch.object(client.db, "save_ohlcv_data", side_effect=Exception("Database error")):
            # Should not raise exception, just log error
            await client._save_to_cache("BTC/USD", "1h", df)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=utils.coinapi_client", "--cov-report=term-missing"])
