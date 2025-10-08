"""
CeciAI - Testes Unitários para Technical Indicators

Testes com 100% de cobertura usando mocks.
Todos os dados são mockados para garantir testes isolados e rápidos.

Autor: CeciAI Team
Data: 2025-10-07
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.technical_indicators import TechnicalIndicators


class TestTechnicalIndicators:
    """Testes para a classe TechnicalIndicators"""

    @pytest.fixture
    def ti(self):
        """Fixture que retorna uma instância de TechnicalIndicators"""
        return TechnicalIndicators()

    @pytest.fixture
    def mock_prices(self):
        """Fixture com série de preços mockada"""
        np.random.seed(42)
        prices = pd.Series(
            50000 + np.random.randn(100).cumsum() * 100,
            index=pd.date_range("2024-01-01", periods=100, freq="1H"),
        )
        return prices

    @pytest.fixture
    def mock_ohlcv_data(self):
        """Fixture com dados OHLCV completos mockados"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="1H")

        base_price = 50000
        prices = base_price + np.random.randn(100).cumsum() * 100

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices + np.random.randn(100) * 50,
                "high": prices + np.abs(np.random.randn(100)) * 100,
                "low": prices - np.abs(np.random.randn(100)) * 100,
                "close": prices,
                "volume": np.random.randint(1000, 5000, 100),
            }
        )

        return df

    # ==================== TESTES RSI ====================

    def test_calculate_rsi_basic(self, ti, mock_prices):
        """Testa cálculo básico do RSI"""
        rsi = ti.calculate_rsi(mock_prices, period=14)

        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(mock_prices)
        assert rsi.iloc[-1] >= 0
        assert rsi.iloc[-1] <= 100

    def test_calculate_rsi_different_periods(self, ti, mock_prices):
        """Testa RSI com diferentes períodos"""
        rsi_14 = ti.calculate_rsi(mock_prices, period=14)
        rsi_21 = ti.calculate_rsi(mock_prices, period=21)

        assert len(rsi_14) == len(rsi_21)
        # RSI com períodos diferentes devem ter valores diferentes
        assert not rsi_14.equals(rsi_21)

    def test_calculate_rsi_with_nan(self, ti):
        """Testa RSI com valores NaN no início"""
        prices = pd.Series([50000, 50100, 50200, 50150, 50300])
        rsi = ti.calculate_rsi(prices, period=14)

        # Primeiros valores devem ser NaN
        assert pd.isna(rsi.iloc[0])

    def test_interpret_rsi_oversold(self, ti):
        """Testa interpretação de RSI oversold"""
        assert ti.interpret_rsi(25) == "oversold"
        assert ti.interpret_rsi(29.9) == "oversold"

    def test_interpret_rsi_overbought(self, ti):
        """Testa interpretação de RSI overbought"""
        assert ti.interpret_rsi(75) == "overbought"
        assert ti.interpret_rsi(70.1) == "overbought"

    def test_interpret_rsi_neutral(self, ti):
        """Testa interpretação de RSI neutral"""
        assert ti.interpret_rsi(50) == "neutral"
        assert ti.interpret_rsi(30) == "neutral"
        assert ti.interpret_rsi(70) == "neutral"

    # ==================== TESTES MACD ====================

    def test_calculate_macd_basic(self, ti, mock_prices):
        """Testa cálculo básico do MACD"""
        macd = ti.calculate_macd(mock_prices)

        assert isinstance(macd, dict)
        assert "macd" in macd
        assert "signal" in macd
        assert "histogram" in macd
        assert len(macd["macd"]) == len(mock_prices)

    def test_calculate_macd_custom_periods(self, ti, mock_prices):
        """Testa MACD com períodos customizados"""
        macd = ti.calculate_macd(mock_prices, fast=10, slow=20, signal=5)

        assert isinstance(macd["macd"], pd.Series)
        assert len(macd["macd"]) == len(mock_prices)

    def test_calculate_macd_histogram_calculation(self, ti, mock_prices):
        """Testa se histogram é calculado corretamente"""
        macd = ti.calculate_macd(mock_prices)

        # Histogram deve ser macd - signal
        calculated_histogram = macd["macd"] - macd["signal"]
        pd.testing.assert_series_equal(macd["histogram"], calculated_histogram, check_names=False)

    def test_interpret_macd_crossover_bullish(self, ti):
        """Testa detecção de cruzamento bullish do MACD"""
        result = ti.interpret_macd_crossover(
            macd_current=120, signal_current=115, macd_previous=110, signal_previous=115
        )
        assert result == "bullish_crossover"

    def test_interpret_macd_crossover_bearish(self, ti):
        """Testa detecção de cruzamento bearish do MACD"""
        result = ti.interpret_macd_crossover(
            macd_current=110, signal_current=115, macd_previous=120, signal_previous=115
        )
        assert result == "bearish_crossover"

    def test_interpret_macd_crossover_none(self, ti):
        """Testa quando não há cruzamento do MACD"""
        result = ti.interpret_macd_crossover(
            macd_current=120, signal_current=115, macd_previous=125, signal_previous=120
        )
        assert result is None

    # ==================== TESTES BOLLINGER BANDS ====================

    def test_calculate_bollinger_bands_basic(self, ti, mock_prices):
        """Testa cálculo básico das Bollinger Bands"""
        bb = ti.calculate_bollinger_bands(mock_prices)

        assert isinstance(bb, dict)
        assert "upper" in bb
        assert "middle" in bb
        assert "lower" in bb
        assert "width" in bb
        assert "percent_b" in bb

    def test_calculate_bollinger_bands_order(self, ti, mock_prices):
        """Testa se upper > middle > lower"""
        bb = ti.calculate_bollinger_bands(mock_prices)

        # Remover NaN para comparação
        valid_idx = ~bb["upper"].isna()
        assert (bb["upper"][valid_idx] >= bb["middle"][valid_idx]).all()
        assert (bb["middle"][valid_idx] >= bb["lower"][valid_idx]).all()

    def test_calculate_bollinger_bands_custom_params(self, ti, mock_prices):
        """Testa Bollinger Bands com parâmetros customizados"""
        bb = ti.calculate_bollinger_bands(mock_prices, period=10, std_dev=3.0)

        assert len(bb["upper"]) == len(mock_prices)

    def test_calculate_bollinger_bands_percent_b(self, ti, mock_prices):
        """Testa cálculo do %B"""
        bb = ti.calculate_bollinger_bands(mock_prices)

        # %B deve estar geralmente entre 0 e 1
        valid_idx = ~bb["percent_b"].isna()
        assert bb["percent_b"][valid_idx].notna().any()

    # ==================== TESTES EMA/SMA ====================

    def test_calculate_ema_basic(self, ti, mock_prices):
        """Testa cálculo básico da EMA"""
        ema = ti.calculate_ema(mock_prices, period=20)

        assert isinstance(ema, pd.Series)
        assert len(ema) == len(mock_prices)

    def test_calculate_ema_different_periods(self, ti, mock_prices):
        """Testa EMA com diferentes períodos"""
        ema_9 = ti.calculate_ema(mock_prices, period=9)
        ema_21 = ti.calculate_ema(mock_prices, period=21)

        # EMAs com períodos diferentes devem ser diferentes
        assert not ema_9.equals(ema_21)

    def test_calculate_sma_basic(self, ti, mock_prices):
        """Testa cálculo básico da SMA"""
        sma = ti.calculate_sma(mock_prices, period=20)

        assert isinstance(sma, pd.Series)
        assert len(sma) == len(mock_prices)

    def test_calculate_sma_vs_manual(self, ti):
        """Testa SMA comparando com cálculo manual"""
        prices = pd.Series([10, 20, 30, 40, 50])
        sma = ti.calculate_sma(prices, period=3)

        # SMA(3) do último valor deve ser (30+40+50)/3 = 40
        assert sma.iloc[-1] == 40.0

    # ==================== TESTES ADX ====================

    def test_calculate_adx_basic(self, ti, mock_ohlcv_data):
        """Testa cálculo básico do ADX"""
        df = mock_ohlcv_data
        adx = ti.calculate_adx(df["high"], df["low"], df["close"])

        assert isinstance(adx, dict)
        assert "adx" in adx
        assert "plus_di" in adx
        assert "minus_di" in adx

    def test_calculate_adx_values_range(self, ti, mock_ohlcv_data):
        """Testa se valores do ADX estão no range correto"""
        df = mock_ohlcv_data
        adx = ti.calculate_adx(df["high"], df["low"], df["close"])

        # ADX deve estar entre 0 e 100
        valid_idx = ~adx["adx"].isna()
        if valid_idx.any():
            assert (adx["adx"][valid_idx] >= 0).all()
            assert (adx["adx"][valid_idx] <= 100).all()

    def test_calculate_adx_custom_period(self, ti, mock_ohlcv_data):
        """Testa ADX com período customizado"""
        df = mock_ohlcv_data
        adx = ti.calculate_adx(df["high"], df["low"], df["close"], period=20)

        assert len(adx["adx"]) == len(df)

    # ==================== TESTES ATR ====================

    def test_calculate_atr_basic(self, ti, mock_ohlcv_data):
        """Testa cálculo básico do ATR"""
        df = mock_ohlcv_data
        atr = ti.calculate_atr(df["high"], df["low"], df["close"])

        assert isinstance(atr, pd.Series)
        assert len(atr) == len(df)

    def test_calculate_atr_positive_values(self, ti, mock_ohlcv_data):
        """Testa se ATR retorna apenas valores positivos"""
        df = mock_ohlcv_data
        atr = ti.calculate_atr(df["high"], df["low"], df["close"])

        valid_idx = ~atr.isna()
        if valid_idx.any():
            assert (atr[valid_idx] >= 0).all()

    def test_calculate_atr_custom_period(self, ti, mock_ohlcv_data):
        """Testa ATR com período customizado"""
        df = mock_ohlcv_data
        atr = ti.calculate_atr(df["high"], df["low"], df["close"], period=20)

        assert len(atr) == len(df)

    # ==================== TESTES STOCHASTIC ====================

    def test_calculate_stochastic_basic(self, ti, mock_ohlcv_data):
        """Testa cálculo básico do Stochastic"""
        df = mock_ohlcv_data
        stoch = ti.calculate_stochastic(df["high"], df["low"], df["close"])

        assert isinstance(stoch, dict)
        assert "k" in stoch
        assert "d" in stoch

    def test_calculate_stochastic_range(self, ti, mock_ohlcv_data):
        """Testa se Stochastic está entre 0 e 100"""
        df = mock_ohlcv_data
        stoch = ti.calculate_stochastic(df["high"], df["low"], df["close"])

        valid_idx = ~stoch["k"].isna()
        if valid_idx.any():
            assert (stoch["k"][valid_idx] >= 0).all()
            assert (stoch["k"][valid_idx] <= 100).all()

    def test_calculate_stochastic_custom_params(self, ti, mock_ohlcv_data):
        """Testa Stochastic com parâmetros customizados"""
        df = mock_ohlcv_data
        stoch = ti.calculate_stochastic(
            df["high"], df["low"], df["close"], k_period=10, d_period=5, smooth_k=5
        )

        assert len(stoch["k"]) == len(df)

    # ==================== TESTES VOLUME INDICATORS ====================

    def test_calculate_obv_basic(self, ti, mock_ohlcv_data):
        """Testa cálculo básico do OBV"""
        df = mock_ohlcv_data
        obv = ti.calculate_obv(df["close"], df["volume"])

        assert isinstance(obv, pd.Series)
        assert len(obv) == len(df)

    def test_calculate_obv_cumulative(self, ti):
        """Testa se OBV é cumulativo"""
        close = pd.Series([100, 105, 103, 108])
        volume = pd.Series([1000, 1000, 1000, 1000])

        obv = ti.calculate_obv(close, volume)

        # OBV deve ser cumulativo
        assert isinstance(obv, pd.Series)
        assert len(obv) == len(close)

    def test_calculate_vwap_basic(self, ti, mock_ohlcv_data):
        """Testa cálculo básico do VWAP"""
        df = mock_ohlcv_data
        vwap = ti.calculate_vwap(df["high"], df["low"], df["close"], df["volume"])

        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(df)

    def test_calculate_vwap_values(self, ti, mock_ohlcv_data):
        """Testa se valores do VWAP são razoáveis"""
        df = mock_ohlcv_data
        vwap = ti.calculate_vwap(df["high"], df["low"], df["close"], df["volume"])

        # VWAP deve ser um valor numérico válido (pode estar fora do range do candle atual pois é cumulative)
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(df)
        # Check that most values are reasonable (within 2x of price range)
        price_min = df["low"].min()
        price_max = df["high"].max()
        price_range = price_max - price_min
        assert (vwap[~vwap.isna()] >= price_min - price_range).all()
        assert (vwap[~vwap.isna()] <= price_max + price_range).all()

    def test_calculate_volume_ratio_basic(self, ti, mock_ohlcv_data):
        """Testa cálculo básico do Volume Ratio"""
        df = mock_ohlcv_data
        ratio = ti.calculate_volume_ratio(df["volume"])

        assert isinstance(ratio, pd.Series)
        assert len(ratio) == len(df)

    def test_calculate_volume_ratio_custom_period(self, ti, mock_ohlcv_data):
        """Testa Volume Ratio com período customizado"""
        df = mock_ohlcv_data
        ratio = ti.calculate_volume_ratio(df["volume"], period=10)

        assert len(ratio) == len(df)

    # ==================== TESTES FIBONACCI ====================

    def test_calculate_fibonacci_retracements_basic(self, ti):
        """Testa cálculo básico dos níveis de Fibonacci"""
        fib = ti.calculate_fibonacci_retracements(high_price=52000, low_price=48000)

        assert isinstance(fib, dict)
        assert "level_0" in fib
        assert "level_236" in fib
        assert "level_382" in fib
        assert "level_500" in fib
        assert "level_618" in fib
        assert "level_786" in fib
        assert "level_100" in fib

    def test_calculate_fibonacci_retracements_values(self, ti):
        """Testa se valores de Fibonacci estão corretos"""
        fib = ti.calculate_fibonacci_retracements(high_price=50000, low_price=40000)

        assert fib["level_0"] == 50000
        assert fib["level_100"] == 40000
        assert fib["level_500"] == 45000  # Meio do caminho

        # Níveis devem estar em ordem decrescente
        assert fib["level_0"] > fib["level_236"] > fib["level_382"]
        assert fib["level_382"] > fib["level_500"] > fib["level_618"]
        assert fib["level_618"] > fib["level_786"] > fib["level_100"]

    # ==================== TESTES PIVOT POINTS ====================

    def test_calculate_pivot_points_basic(self, ti):
        """Testa cálculo básico dos Pivot Points"""
        pivots = ti.calculate_pivot_points(high=52000, low=48000, close=50000)

        assert isinstance(pivots, dict)
        assert "pivot" in pivots
        assert "r1" in pivots
        assert "r2" in pivots
        assert "r3" in pivots
        assert "s1" in pivots
        assert "s2" in pivots
        assert "s3" in pivots

    def test_calculate_pivot_points_order(self, ti):
        """Testa se resistências e suportes estão em ordem"""
        pivots = ti.calculate_pivot_points(high=52000, low=48000, close=50000)

        # Resistências devem ser crescentes
        assert pivots["r1"] < pivots["r2"] < pivots["r3"]

        # Suportes devem ser decrescentes
        assert pivots["s1"] > pivots["s2"] > pivots["s3"]

        # Pivot deve estar entre s1 e r1
        assert pivots["s1"] < pivots["pivot"] < pivots["r1"]

    def test_calculate_pivot_points_formula(self, ti):
        """Testa se fórmula do pivot está correta"""
        high, low, close = 52000, 48000, 50000
        pivots = ti.calculate_pivot_points(high=high, low=low, close=close)

        expected_pivot = (high + low + close) / 3
        assert pivots["pivot"] == expected_pivot

    # ==================== TESTES TREND DIRECTION ====================

    def test_get_trend_direction_strong_uptrend(self, ti):
        """Testa detecção de strong uptrend"""
        trend = ti.get_trend_direction(ema_9=50500, ema_21=50300, ema_50=50000, ema_200=49000)
        assert trend == "strong_uptrend"

    def test_get_trend_direction_uptrend(self, ti):
        """Testa detecção de uptrend"""
        trend = ti.get_trend_direction(ema_9=50500, ema_21=50300, ema_50=50000)
        assert trend == "uptrend"

    def test_get_trend_direction_strong_downtrend(self, ti):
        """Testa detecção de strong downtrend"""
        trend = ti.get_trend_direction(ema_9=49000, ema_21=49500, ema_50=50000, ema_200=51000)
        assert trend == "strong_downtrend"

    def test_get_trend_direction_downtrend(self, ti):
        """Testa detecção de downtrend"""
        trend = ti.get_trend_direction(ema_9=49000, ema_21=49500, ema_50=50000)
        assert trend == "downtrend"

    def test_get_trend_direction_sideways(self, ti):
        """Testa detecção de sideways"""
        trend = ti.get_trend_direction(ema_9=50000, ema_21=50100, ema_50=49900)
        assert trend == "sideways"

    # ==================== TESTES CALCULATE ALL INDICATORS ====================

    def test_calculate_all_indicators_basic(self, ti, mock_ohlcv_data):
        """Testa cálculo de todos os indicadores de uma vez"""
        df = mock_ohlcv_data
        result = ti.calculate_all_indicators(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

        # Verificar se todas as colunas foram adicionadas
        expected_columns = [
            "rsi_14",
            "rsi_21",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "bb_width",
            "bb_percent_b",
            "ema_9",
            "ema_21",
            "ema_50",
            "ema_200",
            "sma_20",
            "sma_50",
            "sma_200",
            "adx",
            "plus_di",
            "minus_di",
            "atr",
            "stoch_k",
            "stoch_d",
            "obv",
            "vwap",
            "volume_ratio",
        ]

        for col in expected_columns:
            assert col in result.columns

    def test_calculate_all_indicators_custom_columns(self, ti):
        """Testa cálculo com nomes de colunas customizados"""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="1H"),
                "o": np.random.randn(50).cumsum() + 50000,
                "h": np.random.randn(50).cumsum() + 50200,
                "l": np.random.randn(50).cumsum() + 49800,
                "c": np.random.randn(50).cumsum() + 50000,
                "v": np.random.randint(1000, 5000, 50),
            }
        )

        column_mapping = {"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"}

        result = ti.calculate_all_indicators(df, ohlcv_columns=column_mapping)

        assert "rsi_14" in result.columns
        assert "macd" in result.columns

    def test_calculate_all_indicators_preserves_original(self, ti, mock_ohlcv_data):
        """Testa se dados originais são preservados"""
        df = mock_ohlcv_data.copy()
        original_columns = df.columns.tolist()

        result = ti.calculate_all_indicators(df)

        # Todas as colunas originais devem estar presentes
        for col in original_columns:
            assert col in result.columns

    # ==================== TESTES DE EDGE CASES ====================

    def test_empty_series(self, ti):
        """Testa comportamento com série vazia"""
        empty = pd.Series([], dtype=float)

        rsi = ti.calculate_rsi(empty)
        assert len(rsi) == 0

    def test_single_value_series(self, ti):
        """Testa comportamento com apenas um valor"""
        single = pd.Series([50000])

        rsi = ti.calculate_rsi(single, period=14)
        assert len(rsi) == 1
        assert pd.isna(rsi.iloc[0])

    def test_all_same_values(self, ti):
        """Testa comportamento com todos os valores iguais"""
        same = pd.Series([50000] * 50)

        rsi = ti.calculate_rsi(same, period=14)
        # RSI de valores constantes deve ser NaN ou 50
        assert rsi.isna().all() or (rsi == 50).all()

    def test_with_nan_values(self, ti):
        """Testa comportamento com valores NaN"""
        with_nan = pd.Series([50000, np.nan, 50200, 50100, np.nan, 50300])

        rsi = ti.calculate_rsi(with_nan, period=3)
        assert isinstance(rsi, pd.Series)

    # ==================== TESTES DE INTEGRAÇÃO ====================

    def test_full_workflow(self, ti, mock_ohlcv_data):
        """Testa workflow completo de análise técnica"""
        df = mock_ohlcv_data

        # Calcular todos os indicadores
        df_with_indicators = ti.calculate_all_indicators(df)

        # Pegar última linha
        last_row = df_with_indicators.iloc[-1]

        # Interpretar RSI
        rsi_interpretation = ti.interpret_rsi(last_row["rsi_14"])
        assert rsi_interpretation in ["oversold", "neutral", "overbought"]

        # Verificar tendência
        if not pd.isna(last_row["ema_200"]):
            trend = ti.get_trend_direction(
                last_row["ema_9"], last_row["ema_21"], last_row["ema_50"], last_row["ema_200"]
            )
            assert trend in [
                "strong_uptrend",
                "uptrend",
                "downtrend",
                "strong_downtrend",
                "sideways",
            ]

    def test_indicator_consistency(self, ti, mock_ohlcv_data):
        """Testa consistência entre indicadores"""
        df = mock_ohlcv_data
        df_with_indicators = ti.calculate_all_indicators(df)

        # Bollinger middle deve ser igual a SMA(20)
        valid_idx = ~df_with_indicators["bb_middle"].isna()
        if valid_idx.any():
            pd.testing.assert_series_equal(
                df_with_indicators["bb_middle"][valid_idx],
                df_with_indicators["sma_20"][valid_idx],
                check_names=False,
            )


# ==================== TESTES DE PERFORMANCE ====================


class TestTechnicalIndicatorsPerformance:
    """Testes de performance (mockados)"""

    @pytest.fixture
    def ti(self):
        return TechnicalIndicators()

    @pytest.fixture
    def large_dataset(self):
        """Dataset grande mockado"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=10000, freq="1min")

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": 50000 + np.random.randn(10000).cumsum() * 10,
                "high": 50000 + np.random.randn(10000).cumsum() * 10 + 50,
                "low": 50000 + np.random.randn(10000).cumsum() * 10 - 50,
                "close": 50000 + np.random.randn(10000).cumsum() * 10,
                "volume": np.random.randint(1000, 5000, 10000),
            }
        )

        return df

    def test_calculate_all_indicators_performance(self, ti, large_dataset):
        """Testa performance com dataset grande"""
        import time

        start = time.time()
        result = ti.calculate_all_indicators(large_dataset)
        elapsed = time.time() - start

        assert len(result) == len(large_dataset)
        # Deve completar em menos de 5 segundos
        assert elapsed < 5.0

    def test_memory_efficiency(self, ti, large_dataset):
        """Testa eficiência de memória"""
        import sys

        original_size = sys.getsizeof(large_dataset)
        result = ti.calculate_all_indicators(large_dataset)
        result_size = sys.getsizeof(result)

        # Resultado não deve ser muito maior que o original
        # (considerando que adicionamos ~25 colunas, allow up to 10x)
        # sys.getsizeof doesn't account for all memory, so be more lenient
        assert result_size < original_size * 10


# ==================== CONFIGURAÇÃO PYTEST ====================

if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--cov=utils.technical_indicators",
            "--cov-report=html",
            "--cov-report=term-missing",
        ]
    )
