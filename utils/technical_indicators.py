"""
CeciAI - Technical Indicators Module

Calcula indicadores técnicos para análise de mercado:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- EMA (Exponential Moving Average)
- SMA (Simple Moving Average)
- ADX (Average Directional Index)
- ATR (Average True Range)
- Stochastic Oscillator
- Fibonacci Retracements
- Pivot Points
- OBV (On-Balance Volume)
- VWAP (Volume Weighted Average Price)

Autor: CeciAI Team
Data: 2025-10-07
"""


import numpy as np
import pandas as pd


class TechnicalIndicators:
    """
    Classe para calcular indicadores técnicos.
    """

    def __init__(self):
        """Inicializa o calculador de indicadores técnicos."""

    # ==================== MOMENTUM INDICATORS ====================

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calcula o RSI (Relative Strength Index).

        Args:
            prices: Série de preços (geralmente close)
            period: Período para cálculo (padrão: 14)

        Returns:
            Série com valores de RSI (0-100)

        Interpretação:
            RSI > 70: Overbought (sobrecomprado)
            RSI < 30: Oversold (sobrevendido)
            RSI 30-70: Neutro
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3,
    ) -> dict[str, pd.Series]:
        """
        Calcula o Stochastic Oscillator.

        Args:
            high: Série de preços máximos
            low: Série de preços mínimos
            close: Série de preços de fechamento
            k_period: Período para %K (padrão: 14)
            d_period: Período para %D (padrão: 3)
            smooth_k: Período de suavização para %K (padrão: 3)

        Returns:
            Dict com 'k' e 'd' (valores 0-100)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_fast = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k = k_fast.rolling(window=smooth_k).mean()
        d = k.rolling(window=d_period).mean()

        return {"k": k, "d": d}

    # ==================== TREND INDICATORS ====================

    @staticmethod
    def calculate_macd(
        prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> dict[str, pd.Series]:
        """
        Calcula o MACD (Moving Average Convergence Divergence).

        Args:
            prices: Série de preços (geralmente close)
            fast: Período EMA rápida (padrão: 12)
            slow: Período EMA lenta (padrão: 26)
            signal: Período da linha de sinal (padrão: 9)

        Returns:
            Dict com 'macd', 'signal', 'histogram'

        Interpretação:
            MACD > Signal: Sinal de alta (bullish)
            MACD < Signal: Sinal de baixa (bearish)
            Histogram > 0: Momentum positivo
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line

        return {"macd": macd, "signal": signal_line, "histogram": histogram}

    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """
        Calcula a EMA (Exponential Moving Average).

        Args:
            prices: Série de preços
            period: Período para cálculo

        Returns:
            Série com valores de EMA
        """
        return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """
        Calcula a SMA (Simple Moving Average).

        Args:
            prices: Série de preços
            period: Período para cálculo

        Returns:
            Série com valores de SMA
        """
        return prices.rolling(window=period).mean()

    @staticmethod
    def calculate_adx(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> dict[str, pd.Series]:
        """
        Calcula o ADX (Average Directional Index).

        Args:
            high: Série de preços máximos
            low: Série de preços mínimos
            close: Série de preços de fechamento
            period: Período para cálculo (padrão: 14)

        Returns:
            Dict com 'adx', 'plus_di', 'minus_di'

        Interpretação:
            ADX > 25: Tendência forte
            ADX < 20: Tendência fraca ou lateral
        """
        # True Range
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_dm = pd.Series(plus_dm, index=high.index).rolling(window=period).mean()
        minus_dm = pd.Series(minus_dm, index=high.index).rolling(window=period).mean()

        # Directional Indicators
        plus_di = 100 * (plus_dm / atr)
        minus_di = 100 * (minus_dm / atr)

        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}

    # ==================== VOLATILITY INDICATORS ====================

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> dict[str, pd.Series]:
        """
        Calcula as Bollinger Bands.

        Args:
            prices: Série de preços (geralmente close)
            period: Período para SMA (padrão: 20)
            std_dev: Número de desvios padrão (padrão: 2.0)

        Returns:
            Dict com 'upper', 'middle', 'lower', 'width', 'percent_b'

        Interpretação:
            Preço na banda inferior: Possível reversão de alta
            Preço na banda superior: Possível reversão de baixa
            Largura aumentando: Volatilidade aumentando
        """
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        width = (upper - lower) / sma

        # %B: Posição do preço dentro das bandas
        percent_b = (prices - lower) / (upper - lower)

        return {
            "upper": upper,
            "middle": sma,
            "lower": lower,
            "width": width,
            "percent_b": percent_b,
        }

    @staticmethod
    def calculate_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Calcula o ATR (Average True Range).

        Args:
            high: Série de preços máximos
            low: Série de preços mínimos
            close: Série de preços de fechamento
            period: Período para cálculo (padrão: 14)

        Returns:
            Série com valores de ATR

        Interpretação:
            ATR alto: Alta volatilidade
            ATR baixo: Baixa volatilidade
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    # ==================== VOLUME INDICATORS ====================

    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calcula o OBV (On-Balance Volume).

        Args:
            close: Série de preços de fechamento
            volume: Série de volume

        Returns:
            Série com valores de OBV

        Interpretação:
            OBV crescente: Pressão compradora
            OBV decrescente: Pressão vendedora
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    @staticmethod
    def calculate_vwap(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """
        Calcula o VWAP (Volume Weighted Average Price).

        Args:
            high: Série de preços máximos
            low: Série de preços mínimos
            close: Série de preços de fechamento
            volume: Série de volume

        Returns:
            Série com valores de VWAP

        Interpretação:
            Preço > VWAP: Tendência de alta
            Preço < VWAP: Tendência de baixa
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap

    @staticmethod
    def calculate_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Calcula a razão entre volume atual e volume médio.

        Args:
            volume: Série de volume
            period: Período para média (padrão: 20)

        Returns:
            Série com razão de volume

        Interpretação:
            Ratio > 1.5: Volume alto
            Ratio < 0.5: Volume baixo
        """
        avg_volume = volume.rolling(window=period).mean()
        ratio = volume / avg_volume
        return ratio

    # ==================== SUPPORT/RESISTANCE ====================

    @staticmethod
    def calculate_fibonacci_retracements(high_price: float, low_price: float) -> dict[str, float]:
        """
        Calcula os níveis de retração de Fibonacci.

        Args:
            high_price: Preço máximo do movimento
            low_price: Preço mínimo do movimento

        Returns:
            Dict com níveis de Fibonacci
        """
        diff = high_price - low_price

        return {
            "level_0": high_price,
            "level_236": high_price - (diff * 0.236),
            "level_382": high_price - (diff * 0.382),
            "level_500": high_price - (diff * 0.500),
            "level_618": high_price - (diff * 0.618),
            "level_786": high_price - (diff * 0.786),
            "level_100": low_price,
        }

    @staticmethod
    def calculate_pivot_points(high: float, low: float, close: float) -> dict[str, float]:
        """
        Calcula os Pivot Points (suporte e resistência).

        Args:
            high: Preço máximo do período anterior
            low: Preço mínimo do período anterior
            close: Preço de fechamento do período anterior

        Returns:
            Dict com pivot, suportes e resistências
        """
        pivot = (high + low + close) / 3

        return {
            "pivot": pivot,
            "r1": (2 * pivot) - low,
            "r2": pivot + (high - low),
            "r3": high + 2 * (pivot - low),
            "s1": (2 * pivot) - high,
            "s2": pivot - (high - low),
            "s3": low - 2 * (high - pivot),
        }

    # ==================== HELPER METHODS ====================

    @staticmethod
    def get_trend_direction(
        ema_9: float, ema_21: float, ema_50: float, ema_200: float | None = None
    ) -> str:
        """
        Determina a direção da tendência com base nas EMAs.

        Args:
            ema_9: EMA de 9 períodos
            ema_21: EMA de 21 períodos
            ema_50: EMA de 50 períodos
            ema_200: EMA de 200 períodos (opcional)

        Returns:
            'strong_uptrend', 'uptrend', 'downtrend', 'strong_downtrend', 'sideways'
        """
        if ema_9 > ema_21 > ema_50:
            if ema_200 and ema_50 > ema_200:
                return "strong_uptrend"
            return "uptrend"
        elif ema_9 < ema_21 < ema_50:
            if ema_200 and ema_50 < ema_200:
                return "strong_downtrend"
            return "downtrend"
        else:
            return "sideways"

    @staticmethod
    def interpret_rsi(rsi: float) -> str:
        """
        Interpreta o valor do RSI.

        Args:
            rsi: Valor do RSI (0-100)

        Returns:
            'oversold', 'neutral', 'overbought'
        """
        if rsi < 30:
            return "oversold"
        elif rsi > 70:
            return "overbought"
        else:
            return "neutral"

    @staticmethod
    def interpret_macd_crossover(
        macd_current: float, signal_current: float, macd_previous: float, signal_previous: float
    ) -> str | None:
        """
        Detecta cruzamentos do MACD.

        Args:
            macd_current: MACD atual
            signal_current: Signal atual
            macd_previous: MACD anterior
            signal_previous: Signal anterior

        Returns:
            'bullish_crossover', 'bearish_crossover', ou None
        """
        if macd_previous <= signal_previous and macd_current > signal_current:
            return "bullish_crossover"
        elif macd_previous >= signal_previous and macd_current < signal_current:
            return "bearish_crossover"
        return None

    # ==================== ALL-IN-ONE METHOD ====================

    def calculate_all_indicators(
        self, df: pd.DataFrame, ohlcv_columns: dict[str, str] | None = None
    ) -> pd.DataFrame:
        """
        Calcula todos os indicadores técnicos de uma vez.

        Args:
            df: DataFrame com dados OHLCV
            ohlcv_columns: Dict mapeando nomes das colunas
                          {'open': 'open', 'high': 'high', 'low': 'low',
                           'close': 'close', 'volume': 'volume'}

        Returns:
            DataFrame original com todas as colunas de indicadores adicionadas
        """
        if ohlcv_columns is None:
            ohlcv_columns = {
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }

        df = df.copy()

        # Extrair colunas
        close = df[ohlcv_columns["close"]]
        high = df[ohlcv_columns["high"]]
        low = df[ohlcv_columns["low"]]
        volume = df[ohlcv_columns["volume"]]

        # RSI
        df["rsi_14"] = self.calculate_rsi(close, 14)
        df["rsi_21"] = self.calculate_rsi(close, 21)

        # MACD
        macd = self.calculate_macd(close)
        df["macd"] = macd["macd"]
        df["macd_signal"] = macd["signal"]
        df["macd_histogram"] = macd["histogram"]

        # Bollinger Bands
        bb = self.calculate_bollinger_bands(close)
        df["bb_upper"] = bb["upper"]
        df["bb_middle"] = bb["middle"]
        df["bb_lower"] = bb["lower"]
        df["bb_width"] = bb["width"]
        df["bb_percent_b"] = bb["percent_b"]

        # EMAs
        df["ema_9"] = self.calculate_ema(close, 9)
        df["ema_21"] = self.calculate_ema(close, 21)
        df["ema_50"] = self.calculate_ema(close, 50)
        df["ema_200"] = self.calculate_ema(close, 200)

        # SMAs
        df["sma_20"] = self.calculate_sma(close, 20)
        df["sma_50"] = self.calculate_sma(close, 50)
        df["sma_200"] = self.calculate_sma(close, 200)

        # ADX
        adx = self.calculate_adx(high, low, close)
        df["adx"] = adx["adx"]
        df["plus_di"] = adx["plus_di"]
        df["minus_di"] = adx["minus_di"]

        # ATR
        df["atr"] = self.calculate_atr(high, low, close)

        # Stochastic
        stoch = self.calculate_stochastic(high, low, close)
        df["stoch_k"] = stoch["k"]
        df["stoch_d"] = stoch["d"]

        # Volume indicators
        df["obv"] = self.calculate_obv(close, volume)
        df["vwap"] = self.calculate_vwap(high, low, close, volume)
        df["volume_ratio"] = self.calculate_volume_ratio(volume)

        return df


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Exemplo de uso
    print("CeciAI - Technical Indicators Module")
    print("=" * 50)

    # Criar dados de exemplo
    dates = pd.date_range("2024-01-01", periods=100, freq="1H")
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 50000 + np.random.randn(100).cumsum() * 100,
            "high": 50000 + np.random.randn(100).cumsum() * 100 + 200,
            "low": 50000 + np.random.randn(100).cumsum() * 100 - 200,
            "close": 50000 + np.random.randn(100).cumsum() * 100,
            "volume": np.random.randint(1000, 5000, 100),
        }
    )

    # Calcular indicadores
    ti = TechnicalIndicators()
    df_with_indicators = ti.calculate_all_indicators(df)

    # Mostrar últimas 5 linhas
    print("\nÚltimas 5 linhas com indicadores:")
    print(df_with_indicators.tail())

    # Interpretar últimos valores
    last_row = df_with_indicators.iloc[-1]
    print("\n📊 Análise Técnica (último candle):")
    print(f"RSI(14): {last_row['rsi_14']:.2f} - {ti.interpret_rsi(last_row['rsi_14'])}")
    print(f"MACD: {last_row['macd']:.2f}")
    print(f"MACD Signal: {last_row['macd_signal']:.2f}")
    print(f"ADX: {last_row['adx']:.2f}")
    print(f"ATR: {last_row['atr']:.2f}")

    trend = ti.get_trend_direction(
        last_row["ema_9"], last_row["ema_21"], last_row["ema_50"], last_row["ema_200"]
    )
    print(f"Tendência: {trend}")


# ==================== FUNÇÕES AUXILIARES PARA IMPORTAÇÃO ====================


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wrapper para TechnicalIndicators.calculate_rsi"""
    return TechnicalIndicators.calculate_rsi(prices, period)


def calculate_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wrapper para TechnicalIndicators.calculate_macd"""
    return TechnicalIndicators.calculate_macd(prices, fast, slow, signal)


def calculate_bollinger_bands(
    prices: pd.Series, period: int = 20, std_dev: int = 2
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wrapper para TechnicalIndicators.calculate_bollinger_bands"""
    return TechnicalIndicators.calculate_bollinger_bands(prices, period, std_dev)


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Wrapper para TechnicalIndicators.calculate_ema"""
    return TechnicalIndicators.calculate_ema(prices, period)


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Wrapper para TechnicalIndicators.calculate_sma"""
    return TechnicalIndicators.calculate_sma(prices, period)


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wrapper para TechnicalIndicators.calculate_adx"""
    return TechnicalIndicators.calculate_adx(high, low, close, period)


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wrapper para TechnicalIndicators.calculate_atr"""
    return TechnicalIndicators.calculate_atr(high, low, close, period)


def calculate_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Wrapper para TechnicalIndicators.calculate_stochastic"""
    return TechnicalIndicators.calculate_stochastic(high, low, close, k_period, d_period)


def calculate_fibonacci_retracements(high: float, low: float) -> dict[str, float]:
    """Wrapper para TechnicalIndicators.calculate_fibonacci_retracements"""
    return TechnicalIndicators.calculate_fibonacci_retracements(high, low)


def calculate_pivot_points(high: float, low: float, close: float) -> dict[str, float]:
    """Wrapper para TechnicalIndicators.calculate_pivot_points"""
    return TechnicalIndicators.calculate_pivot_points(high, low, close)


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Wrapper para TechnicalIndicators.calculate_obv"""
    return TechnicalIndicators.calculate_obv(close, volume)


def calculate_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Wrapper para TechnicalIndicators.calculate_vwap"""
    return TechnicalIndicators.calculate_vwap(high, low, close, volume)


def identify_trend(ema_9: float, ema_21: float, ema_50: float, ema_200: float) -> str:
    """Wrapper para TechnicalIndicators.identify_trend"""
    return TechnicalIndicators.identify_trend(ema_9, ema_21, ema_50, ema_200)
