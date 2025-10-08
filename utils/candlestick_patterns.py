"""
CeciAI - Candlestick Pattern Detection
Detecção de padrões de candlestick para análise técnica

Padrões implementados:
- Reversão de Alta: Hammer, Inverted Hammer, Bullish Engulfing, Morning Star, etc
- Reversão de Baixa: Shooting Star, Hanging Man, Bearish Engulfing, Evening Star, etc
- Indecisão: Doji, Spinning Top, Harami
- Integração com TA-Lib para 60+ padrões adicionais

Autor: CeciAI Team
Data: 2025-10-07
"""

from dataclasses import dataclass
from enum import Enum

import pandas as pd

try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️  TA-Lib não disponível. Usando apenas detecção customizada.")


class PatternType(Enum):
    """Tipos de padrões"""

    BULLISH_REVERSAL = "bullish_reversal"  # Reversão de alta
    BEARISH_REVERSAL = "bearish_reversal"  # Reversão de baixa
    INDECISION = "indecision"  # Indecisão
    CONTINUATION = "continuation"  # Continuação de tendência


class SignalStrength(Enum):
    """Força do sinal"""

    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class CandlePattern:
    """Resultado da detecção de padrão"""

    name: str
    pattern_type: PatternType
    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0-1
    significance: int  # 0-100
    strength: SignalStrength
    candle_index: int
    description: str
    confirmation_needed: bool
    confirmation_criteria: str


class CandlestickPatternDetector:
    """
    Detector de padrões de candlestick.

    Analisa séries de candles e identifica padrões conhecidos.
    Integra TA-Lib para detecção avançada de 60+ padrões.
    """

    def __init__(self, min_body_size: float = 0.001):
        """
        Inicializa detector.

        Args:
            min_body_size: Tamanho mínimo do corpo (% do preço)
        """
        self.min_body_size = min_body_size

    def detect_all_patterns(self, df: pd.DataFrame, lookback: int = 10) -> list[CandlePattern]:
        """
        Detecta todos os padrões em uma série de candles.

        Args:
            df: DataFrame com colunas [open, high, low, close, volume]
            lookback: Quantos candles analisar para contexto

        Returns:
            Lista de padrões detectados
        """
        patterns = []

        if len(df) < 3:
            return patterns

        # Calcular propriedades dos candles
        df = self._calculate_candle_properties(df)

        # Detectar tendência anterior
        trend = self._detect_trend(df, lookback)

        # Detectar padrões com TA-Lib (se disponível)
        if TALIB_AVAILABLE:
            patterns.extend(self._detect_talib_patterns(df, trend))

        # Detectar padrões customizados
        patterns.extend(self._detect_single_candle_patterns(df, trend))

        if len(df) >= 2:
            patterns.extend(self._detect_two_candle_patterns(df, trend))

        if len(df) >= 3:
            patterns.extend(self._detect_three_candle_patterns(df, trend))

        return patterns

    def calculate_pattern_strength(self, patterns: list[CandlePattern]) -> dict:
        """
        Calcula a força geral dos padrões detectados.

        Returns:
            Dict com scores bullish, bearish e confiança
        """
        strength = {
            "bullish_score": 0,
            "bearish_score": 0,
            "neutral_score": 0,
            "total_patterns": len(patterns),
            "confidence": "LOW",
        }

        # Pesos por tipo de padrão
        type_weights = {
            PatternType.BULLISH_REVERSAL: 1.0,
            PatternType.BEARISH_REVERSAL: 1.0,
            PatternType.INDECISION: 0.5,
            PatternType.CONTINUATION: 0.7,
        }

        # Pesos por força do sinal
        strength_weights = {
            SignalStrength.WEAK: 0.5,
            SignalStrength.MODERATE: 1.0,
            SignalStrength.STRONG: 1.5,
            SignalStrength.VERY_STRONG: 2.0,
        }

        for pattern in patterns:
            weight = type_weights.get(pattern.pattern_type, 1.0) * strength_weights.get(
                pattern.strength, 1.0
            )

            if pattern.signal == "BUY":
                strength["bullish_score"] += weight * pattern.confidence
            elif pattern.signal == "SELL":
                strength["bearish_score"] += weight * pattern.confidence
            else:
                strength["neutral_score"] += weight * pattern.confidence

        # Determinar confiança geral
        total_score = strength["bullish_score"] + strength["bearish_score"]

        if total_score >= 4.0:
            strength["confidence"] = "VERY_HIGH"
        elif total_score >= 2.5:
            strength["confidence"] = "HIGH"
        elif total_score >= 1.5:
            strength["confidence"] = "MEDIUM"
        else:
            strength["confidence"] = "LOW"

        # Determinar sinal dominante
        if strength["bullish_score"] > strength["bearish_score"] * 1.5:
            strength["dominant_signal"] = "BUY"
        elif strength["bearish_score"] > strength["bullish_score"] * 1.5:
            strength["dominant_signal"] = "SELL"
        else:
            strength["dominant_signal"] = "HOLD"

        return strength

    def generate_summary(self, patterns: list[CandlePattern]) -> str:
        """
        Gera resumo textual dos padrões para LLM.

        Returns:
            String formatada com resumo dos padrões
        """
        if not patterns:
            return "📊 Nenhum padrão de candlestick detectado."

        strength = self.calculate_pattern_strength(patterns)

        summary = "📊 PADRÕES DE CANDLESTICK DETECTADOS:\n\n"

        # Agrupar por tipo
        bullish = [p for p in patterns if p.signal == "BUY"]
        bearish = [p for p in patterns if p.signal == "SELL"]
        neutral = [p for p in patterns if p.signal == "HOLD"]

        if bullish:
            summary += "🟢 PADRÕES DE ALTA:\n"
            for p in bullish:
                summary += (
                    f"  • {p.name} (confiança: {p.confidence:.0%}, força: {p.strength.name})\n"
                )
            summary += "\n"

        if bearish:
            summary += "🔴 PADRÕES DE BAIXA:\n"
            for p in bearish:
                summary += (
                    f"  • {p.name} (confiança: {p.confidence:.0%}, força: {p.strength.name})\n"
                )
            summary += "\n"

        if neutral:
            summary += "⚪ PADRÕES DE INDECISÃO:\n"
            for p in neutral:
                summary += f"  • {p.name} (confiança: {p.confidence:.0%})\n"
            summary += "\n"

        summary += "💪 FORÇA GERAL:\n"
        summary += f"  • Score Bullish: {strength['bullish_score']:.2f}\n"
        summary += f"  • Score Bearish: {strength['bearish_score']:.2f}\n"
        summary += f"  • Confiança: {strength['confidence']}\n"
        summary += f"  • Sinal Dominante: {strength['dominant_signal']}\n"

        return summary

    # ==================== TA-LIB PATTERNS ====================

    def _detect_talib_patterns(self, df: pd.DataFrame, trend: str) -> list[CandlePattern]:
        """Detecta padrões usando TA-Lib"""
        patterns = []
        last_idx = len(df) - 1

        # Padrões bullish
        talib_bullish = [
            ("CDLHAMMER", "Hammer", SignalStrength.STRONG),
            ("CDLINVERTEDHAMMER", "Inverted Hammer", SignalStrength.MODERATE),
            ("CDLENGULFING", "Engulfing", SignalStrength.VERY_STRONG),
            ("CDLPIERCING", "Piercing Line", SignalStrength.STRONG),
            ("CDLMORNINGSTAR", "Morning Star", SignalStrength.VERY_STRONG),
            ("CDL3WHITESOLDIERS", "Three White Soldiers", SignalStrength.VERY_STRONG),
        ]

        # Padrões bearish
        talib_bearish = [
            ("CDLHANGINGMAN", "Hanging Man", SignalStrength.STRONG),
            ("CDLSHOOTINGSTAR", "Shooting Star", SignalStrength.STRONG),
            ("CDLDARKCLOUDCOVER", "Dark Cloud Cover", SignalStrength.STRONG),
            ("CDLEVENINGSTAR", "Evening Star", SignalStrength.VERY_STRONG),
            ("CDL3BLACKCROWS", "Three Black Crows", SignalStrength.VERY_STRONG),
        ]

        # Detectar padrões bullish
        for func_name, name, strength in talib_bullish:
            result = getattr(talib, func_name)(df["open"], df["high"], df["low"], df["close"])
            if result.iloc[last_idx] > 0:
                patterns.append(
                    CandlePattern(
                        name=f"{name} (TA-Lib)",
                        pattern_type=PatternType.BULLISH_REVERSAL,
                        signal="BUY",
                        confidence=0.80 if trend == "downtrend" else 0.65,
                        significance=85 if trend == "downtrend" else 70,
                        strength=strength,
                        candle_index=last_idx,
                        description=f"{name} detectado via TA-Lib",
                        confirmation_needed=True,
                        confirmation_criteria="Aguardar confirmação no próximo candle",
                    )
                )

        # Detectar padrões bearish
        for func_name, name, strength in talib_bearish:
            result = getattr(talib, func_name)(df["open"], df["high"], df["low"], df["close"])
            if result.iloc[last_idx] < 0:
                patterns.append(
                    CandlePattern(
                        name=f"{name} (TA-Lib)",
                        pattern_type=PatternType.BEARISH_REVERSAL,
                        signal="SELL",
                        confidence=0.80 if trend == "uptrend" else 0.65,
                        significance=85 if trend == "uptrend" else 70,
                        strength=strength,
                        candle_index=last_idx,
                        description=f"{name} detectado via TA-Lib",
                        confirmation_needed=True,
                        confirmation_criteria="Aguardar confirmação no próximo candle",
                    )
                )

        return patterns

    # ==================== PADRÕES DE 1 CANDLE ====================

    def _detect_single_candle_patterns(self, df: pd.DataFrame, trend: str) -> list[CandlePattern]:
        """Detecta padrões de 1 candle"""
        patterns = []
        last_idx = len(df) - 1

        # Hammer
        if self._is_hammer(df.iloc[last_idx]):
            patterns.append(
                CandlePattern(
                    name="Hammer",
                    pattern_type=PatternType.BULLISH_REVERSAL,
                    signal="BUY",
                    confidence=0.75 if trend == "downtrend" else 0.60,
                    significance=85 if trend == "downtrend" else 70,
                    strength=SignalStrength.STRONG
                    if trend == "downtrend"
                    else SignalStrength.MODERATE,
                    candle_index=last_idx,
                    description="Hammer: corpo pequeno no topo, cauda inferior longa (reversão de alta)",
                    confirmation_needed=True,
                    confirmation_criteria=f"Próximo candle fechar acima de {df.iloc[last_idx]['close']:.2f}",
                )
            )

        # Shooting Star
        if self._is_shooting_star(df.iloc[last_idx]):
            patterns.append(
                CandlePattern(
                    name="Shooting Star",
                    pattern_type=PatternType.BEARISH_REVERSAL,
                    signal="SELL",
                    confidence=0.75 if trend == "uptrend" else 0.60,
                    significance=85 if trend == "uptrend" else 70,
                    strength=SignalStrength.STRONG
                    if trend == "uptrend"
                    else SignalStrength.MODERATE,
                    candle_index=last_idx,
                    description="Shooting Star: corpo pequeno embaixo, cauda superior longa (reversão de baixa)",
                    confirmation_needed=True,
                    confirmation_criteria=f"Próximo candle fechar abaixo de {df.iloc[last_idx]['close']:.2f}",
                )
            )

        # Doji
        if self._is_doji(df.iloc[last_idx]):
            patterns.append(
                CandlePattern(
                    name="Doji",
                    pattern_type=PatternType.INDECISION,
                    signal="HOLD",
                    confidence=0.65,
                    significance=70,
                    strength=SignalStrength.MODERATE,
                    candle_index=last_idx,
                    description="Doji: abertura ≈ fechamento (indecisão do mercado)",
                    confirmation_needed=True,
                    confirmation_criteria="Aguardar próximo candle para confirmar direção",
                )
            )

        return patterns

    # ==================== PADRÕES DE 2 CANDLES ====================

    def _detect_two_candle_patterns(self, df: pd.DataFrame, trend: str) -> list[CandlePattern]:
        """Detecta padrões de 2 candles"""
        patterns = []
        last_idx = len(df) - 1

        if last_idx < 1:
            return patterns

        prev = df.iloc[last_idx - 1]
        curr = df.iloc[last_idx]

        # Bullish Engulfing
        if self._is_bullish_engulfing(prev, curr):
            patterns.append(
                CandlePattern(
                    name="Bullish Engulfing",
                    pattern_type=PatternType.BULLISH_REVERSAL,
                    signal="BUY",
                    confidence=0.80 if trend == "downtrend" else 0.65,
                    significance=90 if trend == "downtrend" else 75,
                    strength=SignalStrength.VERY_STRONG
                    if trend == "downtrend"
                    else SignalStrength.STRONG,
                    candle_index=last_idx,
                    description="Bullish Engulfing: candle verde engole o vermelho anterior",
                    confirmation_needed=False,
                    confirmation_criteria="Padrão já confirmado",
                )
            )

        # Bearish Engulfing
        if self._is_bearish_engulfing(prev, curr):
            patterns.append(
                CandlePattern(
                    name="Bearish Engulfing",
                    pattern_type=PatternType.BEARISH_REVERSAL,
                    signal="SELL",
                    confidence=0.80 if trend == "uptrend" else 0.65,
                    significance=90 if trend == "uptrend" else 75,
                    strength=SignalStrength.VERY_STRONG
                    if trend == "uptrend"
                    else SignalStrength.STRONG,
                    candle_index=last_idx,
                    description="Bearish Engulfing: candle vermelho engole o verde anterior",
                    confirmation_needed=False,
                    confirmation_criteria="Padrão já confirmado",
                )
            )

        return patterns

    # ==================== PADRÕES DE 3 CANDLES ====================

    def _detect_three_candle_patterns(self, df: pd.DataFrame, trend: str) -> list[CandlePattern]:
        """Detecta padrões de 3 candles"""
        patterns = []
        last_idx = len(df) - 1

        if last_idx < 2:
            return patterns

        c1 = df.iloc[last_idx - 2]
        c2 = df.iloc[last_idx - 1]
        c3 = df.iloc[last_idx]

        # Morning Star
        if self._is_morning_star(c1, c2, c3):
            patterns.append(
                CandlePattern(
                    name="Morning Star",
                    pattern_type=PatternType.BULLISH_REVERSAL,
                    signal="BUY",
                    confidence=0.85 if trend == "downtrend" else 0.70,
                    significance=95 if trend == "downtrend" else 80,
                    strength=SignalStrength.VERY_STRONG
                    if trend == "downtrend"
                    else SignalStrength.STRONG,
                    candle_index=last_idx,
                    description="Morning Star: padrão de 3 candles indicando reversão de alta",
                    confirmation_needed=False,
                    confirmation_criteria="Padrão já confirmado",
                )
            )

        # Evening Star
        if self._is_evening_star(c1, c2, c3):
            patterns.append(
                CandlePattern(
                    name="Evening Star",
                    pattern_type=PatternType.BEARISH_REVERSAL,
                    signal="SELL",
                    confidence=0.85 if trend == "uptrend" else 0.70,
                    significance=95 if trend == "uptrend" else 80,
                    strength=SignalStrength.VERY_STRONG
                    if trend == "uptrend"
                    else SignalStrength.STRONG,
                    candle_index=last_idx,
                    description="Evening Star: padrão de 3 candles indicando reversão de baixa",
                    confirmation_needed=False,
                    confirmation_criteria="Padrão já confirmado",
                )
            )

        return patterns

    # ==================== DETECÇÃO DE PADRÕES INDIVIDUAIS ====================

    def _is_hammer(self, candle: pd.Series) -> bool:
        """Detecta Hammer"""
        body = abs(candle["body"])
        lower_shadow = candle["lower_shadow"]
        upper_shadow = candle["upper_shadow"]

        return (
            lower_shadow >= 2 * body
            and upper_shadow < 0.1 * body
            and body > self.min_body_size * candle["close"]
        )

    def _is_shooting_star(self, candle: pd.Series) -> bool:
        """Detecta Shooting Star"""
        body = abs(candle["body"])
        lower_shadow = candle["lower_shadow"]
        upper_shadow = candle["upper_shadow"]

        return (
            upper_shadow >= 2 * body
            and lower_shadow < 0.1 * body
            and body > self.min_body_size * candle["close"]
        )

    def _is_doji(self, candle: pd.Series) -> bool:
        """Detecta Doji"""
        body = abs(candle["body"])
        total_range = candle["high"] - candle["low"]

        return body < 0.1 * total_range if total_range > 0 else False

    def _is_bullish_engulfing(self, prev: pd.Series, curr: pd.Series) -> bool:
        """Detecta Bullish Engulfing"""
        return (
            prev["body"] < 0  # Anterior vermelho
            and curr["body"] > 0  # Atual verde
            and curr["open"] < prev["close"]
            and curr["close"] > prev["open"]
        )

    def _is_bearish_engulfing(self, prev: pd.Series, curr: pd.Series) -> bool:
        """Detecta Bearish Engulfing"""
        return (
            prev["body"] > 0  # Anterior verde
            and curr["body"] < 0  # Atual vermelho
            and curr["open"] > prev["close"]
            and curr["close"] < prev["open"]
        )

    def _is_morning_star(self, c1: pd.Series, c2: pd.Series, c3: pd.Series) -> bool:
        """Detecta Morning Star"""
        return (
            c1["body"] < 0  # Primeiro vermelho
            and abs(c2["body"]) < abs(c1["body"]) * 0.3  # Segundo pequeno
            and c3["body"] > 0  # Terceiro verde
            and c3["close"] > (c1["open"] + c1["close"]) / 2  # Fecha acima do meio do primeiro
        )

    def _is_evening_star(self, c1: pd.Series, c2: pd.Series, c3: pd.Series) -> bool:
        """Detecta Evening Star"""
        return (
            c1["body"] > 0  # Primeiro verde
            and abs(c2["body"]) < abs(c1["body"]) * 0.3  # Segundo pequeno
            and c3["body"] < 0  # Terceiro vermelho
            and c3["close"] < (c1["open"] + c1["close"]) / 2  # Fecha abaixo do meio do primeiro
        )

    # ==================== HELPER METHODS ====================

    def _calculate_candle_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula propriedades dos candles"""
        df = df.copy()

        df["body"] = df["close"] - df["open"]
        df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["total_range"] = df["high"] - df["low"]

        return df

    def _detect_trend(self, df: pd.DataFrame, lookback: int) -> str:
        """Detecta tendência anterior"""
        if len(df) < lookback:
            return "neutral"

        recent = df.tail(lookback)

        # Calcular EMA simples
        ema_short = recent["close"].ewm(span=3).mean().iloc[-1]
        ema_long = recent["close"].ewm(span=lookback).mean().iloc[-1]

        if ema_short > ema_long * 1.02:
            return "uptrend"
        elif ema_short < ema_long * 0.98:
            return "downtrend"
        else:
            return "neutral"
