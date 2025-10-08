"""
CeciAI - ML Agent
Agente que integra todos os modelos ML com o pipeline

Autor: CeciAI Team
Data: 2025-10-08
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from agents.ml_models.pattern_recognizer import CNNPatternRecognizer
from agents.ml_models.price_predictor import LSTMPricePredictor
from agents.ml_models.trade_classifier import XGBoostTradeClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLAgent:
    """
    Agente ML - Integra todos os modelos de Machine Learning

    Responsabilidades:
    - Carregar modelos treinados
    - Fazer previsões consolidadas
    - Fornecer confiança agregada
    """

    def __init__(self, models_dir: str = "data/models"):
        """Inicializa o agente ML"""
        self.models_dir = Path(models_dir)

        # Carregar modelos
        lstm_path = self.models_dir / "lstm_price_predictor.pth"
        cnn_path = self.models_dir / "cnn_pattern_recognizer.pth"
        xgb_path = self.models_dir / "xgboost_trade_classifier.pkl"

        self.lstm = LSTMPricePredictor(model_path=str(lstm_path) if lstm_path.exists() else None)
        self.cnn = CNNPatternRecognizer(model_path=str(cnn_path) if cnn_path.exists() else None)
        self.xgboost = XGBoostTradeClassifier(
            model_path=str(xgb_path) if xgb_path.exists() else None
        )

        logger.info(
            f"ML Agent inicializado (LSTM: {self.lstm.is_trained}, CNN: {self.cnn.is_trained}, XGBoost: {self.xgboost.is_trained})"
        )

    async def predict(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Faz previsões com todos os modelos.

        Args:
            df: DataFrame com dados OHLCV e indicadores

        Returns:
            Dict com previsões consolidadas
        """
        # LSTM: Previsão de preço
        lstm_prediction = self.lstm.predict(df)

        # CNN: Reconhecimento de padrões
        cnn_prediction = self.cnn.predict(df)

        # XGBoost: Classificação de trade
        xgb_prediction = self.xgboost.predict(df)

        # Consolidar sinais
        signals = [
            lstm_prediction.get("direction", "NEUTRAL"),
            cnn_prediction.get("signal", "HOLD"),
            xgb_prediction.get("signal", "HOLD"),
        ]

        # Voto majoritário
        buy_votes = signals.count("BUY") + signals.count("UP")
        sell_votes = signals.count("SELL") + signals.count("DOWN")

        if buy_votes > sell_votes:
            consolidated_signal = "BUY"
        elif sell_votes > buy_votes:
            consolidated_signal = "SELL"
        else:
            consolidated_signal = "HOLD"

        # Confiança média
        confidences = [
            lstm_prediction.get("confidence", 0.5),
            cnn_prediction.get("confidence", 0.5),
            xgb_prediction.get("confidence", 0.5),
        ]
        avg_confidence = sum(confidences) / len(confidences)

        return {
            "consolidated_signal": consolidated_signal,
            "confidence": avg_confidence,
            "lstm_prediction": lstm_prediction,
            "cnn_prediction": cnn_prediction,
            "xgboost_prediction": xgb_prediction,
            "models_trained": {
                "lstm": self.lstm.is_trained,
                "cnn": self.cnn.is_trained,
                "xgboost": self.xgboost.is_trained,
            },
        }
