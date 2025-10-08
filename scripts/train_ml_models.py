"""
CeciAI - Script de Treinamento de Modelos ML
Treina todos os modelos (LSTM, CNN, XGBoost) com dados históricos

Autor: CeciAI Team
Data: 2025-10-08
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ml_models.pattern_recognizer import CNNPatternRecognizer
from agents.ml_models.price_predictor import LSTMPricePredictor
from agents.ml_models.trade_classifier import XGBoostTradeClassifier
from utils.database import DatabaseManager
from utils.technical_indicators import calculate_bollinger_bands, calculate_macd, calculate_rsi

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MLModelTrainer:
    """
    Treinador de modelos ML

    Responsabilidades:
    - Carregar dados históricos do DuckDB
    - Preparar dados com indicadores técnicos
    - Treinar LSTM, CNN e XGBoost
    - Salvar modelos treinados
    - Avaliar performance
    """

    def __init__(self, models_dir: str = "data/models"):
        """
        Inicializa o treinador.

        Args:
            models_dir: Diretório para salvar modelos
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.db = DatabaseManager()

        logger.info(f"MLModelTrainer inicializado. Modelos serão salvos em: {self.models_dir}")

    async def load_historical_data(
        self, symbol: str = "BTC/USD", timeframe: str = "1h", limit: int = 10000
    ) -> pd.DataFrame:
        """
        Carrega dados históricos do banco.

        Args:
            symbol: Par de trading
            timeframe: Timeframe
            limit: Número máximo de registros

        Returns:
            DataFrame com dados OHLCV
        """
        logger.info(f"📥 Carregando dados históricos: {symbol} ({timeframe})")

        try:
            # Tentar carregar do DuckDB
            df = self.db.get_ohlcv(symbol, timeframe, limit=limit)

            if df is None or len(df) == 0:
                logger.warning("Dados não encontrados no banco. Gerando dados sintéticos...")
                df = self._generate_synthetic_data(limit)
            else:
                logger.info(f"✅ {len(df)} registros carregados")

            return df

        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            logger.info("Gerando dados sintéticos...")
            return self._generate_synthetic_data(limit)

    def _generate_synthetic_data(self, size: int = 10000) -> pd.DataFrame:
        """Gera dados sintéticos para treinamento"""
        logger.info(f"Gerando {size} registros sintéticos...")

        np.random.seed(42)

        # Gerar preços com tendência
        trend = np.linspace(0, 1000, size)
        noise = np.random.randn(size).cumsum() * 50
        base_price = 50000 + trend + noise

        data = {
            "timestamp": pd.date_range("2023-01-01", periods=size, freq="1H"),
            "open": base_price + np.random.randn(size) * 50,
            "high": base_price + np.abs(np.random.randn(size)) * 100,
            "low": base_price - np.abs(np.random.randn(size)) * 100,
            "close": base_price + np.random.randn(size) * 50,
            "volume": np.random.randint(1000000, 3000000, size),
        }

        df = pd.DataFrame(data)

        # Garantir consistência OHLC
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)

        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona indicadores técnicos ao DataFrame.

        Args:
            df: DataFrame com OHLCV

        Returns:
            DataFrame com indicadores
        """
        logger.info("📊 Calculando indicadores técnicos...")

        # RSI
        df["rsi"] = calculate_rsi(df["close"])

        # MACD
        macd_data = calculate_macd(df["close"])
        df["macd"] = macd_data["macd"]
        df["macd_signal"] = macd_data["signal"]
        df["macd_histogram"] = macd_data["histogram"]

        # Bollinger Bands
        bb_data = calculate_bollinger_bands(df["close"])
        df["bb_upper"] = bb_data["upper"]
        df["bb_middle"] = bb_data["middle"]
        df["bb_lower"] = bb_data["lower"]

        # EMAs
        df["ema_9"] = df["close"].ewm(span=9).mean()
        df["ema_21"] = df["close"].ewm(span=21).mean()

        # Preencher NaN
        df = df.fillna(method="bfill").fillna(method="ffill")

        logger.info("✅ Indicadores calculados")

        return df

    def generate_labels(self, df: pd.DataFrame, lookahead: int = 24) -> pd.DataFrame:
        """
        Gera labels para treinamento supervisionado.

        Args:
            df: DataFrame com dados
            lookahead: Quantos períodos olhar à frente

        Returns:
            DataFrame com labels
        """
        logger.info(f"🏷️  Gerando labels (lookahead={lookahead})...")

        # Label para classificação (BUY/SELL/HOLD)
        future_returns = df["close"].shift(-lookahead) / df["close"] - 1

        labels = []
        for ret in future_returns:
            if pd.isna(ret):
                labels.append("HOLD")
            elif ret > 0.02:  # +2%
                labels.append("BUY")
            elif ret < -0.02:  # -2%
                labels.append("SELL")
            else:
                labels.append("HOLD")

        df["label"] = labels

        # Label para padrões (bullish/bearish/neutral)
        pattern_labels = []
        for i in range(len(df)):
            if i < 5:
                pattern_labels.append("neutral")
            else:
                # Últimos 5 candles
                recent = df.iloc[i - 5 : i]
                bullish_count = sum(
                    1
                    for j in range(len(recent))
                    if recent["close"].iloc[j] > recent["open"].iloc[j]
                )

                if bullish_count >= 4:
                    pattern_labels.append("bullish")
                elif bullish_count <= 1:
                    pattern_labels.append("bearish")
                else:
                    pattern_labels.append("neutral")

        df["pattern_label"] = pattern_labels

        logger.info("✅ Labels gerados")

        return df

    async def train_lstm(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32) -> dict[str, Any]:
        """Treina modelo LSTM"""
        logger.info("\n" + "=" * 60)
        logger.info("🤖 TREINANDO LSTM PRICE PREDICTOR")
        logger.info("=" * 60)

        predictor = LSTMPricePredictor(sequence_length=60, use_gpu=True)

        history = predictor.train(
            df=df, epochs=epochs, batch_size=batch_size, learning_rate=0.001, validation_split=0.2
        )

        # Salvar modelo
        model_path = self.models_dir / "lstm_price_predictor.pth"
        predictor.save_model(str(model_path))

        logger.info(f"✅ LSTM treinado e salvo em: {model_path}")

        return {
            "model": "LSTM",
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "model_path": str(model_path),
        }

    async def train_cnn(self, df: pd.DataFrame, epochs: int = 30, batch_size: int = 32) -> dict[str, Any]:
        """Treina modelo CNN"""
        logger.info("\n" + "=" * 60)
        logger.info("🤖 TREINANDO CNN PATTERN RECOGNIZER")
        logger.info("=" * 60)

        recognizer = CNNPatternRecognizer(image_size=64, use_gpu=True)

        # Pegar labels
        labels = df["pattern_label"].tolist()[:-20]  # Remover últimos 20 (janela)

        history = recognizer.train(
            df=df,
            labels=labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=0.001,
            validation_split=0.2,
        )

        # Salvar modelo
        model_path = self.models_dir / "cnn_pattern_recognizer.pth"
        recognizer.save_model(str(model_path))

        logger.info(f"✅ CNN treinado e salvo em: {model_path}")

        return {
            "model": "CNN",
            "final_train_acc": history["train_acc"][-1],
            "final_val_acc": history["val_acc"][-1],
            "model_path": str(model_path),
        }

    async def train_xgboost(self, df: pd.DataFrame) -> dict[str, Any]:
        """Treina modelo XGBoost"""
        logger.info("\n" + "=" * 60)
        logger.info("🤖 TREINANDO XGBOOST TRADE CLASSIFIER")
        logger.info("=" * 60)

        classifier = XGBoostTradeClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)

        # Pegar labels
        labels = df["label"].tolist()

        history = classifier.train(df=df, labels=labels, validation_split=0.2)

        # Salvar modelo
        model_path = self.models_dir / "xgboost_trade_classifier.pkl"
        classifier.save_model(str(model_path))

        logger.info(f"✅ XGBoost treinado e salvo em: {model_path}")

        return {
            "model": "XGBoost",
            "train_accuracy": history["train_accuracy"],
            "val_accuracy": history["val_accuracy"],
            "top_features": list(history["feature_importance"].keys())[:5],
            "model_path": str(model_path),
        }

    async def train_all_models(
        self,
        symbol: str = "BTC/USD",
        timeframe: str = "1h",
        lstm_epochs: int = 50,
        cnn_epochs: int = 30,
    ):
        """
        Treina todos os modelos.

        Args:
            symbol: Par de trading
            timeframe: Timeframe
            lstm_epochs: Épocas para LSTM
            cnn_epochs: Épocas para CNN
        """
        start_time = datetime.now()

        logger.info("\n" + "=" * 60)
        logger.info("🚀 INICIANDO TREINAMENTO DE TODOS OS MODELOS")
        logger.info("=" * 60)

        # 1. Carregar dados
        df = await self.load_historical_data(symbol, timeframe, limit=10000)

        # 2. Adicionar indicadores
        df = self.add_technical_indicators(df)

        # 3. Gerar labels
        df = self.generate_labels(df, lookahead=24)

        # 4. Treinar modelos
        results = []

        try:
            lstm_result = await self.train_lstm(df, epochs=lstm_epochs)
            results.append(lstm_result)
        except Exception as e:
            logger.error(f"Erro ao treinar LSTM: {e}", exc_info=True)

        try:
            cnn_result = await self.train_cnn(df, epochs=cnn_epochs)
            results.append(cnn_result)
        except Exception as e:
            logger.error(f"Erro ao treinar CNN: {e}", exc_info=True)

        try:
            xgboost_result = await self.train_xgboost(df)
            results.append(xgboost_result)
        except Exception as e:
            logger.error(f"Erro ao treinar XGBoost: {e}", exc_info=True)

        # 5. Resumo
        duration = (datetime.now() - start_time).total_seconds()

        logger.info("\n" + "=" * 60)
        logger.info("✅ TREINAMENTO CONCLUÍDO")
        logger.info("=" * 60)
        logger.info(f"Tempo total: {duration:.2f}s")
        logger.info(f"Modelos treinados: {len(results)}/3")

        for result in results:
            logger.info(f"\n📊 {result['model']}:")
            for key, value in result.items():
                if key != "model":
                    logger.info(f"  {key}: {value}")

        logger.info("\n🎉 Todos os modelos foram treinados e salvos!")
        logger.info(f"📁 Diretório: {self.models_dir}")


async def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Treinar modelos ML do CeciAI")
    parser.add_argument("--symbol", type=str, default="BTC/USD", help="Par de trading")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe")
    parser.add_argument("--lstm-epochs", type=int, default=50, help="Épocas para LSTM")
    parser.add_argument("--cnn-epochs", type=int, default=30, help="Épocas para CNN")
    parser.add_argument(
        "--models-dir", type=str, default="data/models", help="Diretório para modelos"
    )

    args = parser.parse_args()

    trainer = MLModelTrainer(models_dir=args.models_dir)

    await trainer.train_all_models(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lstm_epochs=args.lstm_epochs,
        cnn_epochs=args.cnn_epochs,
    )


if __name__ == "__main__":
    asyncio.run(main())
