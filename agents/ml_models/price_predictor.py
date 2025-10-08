"""
CeciAI - LSTM Price Predictor
Modelo LSTM para previsão de preços de criptomoedas

Responsabilidades:
- Prever preços futuros (1h, 4h, 24h)
- Usar dados históricos OHLCV + indicadores técnicos
- Treinar com GPU (RTX 3060 Ti)
- Fornecer confiança da previsão

Autor: CeciAI Team
Data: 2025-10-08
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """
    Modelo LSTM para previsão de preços

    Arquitetura:
    - Input: sequência de 60 timesteps com features (OHLCV + indicadores)
    - LSTM Layer 1: 128 unidades
    - Dropout: 0.2
    - LSTM Layer 2: 64 unidades
    - Dropout: 0.2
    - Dense: 32 unidades
    - Output: 3 valores (preço 1h, 4h, 24h)
    """

    def __init__(
        self, input_size: int, hidden_size: int = 128, num_layers: int = 2, output_size: int = 3
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.dropout1 = nn.Dropout(0.2)

        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.dropout2 = nn.Dropout(0.2)

        # Dense layers
        self.fc1 = nn.Linear(hidden_size // 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        # LSTM 1
        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        # LSTM 2
        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        # Pegar último timestep
        out = out[:, -1, :]

        # Dense layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


class LSTMPricePredictor:
    """
    Preditor de preços usando LSTM

    Features:
    - Treina com dados históricos
    - Prevê preços futuros (1h, 4h, 24h)
    - Usa GPU se disponível
    - Salva/carrega modelos treinados
    """

    def __init__(
        self, model_path: str | None = None, sequence_length: int = 60, use_gpu: bool = True
    ):
        """
        Inicializa o preditor.

        Args:
            model_path: Caminho para modelo salvo
            sequence_length: Tamanho da sequência de entrada
            use_gpu: Usar GPU se disponível
        """
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        # Features que serão usadas
        self.feature_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi",
            "macd",
            "macd_signal",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "ema_9",
            "ema_21",
        ]

        self.model = None
        self.scaler_params = None
        self.is_trained = False

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

        logger.info(f"LSTM Price Predictor inicializado (device: {self.device})")

    def prepare_data(
        self, df: pd.DataFrame, fit_scaler: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepara dados para treinamento/previsão.

        Args:
            df: DataFrame com dados OHLCV e indicadores
            fit_scaler: Se True, calcula parâmetros de normalização

        Returns:
            X, y: Arrays normalizados
        """
        # Selecionar features
        available_features = [col for col in self.feature_columns if col in df.columns]
        data = df[available_features].values

        # Normalizar dados
        if fit_scaler or self.scaler_params is None:
            self.scaler_params = {"mean": data.mean(axis=0), "std": data.std(axis=0) + 1e-8}

        data_normalized = (data - self.scaler_params["mean"]) / self.scaler_params["std"]

        # Criar sequências
        X, y = [], []

        for i in range(len(data_normalized) - self.sequence_length - 24):
            # Input: 60 timesteps
            X.append(data_normalized[i : i + self.sequence_length])

            # Output: preço close em 1h, 4h, 24h (índices 3 = close)
            close_idx = available_features.index("close")
            y.append(
                [
                    data[i + self.sequence_length + 1, close_idx],  # +1h
                    data[i + self.sequence_length + 4, close_idx],  # +4h
                    data[i + self.sequence_length + 24, close_idx],  # +24h
                ]
            )

        return np.array(X), np.array(y)

    def train(
        self,
        df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> dict[str, Any]:
        """
        Treina o modelo LSTM.

        Args:
            df: DataFrame com dados históricos
            epochs: Número de épocas
            batch_size: Tamanho do batch
            learning_rate: Taxa de aprendizado
            validation_split: Proporção para validação

        Returns:
            Dict com histórico de treinamento
        """
        logger.info("🚀 Iniciando treinamento do modelo LSTM...")

        # Preparar dados
        X, y = self.prepare_data(df, fit_scaler=True)

        # Split train/validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        logger.info(f"Dados de treino: {X_train.shape}, Validação: {X_val.shape}")

        # Converter para tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # Criar modelo
        input_size = X_train.shape[2]
        self.model = LSTMModel(input_size=input_size).to(self.device)

        # Loss e optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.model.train()

            # Mini-batches
            total_loss = 0
            num_batches = 0

            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i : i + batch_size]
                batch_y = y_train_tensor[i : i + batch_size]

                # Forward
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_train_loss = total_loss / num_batches

            # Validação
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

        self.is_trained = True
        logger.info("✅ Treinamento concluído!")

        return history

    def predict(self, df: pd.DataFrame, return_confidence: bool = True) -> dict[str, Any]:
        """
        Faz previsão de preços futuros.

        Args:
            df: DataFrame com dados recentes (mínimo sequence_length)
            return_confidence: Se True, calcula confiança

        Returns:
            Dict com previsões e confiança
        """
        if not self.is_trained or self.model is None:
            logger.warning("Modelo não treinado. Retornando previsão padrão.")
            return self._get_default_prediction(df)

        try:
            # Preparar dados
            X, _ = self.prepare_data(df, fit_scaler=False)

            if len(X) == 0:
                return self._get_default_prediction(df)

            # Pegar última sequência
            X_last = X[-1:]
            X_tensor = torch.FloatTensor(X_last).to(self.device)

            # Predição
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(X_tensor).cpu().numpy()[0]

            current_price = float(df["close"].iloc[-1])

            # Calcular direção e confiança
            price_1h, price_4h, price_24h = prediction

            direction = (
                "UP"
                if price_24h > current_price
                else "DOWN"
                if price_24h < current_price
                else "NEUTRAL"
            )

            # Confiança baseada na consistência das previsões
            if return_confidence:
                changes = [
                    (price_1h - current_price) / current_price,
                    (price_4h - current_price) / current_price,
                    (price_24h - current_price) / current_price,
                ]

                # Se todas mudanças têm mesmo sinal = alta confiança
                all_positive = all(c > 0 for c in changes)
                all_negative = all(c < 0 for c in changes)

                if all_positive or all_negative:
                    confidence = min(0.85, 0.65 + abs(changes[-1]) * 10)
                else:
                    confidence = 0.50
            else:
                confidence = 0.70

            return {
                "current_price": current_price,
                "price_1h": float(price_1h),
                "price_4h": float(price_4h),
                "price_24h": float(price_24h),
                "direction": direction,
                "confidence": confidence,
                "change_1h_pct": ((price_1h - current_price) / current_price) * 100,
                "change_4h_pct": ((price_4h - current_price) / current_price) * 100,
                "change_24h_pct": ((price_24h - current_price) / current_price) * 100,
                "model": "LSTM",
                "is_trained": True,
            }

        except Exception as e:
            logger.error(f"Erro na previsão: {e}", exc_info=True)
            return self._get_default_prediction(df)

    def _get_default_prediction(self, df: pd.DataFrame) -> dict[str, Any]:
        """Retorna previsão padrão quando modelo não disponível"""
        current_price = float(df["close"].iloc[-1])

        # Usar tendência simples como fallback
        if len(df) >= 10:
            recent_change = (df["close"].iloc[-1] - df["close"].iloc[-10]) / df["close"].iloc[-10]
            direction = "UP" if recent_change > 0 else "DOWN"
        else:
            direction = "NEUTRAL"
            recent_change = 0

        return {
            "current_price": current_price,
            "price_1h": current_price * (1 + recent_change * 0.1),
            "price_4h": current_price * (1 + recent_change * 0.3),
            "price_24h": current_price * (1 + recent_change),
            "direction": direction,
            "confidence": 0.50,
            "change_1h_pct": recent_change * 10,
            "change_4h_pct": recent_change * 30,
            "change_24h_pct": recent_change * 100,
            "model": "FALLBACK",
            "is_trained": False,
        }

    def save_model(self, path: str):
        """Salva modelo treinado"""
        if self.model is None:
            logger.warning("Nenhum modelo para salvar")
            return

        save_dict = {
            "model_state": self.model.state_dict(),
            "scaler_params": self.scaler_params,
            "feature_columns": self.feature_columns,
            "sequence_length": self.sequence_length,
            "is_trained": self.is_trained,
        }

        torch.save(save_dict, path)
        logger.info(f"Modelo salvo em: {path}")

    def load_model(self, path: str):
        """Carrega modelo treinado"""
        try:
            checkpoint = torch.load(path, map_location=self.device)

            self.scaler_params = checkpoint["scaler_params"]
            self.feature_columns = checkpoint["feature_columns"]
            self.sequence_length = checkpoint["sequence_length"]
            self.is_trained = checkpoint["is_trained"]

            # Recriar modelo
            input_size = len(self.feature_columns)
            self.model = LSTMModel(input_size=input_size).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()

            logger.info(f"Modelo carregado de: {path}")

        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            self.is_trained = False


# Exemplo de uso
if __name__ == "__main__":
    # Gerar dados de teste
    dates = pd.date_range("2024-01-01", periods=1000, freq="1H")
    data = {
        "timestamp": dates,
        "open": np.random.randn(1000).cumsum() + 50000,
        "high": np.random.randn(1000).cumsum() + 50200,
        "low": np.random.randn(1000).cumsum() + 49800,
        "close": np.random.randn(1000).cumsum() + 50100,
        "volume": np.random.randint(1000000, 2000000, 1000),
        "rsi": np.random.uniform(30, 70, 1000),
        "macd": np.random.randn(1000),
        "macd_signal": np.random.randn(1000),
        "bb_upper": np.random.randn(1000).cumsum() + 51000,
        "bb_middle": np.random.randn(1000).cumsum() + 50000,
        "bb_lower": np.random.randn(1000).cumsum() + 49000,
        "ema_9": np.random.randn(1000).cumsum() + 50000,
        "ema_21": np.random.randn(1000).cumsum() + 50000,
    }
    df = pd.DataFrame(data)

    # Criar e treinar modelo
    predictor = LSTMPricePredictor(use_gpu=False)

    print("\n🚀 Treinando modelo LSTM...")
    history = predictor.train(df, epochs=20, batch_size=32)

    print("\n✅ Treinamento concluído!")
    print(f"Loss final (treino): {history['train_loss'][-1]:.6f}")
    print(f"Loss final (validação): {history['val_loss'][-1]:.6f}")

    # Fazer previsão
    print("\n📊 Fazendo previsão...")
    prediction = predictor.predict(df)

    print(f"\nPreço Atual: ${prediction['current_price']:,.2f}")
    print(f"Previsão 1h: ${prediction['price_1h']:,.2f} ({prediction['change_1h_pct']:+.2f}%)")
    print(f"Previsão 4h: ${prediction['price_4h']:,.2f} ({prediction['change_4h_pct']:+.2f}%)")
    print(f"Previsão 24h: ${prediction['price_24h']:,.2f} ({prediction['change_24h_pct']:+.2f}%)")
    print(f"Direção: {prediction['direction']}")
    print(f"Confiança: {prediction['confidence']:.0%}")
