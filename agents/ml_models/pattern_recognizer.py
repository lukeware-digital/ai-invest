"""
CeciAI - CNN Pattern Recognizer
Modelo CNN para reconhecimento de padrões em candlesticks

Responsabilidades:
- Reconhecer padrões visuais em candles
- Classificar padrões (bullish/bearish/neutral)
- Usar imagens de candles como input
- Fornecer confiança do reconhecimento

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
import torch.nn.functional as f

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNNModel(nn.Module):
    """
    Modelo CNN para reconhecimento de padrões em candlesticks

    Arquitetura:
    - Input: imagem 64x64 (representação de candles)
    - Conv2D: 32 filtros, kernel 3x3
    - MaxPool: 2x2
    - Conv2D: 64 filtros, kernel 3x3
    - MaxPool: 2x2
    - Conv2D: 128 filtros, kernel 3x3
    - MaxPool: 2x2
    - Flatten
    - Dense: 128 unidades
    - Dropout: 0.5
    - Output: 3 classes (bullish, bearish, neutral)
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Dense layers
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = f.relu(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = f.relu(x)
        x = self.pool2(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = f.relu(x)
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense layers
        x = self.fc1(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class CNNPatternRecognizer:
    """
    Reconhecedor de padrões usando CNN

    Features:
    - Converte candles em imagens
    - Reconhece padrões visuais
    - Classifica como bullish/bearish/neutral
    - Usa GPU se disponível
    """

    def __init__(
        self, model_path: str | None = None, image_size: int = 64, use_gpu: bool = True
    ):
        """
        Inicializa o reconhecedor.

        Args:
            model_path: Caminho para modelo salvo
            image_size: Tamanho da imagem (64x64)
            use_gpu: Usar GPU se disponível
        """
        self.image_size = image_size
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        self.classes = ["bullish", "bearish", "neutral"]
        self.num_classes = len(self.classes)

        self.model = None
        self.is_trained = False

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

        logger.info(f"CNN Pattern Recognizer inicializado (device: {self.device})")

    def candles_to_image(self, df: pd.DataFrame, window_size: int = 20) -> np.ndarray:
        """
        Converte candles em imagem 64x64.

        Args:
            df: DataFrame com OHLCV
            window_size: Número de candles para visualizar

        Returns:
            Array numpy (64, 64) representando os candles
        """
        # Pegar últimos N candles
        candles = df.tail(window_size)

        # Criar imagem vazia
        image = np.zeros((self.image_size, self.image_size))

        # Normalizar preços para caber na imagem
        high_max = candles["high"].max()
        low_min = candles["low"].min()
        price_range = high_max - low_min

        if price_range == 0:
            return image

        # Largura de cada candle
        candle_width = self.image_size // window_size

        for i, (_, row) in enumerate(candles.iterrows()):
            # Posição X
            x_start = i * candle_width
            x_end = min(x_start + candle_width - 1, self.image_size - 1)

            # Normalizar preços para Y (invertido, pois imagem começa do topo)
            open_y = int((1 - (row["open"] - low_min) / price_range) * (self.image_size - 1))
            close_y = int((1 - (row["close"] - low_min) / price_range) * (self.image_size - 1))
            high_y = int((1 - (row["high"] - low_min) / price_range) * (self.image_size - 1))
            low_y = int((1 - (row["low"] - low_min) / price_range) * (self.image_size - 1))

            # Desenhar wick (high-low)
            wick_x = (x_start + x_end) // 2
            for y in range(high_y, low_y + 1):
                if 0 <= y < self.image_size:
                    image[y, wick_x] = 0.5

            # Desenhar corpo (open-close)
            body_top = min(open_y, close_y)
            body_bottom = max(open_y, close_y)

            # Cor: branco (1.0) se bullish, cinza escuro (0.3) se bearish
            color = 1.0 if row["close"] > row["open"] else 0.3

            for y in range(body_top, body_bottom + 1):
                for x in range(x_start, x_end + 1):
                    if 0 <= y < self.image_size and 0 <= x < self.image_size:
                        image[y, x] = color

        return image

    def prepare_data(
        self, df: pd.DataFrame, labels: list[str] | None = None, window_size: int = 20
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Prepara dados para treinamento/previsão.

        Args:
            df: DataFrame com OHLCV
            labels: Lista de labels (bullish/bearish/neutral)
            window_size: Tamanho da janela de candles

        Returns:
            X: Array de imagens (N, 1, 64, 64)
            y: Array de labels (N,) ou None
        """
        images = []

        # Criar imagens deslizantes
        for i in range(len(df) - window_size + 1):
            window_df = df.iloc[i : i + window_size]
            image = self.candles_to_image(window_df, window_size)
            images.append(image)

        X = np.array(images)
        X = X.reshape(-1, 1, self.image_size, self.image_size)

        if labels is not None:
            # Converter labels para índices
            label_to_idx = {label: index for index, label in enumerate(self.classes)}
            y = np.array([label_to_idx[label] for label in labels])
            return X, y

        return X, None

    def train(
        self,
        df: pd.DataFrame,
        labels: list[str],
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> dict[str, Any]:
        """
        Treina o modelo CNN.

        Args:
            df: DataFrame com dados históricos
            labels: Labels para cada janela
            epochs: Número de épocas
            batch_size: Tamanho do batch
            learning_rate: Taxa de aprendizado
            validation_split: Proporção para validação

        Returns:
            Dict com histórico de treinamento
        """
        logger.info("🚀 Iniciando treinamento do modelo CNN...")

        # Preparar dados
        X, y = self.prepare_data(df, labels)

        # Split train/validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        logger.info(f"Dados de treino: {X_train.shape}, Validação: {X_val.shape}")

        # Converter para tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)

        # Criar modelo
        self.model = CNNModel(num_classes=self.num_classes).to(self.device)

        # Loss e optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            self.model.train()

            total_loss = 0
            correct = 0
            total = 0

            # Mini-batches
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

                # Métricas
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            train_loss = total_loss / (len(X_train_tensor) // batch_size)
            train_acc = correct / total

            # Validação
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

                _, val_predicted = torch.max(val_outputs.data, 1)
                val_acc = (val_predicted == y_val_tensor).sum().item() / len(y_val_tensor)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}"
                )

        self.is_trained = True
        logger.info("✅ Treinamento concluído!")

        return history

    def predict(self, df: pd.DataFrame, window_size: int = 20) -> dict[str, Any]:
        """
        Reconhece padrões nos candles.

        Args:
            df: DataFrame com dados recentes
            window_size: Tamanho da janela

        Returns:
            Dict com classificação e confiança
        """
        if not self.is_trained or self.model is None:
            logger.warning("Modelo não treinado. Retornando classificação padrão.")
            return self._get_default_prediction(df)

        try:
            # Preparar dados
            X, _ = self.prepare_data(df, window_size=window_size)

            if len(X) == 0:
                return self._get_default_prediction(df)

            # Pegar última janela
            X_last = X[-1:]
            X_tensor = torch.FloatTensor(X_last).to(self.device)

            # Predição
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probabilities = f.softmax(outputs, dim=1).cpu().numpy()[0]
                predicted_class = np.argmax(probabilities)

            pattern_type = self.classes[predicted_class]
            confidence = float(probabilities[predicted_class])

            return {
                "pattern_type": pattern_type,
                "confidence": confidence,
                "probabilities": {
                    "bullish": float(probabilities[0]),
                    "bearish": float(probabilities[1]),
                    "neutral": float(probabilities[2]),
                },
                "signal": "BUY"
                if pattern_type == "bullish"
                else "SELL"
                if pattern_type == "bearish"
                else "HOLD",
                "model": "CNN",
                "is_trained": True,
            }

        except Exception as e:
            logger.error(f"Erro no reconhecimento: {e}", exc_info=True)
            return self._get_default_prediction(df)

    def _get_default_prediction(self, df: pd.DataFrame) -> dict[str, Any]:
        """Retorna classificação padrão quando modelo não disponível"""
        # Usar tendência simples como fallback
        if len(df) >= 5:
            recent_bullish = sum(
                1 for i in range(-5, 0) if df["close"].iloc[i] > df["open"].iloc[i]
            )

            if recent_bullish >= 4:
                pattern_type = "bullish"
            elif recent_bullish <= 1:
                pattern_type = "bearish"
            else:
                pattern_type = "neutral"
        else:
            pattern_type = "neutral"

        return {
            "pattern_type": pattern_type,
            "confidence": 0.50,
            "probabilities": {"bullish": 0.33, "bearish": 0.33, "neutral": 0.34},
            "signal": "BUY"
            if pattern_type == "bullish"
            else "SELL"
            if pattern_type == "bearish"
            else "HOLD",
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
            "classes": self.classes,
            "image_size": self.image_size,
            "is_trained": self.is_trained,
        }

        torch.save(save_dict, path)
        logger.info(f"Modelo salvo em: {path}")

    def load_model(self, path: str):
        """Carrega modelo treinado"""
        try:
            checkpoint = torch.load(path, map_location=self.device)

            self.classes = checkpoint["classes"]
            self.num_classes = len(self.classes)
            self.image_size = checkpoint["image_size"]
            self.is_trained = checkpoint["is_trained"]

            # Recriar modelo
            self.model = CNNModel(num_classes=self.num_classes).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()

            logger.info(f"Modelo carregado de: {path}")

        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            self.is_trained = False


# Exemplo de uso
if __name__ == "__main__":
    # Gerar dados de teste
    np.random.seed(42)

    data = {
        "open": np.random.randn(200).cumsum() + 50000,
        "high": np.random.randn(200).cumsum() + 50200,
        "low": np.random.randn(200).cumsum() + 49800,
        "close": np.random.randn(200).cumsum() + 50100,
        "volume": np.random.randint(1000000, 2000000, 200),
    }
    df = pd.DataFrame(data)

    # Gerar labels aleatórios para teste
    labels = np.random.choice(["bullish", "bearish", "neutral"], size=181)

    # Criar e treinar modelo
    recognizer = CNNPatternRecognizer(use_gpu=False)

    print("\n🚀 Treinando modelo CNN...")
    history = recognizer.train(df, labels.tolist(), epochs=10, batch_size=16)

    print("\n✅ Treinamento concluído!")
    print(f"Acurácia final (treino): {history['train_acc'][-1]:.2%}")
    print(f"Acurácia final (validação): {history['val_acc'][-1]:.2%}")

    # Fazer previsão
    print("\n📊 Reconhecendo padrões...")
    prediction = recognizer.predict(df)

    print(f"\nPadrão Detectado: {prediction['pattern_type']}")
    print(f"Confiança: {prediction['confidence']:.0%}")
    print(f"Sinal: {prediction['signal']}")
    print("\nProbabilidades:")
    for pattern, prob in prediction["probabilities"].items():
        print(f"  {pattern}: {prob:.0%}")
