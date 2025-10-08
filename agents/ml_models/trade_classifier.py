"""
CeciAI - XGBoost Trade Classifier
Modelo XGBoost para classificação de trades (BUY/SELL/HOLD)

Responsabilidades:
- Classificar oportunidades de trade
- Usar features técnicas + fundamentais
- Rápido e eficiente (CPU-friendly)
- Fornecer importância das features

Autor: CeciAI Team
Data: 2025-10-08
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostTradeClassifier:
    """
    Classificador de trades usando XGBoost

    Features:
    - Classifica como BUY/SELL/HOLD
    - Usa múltiplas features técnicas
    - Rápido e eficiente
    - Explica decisões (feature importance)
    """

    def __init__(
        self,
        model_path: str | None = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
    ):
        """
        Inicializa o classificador.

        Args:
            model_path: Caminho para modelo salvo
            n_estimators: Número de árvores
            max_depth: Profundidade máxima
            learning_rate: Taxa de aprendizado
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        self.classes = ["HOLD", "BUY", "SELL"]  # 0, 1, 2
        self.num_classes = len(self.classes)

        self.feature_columns = [
            # Preços
            "close",
            "volume",
            # Indicadores técnicos
            "rsi",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bb_position",
            "bb_width",
            "ema_9",
            "ema_21",
            "ema_crossover",
            # Tendência
            "price_change_1h",
            "price_change_4h",
            "price_change_24h",
            "volatility",
            # Volume
            "volume_ratio",
            "volume_trend",
            # Padrões (se disponível)
            "bullish_patterns",
            "bearish_patterns",
        ]

        self.model = None
        self.scaler_params = None
        self.is_trained = False

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

        logger.info("XGBoost Trade Classifier inicializado")

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrai features do DataFrame.

        Args:
            df: DataFrame com dados OHLCV e indicadores

        Returns:
            DataFrame com features
        """
        features = pd.DataFrame()

        # Preços
        features["close"] = df["close"]
        features["volume"] = df["volume"]

        # Indicadores (se disponíveis)
        if "rsi" in df.columns:
            features["rsi"] = df["rsi"]
        else:
            features["rsi"] = 50.0

        if "macd" in df.columns:
            features["macd"] = df["macd"]
            features["macd_signal"] = df.get("macd_signal", 0)
            features["macd_histogram"] = features["macd"] - features["macd_signal"]
        else:
            features["macd"] = 0.0
            features["macd_signal"] = 0.0
            features["macd_histogram"] = 0.0

        # Bollinger Bands
        if "bb_upper" in df.columns:
            bb_upper = df["bb_upper"]
            bb_lower = df["bb_lower"]
            bb_middle = df["bb_middle"]

            # Posição relativa (0 = lower, 0.5 = middle, 1 = upper)
            features["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-8)
            features["bb_width"] = ((bb_upper - bb_lower) / bb_middle) * 100
        else:
            features["bb_position"] = 0.5
            features["bb_width"] = 2.0

        # EMAs
        if "ema_9" in df.columns:
            features["ema_9"] = df["ema_9"]
            features["ema_21"] = df.get("ema_21", df["close"])
            features["ema_crossover"] = (features["ema_9"] > features["ema_21"]).astype(int)
        else:
            features["ema_9"] = df["close"]
            features["ema_21"] = df["close"]
            features["ema_crossover"] = 0

        # Mudanças de preço
        features["price_change_1h"] = df["close"].pct_change(1) * 100
        features["price_change_4h"] = df["close"].pct_change(4) * 100
        features["price_change_24h"] = df["close"].pct_change(24) * 100

        # Volatilidade
        features["volatility"] = df["close"].pct_change().rolling(20).std() * 100

        # Volume
        avg_volume = df["volume"].rolling(20).mean()
        features["volume_ratio"] = df["volume"] / (avg_volume + 1e-8)
        features["volume_trend"] = df["volume"].pct_change(5) * 100

        # Padrões (placeholder - será preenchido se disponível)
        features["bullish_patterns"] = 0
        features["bearish_patterns"] = 0

        # Preencher NaN
        features = features.fillna(0)

        return features

    def prepare_data(
        self, df: pd.DataFrame, labels: list[str] | None = None, fit_scaler: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Prepara dados para treinamento/previsão.

        Args:
            df: DataFrame com dados
            labels: Labels (BUY/SELL/HOLD)
            fit_scaler: Se True, calcula parâmetros de normalização

        Returns:
            X, y: Arrays preparados
        """
        # Extrair features
        features_df = self.extract_features(df)

        # Selecionar apenas features disponíveis
        available_features = [col for col in self.feature_columns if col in features_df.columns]
        X = features_df[available_features].values

        # Normalizar
        if fit_scaler or self.scaler_params is None:
            self.scaler_params = {
                "mean": X.mean(axis=0),
                "std": X.std(axis=0) + 1e-8,
                "features": available_features,
            }

        X_normalized = (X - self.scaler_params["mean"]) / self.scaler_params["std"]

        # Converter labels
        if labels is not None:
            label_to_idx = {label: idx for idx, label in enumerate(self.classes)}
            y = np.array([label_to_idx[label] for label in labels])
            return X_normalized, y

        return X_normalized, None

    def train(
        self, df: pd.DataFrame, labels: list[str], validation_split: float = 0.2
    ) -> dict[str, Any]:
        """
        Treina o modelo XGBoost.

        Args:
            df: DataFrame com dados históricos
            labels: Labels para cada linha
            validation_split: Proporção para validação

        Returns:
            Dict com métricas de treinamento
        """
        logger.info("🚀 Iniciando treinamento do modelo XGBoost...")

        # Preparar dados
        X, y = self.prepare_data(df, labels, fit_scaler=True)

        # Split train/validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        logger.info(f"Dados de treino: {X_train.shape}, Validação: {X_val.shape}")

        # Criar DMatrix (formato XGBoost)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Parâmetros
        params = {
            "objective": "multi:softprob",
            "num_class": self.num_classes,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "eval_metric": "mlogloss",
            "seed": 42,
        }

        # Treinar
        evals = [(dtrain, "train"), (dval, "val")]
        evals_result = {}

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            evals_result=evals_result,
            verbose_eval=10,
        )

        # Calcular acurácia
        y_train_pred = self.model.predict(dtrain).argmax(axis=1)
        y_val_pred = self.model.predict(dval).argmax(axis=1)

        train_acc = (y_train_pred == y_train).mean()
        val_acc = (y_val_pred == y_val).mean()

        self.is_trained = True
        logger.info("✅ Treinamento concluído!")

        return {
            "train_loss": evals_result["train"]["mlogloss"],
            "val_loss": evals_result["val"]["mlogloss"],
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "feature_importance": self.get_feature_importance(),
        }

    def predict(self, df: pd.DataFrame, return_probabilities: bool = True) -> dict[str, Any]:
        """
        Classifica trade.

        Args:
            df: DataFrame com dados recentes
            return_probabilities: Se True, retorna probabilidades

        Returns:
            Dict com classificação e confiança
        """
        if not self.is_trained or self.model is None:
            logger.warning("Modelo não treinado. Retornando classificação padrão.")
            return self._get_default_prediction(df)

        try:
            # Preparar dados
            X, _ = self.prepare_data(df, fit_scaler=False)

            if len(X) == 0:
                return self._get_default_prediction(df)

            # Pegar última linha
            X_last = X[-1:]
            dtest = xgb.DMatrix(X_last)

            # Predição
            probabilities = self.model.predict(dtest)[0]
            predicted_class = np.argmax(probabilities)

            signal = self.classes[predicted_class]
            confidence = float(probabilities[predicted_class])

            result = {
                "signal": signal,
                "confidence": confidence,
                "model": "XGBoost",
                "is_trained": True,
            }

            if return_probabilities:
                result["probabilities"] = {
                    "HOLD": float(probabilities[0]),
                    "BUY": float(probabilities[1]),
                    "SELL": float(probabilities[2]),
                }

            return result

        except Exception as e:
            logger.error(f"Erro na classificação: {e}", exc_info=True)
            return self._get_default_prediction(df)

    def _get_default_prediction(self, df: pd.DataFrame) -> dict[str, Any]:
        """Retorna classificação padrão quando modelo não disponível"""
        # Usar tendência simples como fallback
        if len(df) >= 10:
            recent_change = (df["close"].iloc[-1] - df["close"].iloc[-10]) / df["close"].iloc[-10]

            if recent_change > 0.02:
                signal = "BUY"
                confidence = 0.60
            elif recent_change < -0.02:
                signal = "SELL"
                confidence = 0.60
            else:
                signal = "HOLD"
                confidence = 0.70
        else:
            signal = "HOLD"
            confidence = 0.50

        return {
            "signal": signal,
            "confidence": confidence,
            "probabilities": {"HOLD": 0.33, "BUY": 0.33, "SELL": 0.34},
            "model": "FALLBACK",
            "is_trained": False,
        }

    def get_feature_importance(self) -> dict[str, float]:
        """Retorna importância das features"""
        if self.model is None:
            return {}

        importance = self.model.get_score(importance_type="weight")

        # Mapear de f0, f1, ... para nomes reais
        feature_names = self.scaler_params.get("features", self.feature_columns)

        importance_dict = {}
        for i, name in enumerate(feature_names):
            key = f"f{i}"
            if key in importance:
                importance_dict[name] = importance[key]

        # Ordenar por importância
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return importance_dict

    def save_model(self, path: str):
        """Salva modelo treinado"""
        if self.model is None:
            logger.warning("Nenhum modelo para salvar")
            return

        # Salvar modelo XGBoost
        model_path = path.replace(".pkl", ".xgb")
        self.model.save_model(model_path)

        # Salvar metadata
        metadata = {
            "scaler_params": {
                "mean": self.scaler_params["mean"].tolist(),
                "std": self.scaler_params["std"].tolist(),
                "features": self.scaler_params["features"],
            },
            "classes": self.classes,
            "feature_columns": self.feature_columns,
            "is_trained": self.is_trained,
        }

        metadata_path = path.replace(".pkl", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Modelo salvo em: {model_path}")

    def load_model(self, path: str):
        """Carrega modelo treinado"""
        try:
            # Carregar modelo XGBoost
            model_path = path.replace(".pkl", ".xgb")
            self.model = xgb.Booster()
            self.model.load_model(model_path)

            # Carregar metadata
            metadata_path = path.replace(".pkl", "_metadata.json")
            with open(metadata_path) as f:
                metadata = json.load(f)

            self.scaler_params = {
                "mean": np.array(metadata["scaler_params"]["mean"]),
                "std": np.array(metadata["scaler_params"]["std"]),
                "features": metadata["scaler_params"]["features"],
            }
            self.classes = metadata["classes"]
            self.feature_columns = metadata["feature_columns"]
            self.is_trained = metadata["is_trained"]

            logger.info(f"Modelo carregado de: {model_path}")

        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            self.is_trained = False


# Exemplo de uso
if __name__ == "__main__":
    # Gerar dados de teste
    np.random.seed(42)

    data = {
        "open": np.random.randn(500).cumsum() + 50000,
        "high": np.random.randn(500).cumsum() + 50200,
        "low": np.random.randn(500).cumsum() + 49800,
        "close": np.random.randn(500).cumsum() + 50100,
        "volume": np.random.randint(1000000, 2000000, 500),
        "rsi": np.random.uniform(30, 70, 500),
        "macd": np.random.randn(500),
        "macd_signal": np.random.randn(500),
        "bb_upper": np.random.randn(500).cumsum() + 51000,
        "bb_middle": np.random.randn(500).cumsum() + 50000,
        "bb_lower": np.random.randn(500).cumsum() + 49000,
        "ema_9": np.random.randn(500).cumsum() + 50000,
        "ema_21": np.random.randn(500).cumsum() + 50000,
    }
    df = pd.DataFrame(data)

    # Gerar labels baseados em tendência
    labels = []
    for i in range(len(df)):
        if i < 10:
            labels.append("HOLD")
        else:
            change = (df["close"].iloc[i] - df["close"].iloc[i - 10]) / df["close"].iloc[i - 10]
            if change > 0.02:
                labels.append("BUY")
            elif change < -0.02:
                labels.append("SELL")
            else:
                labels.append("HOLD")

    # Criar e treinar modelo
    classifier = XGBoostTradeClassifier(n_estimators=50, max_depth=5)

    print("\n🚀 Treinando modelo XGBoost...")
    history = classifier.train(df, labels)

    print("\n✅ Treinamento concluído!")
    print(f"Acurácia (treino): {history['train_accuracy']:.2%}")
    print(f"Acurácia (validação): {history['val_accuracy']:.2%}")

    print("\n📊 Feature Importance (Top 10):")
    importance = history["feature_importance"]
    for i, (feature, score) in enumerate(list(importance.items())[:10], 1):
        print(f"  {i}. {feature}: {score}")

    # Fazer previsão
    print("\n🎯 Classificando trade...")
    prediction = classifier.predict(df)

    print(f"\nSinal: {prediction['signal']}")
    print(f"Confiança: {prediction['confidence']:.0%}")
    print("\nProbabilidades:")
    for signal, prob in prediction["probabilities"].items():
        print(f"  {signal}: {prob:.0%}")
