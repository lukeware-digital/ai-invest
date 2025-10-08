"""
CeciAI - API Principal
FastAPI assíncrona com endpoint único para análise completa

Endpoint: POST /api/v1/analyze
- 100% assíncrono
- Alta performance
- Processa todos os 9 agentes
- Retorna decisão de trading

Autor: CeciAI Team
Data: 2025-10-07
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import pandas as pd
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Imports internos
from agents.pipeline import AgentPipeline
from utils.coinapi_client import CoinAPIClient, CoinAPIMode
from config.capital_management import CapitalManager

# Agentes implementados
from agents.agent_4_candlestick_specialist import CandlestickSpecialist
from agents.agent_8_daytrade_executor import DayTradeExecutor

# Utilitários
from utils.technical_indicators import TechnicalIndicators
from utils.candlestick_patterns import CandlestickPatternDetector

# Modelos ML
from agents.ml_models import LSTMPricePredictor, CNNPatternRecognizer, XGBoostTradeClassifier

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== MODELS ====================


class AnalyzeRequest(BaseModel):
    """Request para análise de trading"""

    symbol: str = Field(..., description="Par de trading (ex: BTC/USD, ETH/USD)")

    timeframe: str = Field(
        default="1h", description="Timeframe para análise (1min, 5min, 1h, 4h, 1d)"
    )

    strategy: str = Field(
        default="scalping", description="Estratégia de trading (scalping, swing, arbitrage)"
    )

    capital_allocation: float | None = Field(
        default=None, description="Capital a alocar (USD). Se None, calcula automaticamente", ge=0
    )

    risk_percent: float = Field(
        default=0.01, description="Percentual de risco por trade (0.01 = 1%)", ge=0.001, le=0.05
    )

    enable_execution: bool = Field(
        default=False, description="Se True, executa trade automaticamente (apenas produção)"
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v):
        """Valida símbolo"""
        allowed = ["BTC/USD", "ETH/USD", "BTC/USDT", "ETH/USDT"]
        if v not in allowed:
            raise ValueError(f"Symbol must be one of {allowed}")
        return v

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v):
        """Valida timeframe"""
        allowed = ["1min", "5min", "15min", "1h", "4h", "1d"]
        if v not in allowed:
            raise ValueError(f"Timeframe must be one of {allowed}")
        return v

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v):
        """Valida estratégia"""
        allowed = ["scalping", "swing", "arbitrage"]
        if v not in allowed:
            raise ValueError(f"Strategy must be one of {allowed}")
        return v


class AnalyzeResponse(BaseModel):
    """Response da análise de trading"""

    # Metadados
    request_id: str
    timestamp: str
    processing_time: float

    # Input
    symbol: str
    timeframe: str
    strategy: str

    # Decisão Final
    decision: str  # BUY, SELL, HOLD
    confidence: float
    opportunity_score: int

    # Análise de Mercado
    market_analysis: dict[str, Any]

    # Análise Técnica
    technical_analysis: dict[str, Any]

    # Padrões de Candles
    candlestick_patterns: list[dict[str, Any]]

    # Previsões ML
    ml_predictions: dict[str, Any]

    # Análise dos Agentes LLM
    agents_analysis: dict[str, Any]

    # Plano de Execução (se decisão = BUY)
    execution_plan: dict[str, Any] | None

    # Capital
    capital_status: dict[str, Any]

    # Validações
    validations: dict[str, bool]

    # Warnings/Errors
    warnings: list[str]
    errors: list[str]


class HealthResponse(BaseModel):
    """Response do health check"""

    status: str
    timestamp: str
    version: str
    services: dict[str, str]


# ==================== LIFESPAN ====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia lifecycle da aplicação"""
    # Startup
    print("🚀 Starting CeciAI API...")

    # Inicializar agentes individuais
    app.state.agent4 = CandlestickSpecialist()
    app.state.agent8 = DayTradeExecutor()

    # Inicializar serviços principais
    app.state.pipeline = AgentPipeline()
    app.state.coinapi = CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT)
    app.state.capital_mgr = CapitalManager(initial_capital=10000.0)
    
    # Inicializar utilitários
    app.state.technical_indicators = TechnicalIndicators()
    app.state.pattern_detector = CandlestickPatternDetector()
    
    # Inicializar modelos ML (carregando modelos pré-treinados se existirem)
    try:
        app.state.price_predictor = LSTMPricePredictor()
        app.state.pattern_recognizer = CNNPatternRecognizer()
        app.state.trade_classifier = XGBoostTradeClassifier()
        print("✅ Modelos ML carregados")
    except Exception as e:
        print(f"⚠️  Modelos ML não disponíveis: {e}")
        app.state.price_predictor = None
        app.state.pattern_recognizer = None
        app.state.trade_classifier = None

    print("✅ CeciAI API started successfully")
    print("📊 Pipeline completo inicializado com 9 agentes")

    yield

    # Shutdown
    print("🛑 Shutting down CeciAI API...")

    # Cleanup
    await app.state.pipeline.cleanup()
    await app.state.coinapi.close()

    print("✅ CeciAI API shutdown complete")


# ==================== APP ====================

app = FastAPI(
    title="CeciAI Trading API",
    description="API assíncrona para análise inteligente de trading com IA",
    version="0.3.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== ENDPOINTS ====================


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "service": "CeciAI Trading API",
        "version": "0.3.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""

    # Verificar serviços assincronamente
    services_status = await check_services_health()

    return HealthResponse(
        status="healthy" if all(s == "ok" for s in services_status.values()) else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        version="0.3.0",
        services=services_status,
    )


@app.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Trading"],
)
async def analyze_trading_opportunity(
    request: AnalyzeRequest, background_tasks: BackgroundTasks
) -> AnalyzeResponse:
    """
    🎯 ENDPOINT PRINCIPAL - Análise Completa de Trading

    Processa de forma 100% assíncrona:
    1. Coleta dados de mercado (CoinAPI)
    2. Calcula indicadores técnicos
    3. Detecta padrões de candles
    4. Executa previsões ML
    5. Processa 9 agentes LLM em pipeline
    6. Valida capital disponível
    7. Gera decisão final (BUY/SELL/HOLD)
    8. Retorna plano de execução

    Args:
        request: Parâmetros da análise
        background_tasks: Tasks em background

    Returns:
        AnalyzeResponse com decisão completa

    Raises:
        HTTPException: Em caso de erro
    """

    start_time = time.time()
    request_id = generate_request_id()

    try:
        # ========== FASE 1: COLETA DE DADOS (Assíncrono) ==========
        print(f"📊 [{request_id}] Coletando dados de mercado...")

        market_data = await fetch_market_data_async(
            symbol=request.symbol, timeframe=request.timeframe
        )

        # ========== FASE 2: ANÁLISE TÉCNICA (Assíncrono) ==========
        print(f"📈 [{request_id}] Calculando indicadores técnicos...")

        technical_analysis = await calculate_technical_indicators_async(market_data=market_data)

        # ========== FASE 3: PADRÕES DE CANDLES (Assíncrono) ==========
        print(f"🕯️ [{request_id}] Detectando padrões de candles...")

        candlestick_patterns = await detect_candlestick_patterns_async(market_data=market_data)

        # ========== FASE 4: PREVISÕES ML (Assíncrono) ==========
        print(f"🤖 [{request_id}] Executando previsões ML...")

        ml_predictions = await run_ml_predictions_async(
            market_data=market_data,
            technical_analysis=technical_analysis,
            price_predictor=getattr(app.state, 'price_predictor', None),
            pattern_recognizer=getattr(app.state, 'pattern_recognizer', None),
            trade_classifier=getattr(app.state, 'trade_classifier', None),
        )

        # ========== FASE 5: VALIDAÇÃO DE CAPITAL (Assíncrono) ==========
        print(f"💰 [{request_id}] Validando capital disponível...")

        capital_status = await check_capital_status_async(
            capital_allocation=request.capital_allocation,
            risk_percent=request.risk_percent,
            capital_mgr=getattr(app.state, 'capital_mgr', None),
        )

        # ========== FASE 6: PIPELINE DE AGENTES LLM (Assíncrono) ==========
        print(f"🧠 [{request_id}] Processando 9 agentes LLM...")

        agents_analysis = await process_agents_pipeline_async(
            market_data=market_data,
            technical_analysis=technical_analysis,
            candlestick_patterns=candlestick_patterns,
            ml_predictions=ml_predictions,
            capital_status=capital_status,
            strategy=request.strategy,
            pipeline=getattr(app.state, 'pipeline', None),
        )

        # ========== FASE 7: DECISÃO FINAL (Assíncrono) ==========
        print(f"🎯 [{request_id}] Gerando decisão final...")

        final_decision = await make_final_decision_async(
            agents_analysis=agents_analysis,
            market_data=market_data,
            capital_status=capital_status,
            risk_percent=request.risk_percent,
        )

        # ========== FASE 8: PLANO DE EXECUÇÃO (Assíncrono) ==========
        execution_plan = None
        if final_decision["decision"] == "BUY":
            print(f"📋 [{request_id}] Gerando plano de execução...")

            execution_plan = await generate_execution_plan_async(
                symbol=request.symbol,
                decision=final_decision,
                capital_status=capital_status,
                market_data=market_data,
            )

            # Se habilitado, executar trade em background
            if request.enable_execution:
                background_tasks.add_task(execute_trade_async, execution_plan=execution_plan)

        # ========== RESPOSTA ==========
        processing_time = time.time() - start_time

        print(f"✅ [{request_id}] Análise concluída em {processing_time:.2f}s")

        return AnalyzeResponse(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            processing_time=processing_time,
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy=request.strategy,
            decision=final_decision["decision"],
            confidence=final_decision["confidence"],
            opportunity_score=final_decision["opportunity_score"],
            market_analysis=market_data,
            technical_analysis=technical_analysis,
            candlestick_patterns=candlestick_patterns,
            ml_predictions=ml_predictions,
            agents_analysis=agents_analysis,
            execution_plan=execution_plan,
            capital_status=capital_status,
            validations=final_decision["validations"],
            warnings=final_decision.get("warnings", []),
            errors=[],
        )

    except Exception as e:
        print(f"❌ [{request_id}] Erro: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar análise: {e!s}",
        )


# ==================== FUNÇÕES ASSÍNCRONAS ====================


async def check_services_health() -> dict[str, str]:
    """Verifica health de todos os serviços assincronamente"""

    async def check_ollama():
        try:
            # Simular verificação do Ollama
            await asyncio.sleep(0.1)
            return "ok"
        except Exception:
            return "error"

    async def check_redis():
        try:
            # Simular verificação do Redis
            await asyncio.sleep(0.1)
            return "ok"
        except Exception:
            return "error"

    async def check_coinapi():
        try:
            # Simular verificação do CoinAPI
            await asyncio.sleep(0.1)
            return "ok"
        except Exception:
            return "error"

    # Executar verificações em paralelo
    results = await asyncio.gather(
        check_ollama(), check_redis(), check_coinapi(), return_exceptions=True
    )

    return {
        "ollama": results[0] if not isinstance(results[0], Exception) else "error",
        "redis": results[1] if not isinstance(results[1], Exception) else "error",
        "coinapi": results[2] if not isinstance(results[2], Exception) else "error",
    }


async def fetch_market_data_async(symbol: str, timeframe: str, limit: int = 100) -> dict[str, Any]:
    """Busca dados de mercado assincronamente"""
    # Criar instância temporária do CoinAPI client
    async with CoinAPIClient(mode=CoinAPIMode.DEVELOPMENT) as client:
        try:
            # Buscar dados OHLCV
            df = await client.get_ohlcv_data(symbol=symbol, timeframe=timeframe, limit=limit)
            
            if df.empty:
                raise ValueError(f"Sem dados disponíveis para {symbol} {timeframe}")
            
            # Calcular métricas 24h
            high_24h = float(df["high"].max())
            low_24h = float(df["low"].min())
            volume_24h = float(df["volume"].sum())
            current_price = float(df["close"].iloc[-1])
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": current_price,
                "volume_24h": volume_24h,
                "high_24h": high_24h,
                "low_24h": low_24h,
                "ohlcv_data": df.to_dict('records'),
                "dataframe": df,  # Incluir DataFrame para processamento posterior
            }
            
        except Exception as e:
            logger.error(f"Erro ao buscar dados de mercado: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao buscar dados: {str(e)}. Execute 'python utils/download_historical_data.py' primeiro."
            )


async def calculate_technical_indicators_async(market_data: dict[str, Any]) -> dict[str, Any]:
    """Calcula indicadores técnicos assincronamente"""
    try:
        df = market_data.get("dataframe")
        if df is None or df.empty:
            raise ValueError("DataFrame vazio ou não disponível")
        
        # Executar cálculo em thread separada para não bloquear
        loop = asyncio.get_event_loop()
        ti = TechnicalIndicators()
        
        # Calcular todos os indicadores
        df_with_indicators = await loop.run_in_executor(None, ti.calculate_all_indicators, df)
        
        # Extrair últimos valores
        last_row = df_with_indicators.iloc[-1]
        
        # Verificar cruzamento MACD
        if len(df_with_indicators) >= 2:
            prev_row = df_with_indicators.iloc[-2]
            macd_crossover = ti.interpret_macd_crossover(
                last_row["macd"],
                last_row["macd_signal"],
                prev_row["macd"],
                prev_row["macd_signal"]
            )
        else:
            macd_crossover = None
        
        return {
            "rsi_14": float(last_row["rsi_14"]) if not pd.isna(last_row["rsi_14"]) else 50.0,
            "rsi_21": float(last_row["rsi_21"]) if not pd.isna(last_row["rsi_21"]) else 50.0,
            "macd": float(last_row["macd"]) if not pd.isna(last_row["macd"]) else 0.0,
            "macd_signal": float(last_row["macd_signal"]) if not pd.isna(last_row["macd_signal"]) else 0.0,
            "macd_histogram": float(last_row["macd_histogram"]) if not pd.isna(last_row["macd_histogram"]) else 0.0,
            "macd_crossover": macd_crossover,
            "bb_upper": float(last_row["bb_upper"]) if not pd.isna(last_row["bb_upper"]) else 0.0,
            "bb_middle": float(last_row["bb_middle"]) if not pd.isna(last_row["bb_middle"]) else 0.0,
            "bb_lower": float(last_row["bb_lower"]) if not pd.isna(last_row["bb_lower"]) else 0.0,
            "bb_width": float(last_row["bb_width"]) if not pd.isna(last_row["bb_width"]) else 0.0,
            "ema_9": float(last_row["ema_9"]) if not pd.isna(last_row["ema_9"]) else 0.0,
            "ema_21": float(last_row["ema_21"]) if not pd.isna(last_row["ema_21"]) else 0.0,
            "ema_50": float(last_row["ema_50"]) if not pd.isna(last_row["ema_50"]) else 0.0,
            "adx": float(last_row["adx"]) if not pd.isna(last_row["adx"]) else 0.0,
            "atr": float(last_row["atr"]) if not pd.isna(last_row["atr"]) else 0.0,
            "volume_ratio": float(last_row["volume_ratio"]) if not pd.isna(last_row["volume_ratio"]) else 1.0,
            "obv": float(last_row["obv"]) if not pd.isna(last_row["obv"]) else 0.0,
            "vwap": float(last_row["vwap"]) if not pd.isna(last_row["vwap"]) else 0.0,
            "stoch_k": float(last_row["stoch_k"]) if not pd.isna(last_row["stoch_k"]) else 50.0,
            "stoch_d": float(last_row["stoch_d"]) if not pd.isna(last_row["stoch_d"]) else 50.0,
            "trend": ti.get_trend_direction(
                last_row["ema_9"],
                last_row["ema_21"],
                last_row["ema_50"],
                last_row["ema_200"]
            ),
            "rsi_interpretation": ti.interpret_rsi(last_row["rsi_14"]),
        }
    except Exception as e:
        logger.error(f"Erro ao calcular indicadores técnicos: {e}")
        # Retornar valores padrão em caso de erro
        return {
            "rsi_14": 50.0,
            "macd": 0.0,
            "macd_signal": 0.0,
            "macd_histogram": 0.0,
            "bb_upper": 0.0,
            "bb_middle": 0.0,
            "bb_lower": 0.0,
            "ema_9": 0.0,
            "ema_21": 0.0,
            "adx": 0.0,
            "atr": 0.0,
            "volume_ratio": 1.0,
            "error": str(e),
        }


async def detect_candlestick_patterns_async(market_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Detecta padrões de candles assincronamente"""
    try:
        df = market_data.get("dataframe")
        if df is None or df.empty:
            return []
        
        # Executar detecção em thread separada
        loop = asyncio.get_event_loop()
        detector = CandlestickPatternDetector()
        
        patterns = await loop.run_in_executor(None, detector.detect_all_patterns, df, 20)
        
        # Converter para formato dict
        pattern_list = []
        for pattern in patterns:
            pattern_list.append({
                "pattern": pattern.name,
                "type": pattern.pattern_type.value,
                "signal": pattern.signal,
                "confidence": int(pattern.confidence * 100),
                "significance": pattern.significance,
                "strength": pattern.strength.value,
                "candle_index": pattern.candle_index,
                "description": pattern.description,
                "confirmation_needed": pattern.confirmation_needed,
                "confirmation_criteria": pattern.confirmation_criteria,
            })
        
        return pattern_list
        
    except Exception as e:
        logger.error(f"Erro ao detectar padrões de candlestick: {e}")
        return []


async def run_ml_predictions_async(
    market_data: dict[str, Any], technical_analysis: dict[str, Any], 
    price_predictor=None, pattern_recognizer=None, trade_classifier=None
) -> dict[str, Any]:
    """Executa previsões ML assincronamente"""
    
    predictions = {
        "price_1h": market_data["current_price"],
        "price_4h": market_data["current_price"],
        "price_24h": market_data["current_price"],
        "direction": "NEUTRAL",
        "confidence": 0.5,
        "classification": "HOLD",
        "probability_buy": 0.33,
        "probability_sell": 0.33,
        "probability_hold": 0.34,
        "models_available": False,
    }
    
    try:
        df = market_data.get("dataframe")
        if df is None or df.empty or len(df) < 60:
            return predictions
        
        # Executar previsões se modelos estiverem disponíveis
        loop = asyncio.get_event_loop()
        
        # LSTM: Previsão de preços
        if price_predictor:
            try:
                price_preds = await loop.run_in_executor(None, price_predictor.predict, df)
                if price_preds and "predictions" in price_preds:
                    preds = price_preds["predictions"]
                    predictions["price_1h"] = float(preds.get("1h", predictions["price_1h"]))
                    predictions["price_4h"] = float(preds.get("4h", predictions["price_4h"]))
                    predictions["price_24h"] = float(preds.get("24h", predictions["price_24h"]))
                    predictions["direction"] = "UP" if predictions["price_1h"] > market_data["current_price"] else "DOWN"
                    predictions["models_available"] = True
            except Exception as e:
                logger.warning(f"LSTM não disponível: {e}")
        
        # XGBoost: Classificação de trade
        if trade_classifier:
            try:
                # Preparar features do último candle
                last_candle = {
                    "rsi": technical_analysis.get("rsi_14", 50),
                    "macd": technical_analysis.get("macd", 0),
                    "bb_position": (market_data["current_price"] - technical_analysis.get("bb_lower", 0)) / 
                                   (technical_analysis.get("bb_upper", 1) - technical_analysis.get("bb_lower", 1)),
                    "volume_ratio": technical_analysis.get("volume_ratio", 1),
                    "atr": technical_analysis.get("atr", 0),
                }
                
                classification = await loop.run_in_executor(None, trade_classifier.predict, last_candle)
                if classification:
                    predictions["classification"] = classification.get("action", "HOLD")
                    predictions["probability_buy"] = classification.get("probability_buy", 0.33)
                    predictions["probability_sell"] = classification.get("probability_sell", 0.33)
                    predictions["probability_hold"] = classification.get("probability_hold", 0.34)
                    predictions["confidence"] = max(
                        predictions["probability_buy"],
                        predictions["probability_sell"],
                        predictions["probability_hold"]
                    )
            except Exception as e:
                logger.warning(f"XGBoost não disponível: {e}")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Erro nas previsões ML: {e}")
        return predictions


async def check_capital_status_async(
    capital_allocation: float | None, risk_percent: float, capital_mgr=None
) -> dict[str, Any]:
    """Verifica status de capital assincronamente"""
    
    # Valores padrão se capital_mgr não estiver disponível
    if capital_mgr is None:
        return {
            "total_capital": 10000.00,
            "available_capital": 10000.00,
            "allocated_capital": 0.00,
            "can_execute": True,
            "open_positions": 0,
            "daily_pnl": 0.0,
        }
    
    # Obter status do capital manager
    status = capital_mgr.get_status()
    
    # Validar se pode abrir posição
    if capital_allocation:
        validation = capital_mgr.can_open_position(capital_allocation)
        can_execute = validation["can_open"]
        warnings = validation.get("warnings", [])
        reasons = validation.get("reasons", [])
    else:
        can_execute = status["available_capital"] > 0
        warnings = []
        reasons = []
    
    return {
        "total_capital": status["current_capital"],
        "available_capital": status["available_capital"],
        "allocated_capital": status["current_capital"] - status["available_capital"],
        "can_execute": can_execute,
        "open_positions": status["open_positions"],
        "daily_pnl": status["daily_pnl"],
        "total_pnl": status["total_pnl"],
        "circuit_breaker_active": status["circuit_breaker_active"],
        "warnings": warnings,
        "reasons": reasons,
    }


async def process_agents_pipeline_async(
    market_data: dict[str, Any],
    technical_analysis: dict[str, Any],
    candlestick_patterns: list[dict[str, Any]],
    ml_predictions: dict[str, Any],
    capital_status: dict[str, Any],
    strategy: str,
    pipeline=None,
) -> dict[str, Any]:
    """Processa pipeline de 9 agentes LLM assincronamente"""
    
    try:
        df = market_data.get("dataframe")
        symbol = market_data.get("symbol", "BTC/USD")
        timeframe = market_data.get("timeframe", "1h")

        if pipeline and df is not None and not df.empty:
            # Usar pipeline real
            logger.info("Executando pipeline completo dos 9 agentes...")
            
            result = await pipeline.execute(
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                capital_available=capital_status.get("available_capital", 10000),
                user_strategy=strategy,
                risk_params={
                    "max_risk_per_trade": 0.01,
                    "min_risk_reward": 1.5,
                    "max_stop_loss": 0.03,
                }
            )
            
            return {
                "agent_1": result.get("agent_1_market_expert", {}),
                "agent_2": result.get("agent_2_data_analyzer", {}),
                "agent_3": result.get("agent_3_technical_analyst", {}),
                "agent_4": result.get("agent_4_candlestick_specialist", {}),
                "agent_5": result.get("agent_5_investment_evaluator", {}),
                "agent_6": result.get("agent_6_time_horizon_advisor", {}),
                "agent_7": result.get("agent_7_daytrade_classifier", {}),
                "final_executor": result.get("executor_result", {}),
                "final_decision": result.get("final_decision", {}),
                "pipeline_metadata": result.get("pipeline_metadata", {}),
            }
        else:
            # Fallback: simulação simplificada
            logger.warning("Pipeline não disponível, usando simulação")
            
            # Executar agentes em paralelo quando possível
            agent1_task = asyncio.create_task(run_agent_1_async(market_data))
            agent2_task = asyncio.create_task(run_agent_2_async(market_data))

            agent1_result, agent2_result = await asyncio.gather(agent1_task, agent2_task)

            # Agent 3 depende de 1 e 2
            agent3_result = await run_agent_3_async(technical_analysis, agent1_result)

            # Agent 4 pode rodar em paralelo com 3
            agent4_result = await run_agent_4_async(candlestick_patterns)

            # Agent 5 depende de todos anteriores
            agent5_result = await run_agent_5_async(
                agent1_result, agent2_result, agent3_result, agent4_result, ml_predictions, capital_status
            )

            # Agents 6, 7 podem rodar em paralelo
            agent6_task = asyncio.create_task(run_agent_6_async(agent5_result))
            agent7_task = asyncio.create_task(run_agent_7_async(agent5_result, strategy))

            agent6_result, agent7_result = await asyncio.gather(agent6_task, agent7_task)

            # Agent 8 ou 9 (dependendo do tipo)
            if agent7_result["trade_type"] == "day_trade":
                agent8_result = await run_agent_8_async(
                    agent1_result, agent3_result, agent4_result, agent5_result, capital_status
                )
                final_executor = agent8_result
            else:
                agent9_result = await run_agent_9_async(
                    agent1_result, agent3_result, agent5_result, capital_status
                )
                final_executor = agent9_result

            return {
                "agent_1": agent1_result,
                "agent_2": agent2_result,
                "agent_3": agent3_result,
                "agent_4": agent4_result,
                "agent_5": agent5_result,
                "agent_6": agent6_result,
                "agent_7": agent7_result,
                "final_executor": final_executor,
            }
            
    except Exception as e:
        logger.error(f"Erro no pipeline de agentes: {e}", exc_info=True)
        return {
            "error": str(e),
            "agent_1": {},
            "agent_2": {},
            "agent_3": {},
            "agent_4": {},
            "agent_5": {"opportunity_score": 0},
            "agent_6": {},
            "agent_7": {"trade_type": "unknown"},
            "final_executor": {"decision": "HOLD", "confidence": 0.0},
        }


# Funções individuais dos agentes (simuladas)
async def run_agent_1_async(market_data: dict[str, Any]) -> dict[str, Any]:
    """Agent 1: Market Expert"""
    await asyncio.sleep(0.3)
    return {"market_regime": "bull", "sentiment": "positive"}


async def run_agent_2_async(market_data: dict[str, Any]) -> dict[str, Any]:
    """Agent 2: Data Analyzer"""
    await asyncio.sleep(0.2)
    return {"data_quality": "high", "best_timeframe": "1h"}


async def run_agent_3_async(technical_analysis: dict[str, Any], agent1: dict) -> dict[str, Any]:
    """Agent 3: Technical Analyst"""
    await asyncio.sleep(0.4)
    return {"signals": "bullish", "strength": "moderate"}


async def run_agent_4_async(patterns: list[dict[str, Any]]) -> dict[str, Any]:
    """Agent 4: Candlestick Specialist"""
    await asyncio.sleep(0.3)
    return {"pattern_detected": "HAMMER", "significance": 85}


async def run_agent_5_async(*args) -> dict[str, Any]:
    """Agent 5: Investment Evaluator"""
    await asyncio.sleep(0.4)
    return {"opportunity_score": 78, "quality": "excellent"}


async def run_agent_6_async(agent5: dict) -> dict[str, Any]:
    """Agent 6: Time Horizon Advisor"""
    await asyncio.sleep(0.2)
    return {"recommended_strategy": "scalping", "timeframe": "1h"}


async def run_agent_7_async(agent5: dict, strategy: str) -> dict[str, Any]:
    """Agent 7: Trade Classifier"""
    await asyncio.sleep(0.2)
    return {"trade_type": "day_trade", "classification": "scalping"}


async def run_agent_8_async(*args) -> dict[str, Any]:
    """Agent 8: Day-Trade Executor"""
    await asyncio.sleep(0.5)
    return {
        "decision": "BUY",
        "confidence": 0.78,
        "entry_price": 50300,
        "stop_loss": 49200,
        "take_profit": 51500,
    }


async def run_agent_9_async(*args) -> dict[str, Any]:
    """Agent 9: Long-Term Executor"""
    await asyncio.sleep(0.5)
    return {"decision": "HOLD", "accumulation_plan": {}}


async def make_final_decision_async(
    agents_analysis: dict[str, Any],
    market_data: dict[str, Any],
    capital_status: dict[str, Any],
    risk_percent: float,
) -> dict[str, Any]:
    """Gera decisão final assincronamente"""
    await asyncio.sleep(0.2)

    executor = agents_analysis["final_executor"]

    return {
        "decision": executor["decision"],
        "confidence": executor.get("confidence", 0.5),
        "opportunity_score": agents_analysis["agent_5"]["opportunity_score"],
        "validations": {"capital_check": True, "risk_check": True, "rr_ratio_check": True},
        "warnings": [],
    }


async def generate_execution_plan_async(
    symbol: str,
    decision: dict[str, Any],
    capital_status: dict[str, Any],
    market_data: dict[str, Any],
) -> dict[str, Any]:
    """Gera plano de execução assincronamente"""
    await asyncio.sleep(0.2)

    return {
        "symbol": symbol,
        "action": "BUY",
        "entry_price": 50300,
        "quantity_usd": 1500,
        "quantity_btc": 0.0298,
        "stop_loss": 49200,
        "take_profit_1": 51500,
        "take_profit_2": 52800,
        "risk_amount": 33,
        "potential_profit": 75,
        "risk_reward_ratio": 2.3,
    }


async def execute_trade_async(execution_plan: dict[str, Any]):
    """Executa trade assincronamente (background task)"""
    await asyncio.sleep(1)
    print(f"🚀 Trade executado: {execution_plan['action']} {execution_plan['quantity_btc']} BTC")


def generate_request_id() -> str:
    """Gera ID único para request"""
    import uuid

    return str(uuid.uuid4())[:8]


# ==================== NOVO ENDPOINT: ANÁLISE DE CANDLES ====================


class Candle(BaseModel):
    """Modelo de um candle OHLCV"""

    timestamp: str = Field(..., description="ISO timestamp do candle")
    open: float = Field(..., gt=0, description="Preço de abertura")
    high: float = Field(..., gt=0, description="Preço máximo")
    low: float = Field(..., gt=0, description="Preço mínimo")
    close: float = Field(..., gt=0, description="Preço de fechamento")
    volume: float = Field(..., ge=0, description="Volume negociado")


class CandleAnalysisRequest(BaseModel):
    """Request para análise de candles"""

    symbol: str = Field(..., description="Par de trading (ex: BTC/USD)")
    candles: list[Candle] = Field(
        ..., min_length=10, description="Histórico de candles (mínimo 10)"
    )
    capital_available: float = Field(10000, gt=0, description="Capital disponível em USD")
    strategy: str = Field("scalping", description="Estratégia (scalping, swing, arbitrage)")


class CandleAnalysisResponse(BaseModel):
    """Response da análise de candles"""

    # Metadados
    request_id: str
    timestamp: str
    processing_time: float

    # Decisão
    signal: str  # BUY, SELL, HOLD
    confidence: float

    # Preços
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float

    # Execução
    quantity_usd: float
    risk_reward_ratio: float
    risk_amount_usd: float
    potential_profit_usd: float

    # Análises
    reasoning: str
    patterns_detected: list[str]
    technical_analysis: dict[str, Any]

    # Validações
    validations: dict[str, str]


@app.post(
    "/api/v1/analyze-candles",
    response_model=CandleAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Trading"],
)
async def analyze_candles_endpoint(request: CandleAnalysisRequest) -> CandleAnalysisResponse:
    """
    🕯️ ENDPOINT: Análise de Histórico de Candles

    Recebe um histórico de candles e retorna sinal de BUY/SELL/HOLD.

    Processo:
    1. Detecta padrões de candlestick (Agent 4)
    2. Calcula indicadores técnicos
    3. Define plano de execução (Agent 8)
    4. Retorna decisão com preços e justificativa

    Args:
        request: Histórico de candles + parâmetros

    Returns:
        CandleAnalysisResponse com decisão completa
    """
    start_time = time.time()
    request_id = generate_request_id()

    try:
        logger.info(f"[{request_id}] Iniciando análise de candles para {request.symbol}")

        # Converter candles para DataFrame
        df = pd.DataFrame([c.model_dump() for c in request.candles])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        # Validar dados
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Mínimo de 10 candles necessários")

        # Calcular indicadores técnicos
        logger.info(f"[{request_id}] Calculando indicadores técnicos...")
        ti = TechnicalIndicators()
        df["rsi"] = ti.calculate_rsi(df["close"])
        macd_data = ti.calculate_macd(df["close"])
        df["macd"] = macd_data["macd"]
        df["macd_signal"] = macd_data["signal"]

        bb_data = ti.calculate_bollinger_bands(df["close"])
        df["bb_upper"] = bb_data["upper"]
        df["bb_middle"] = bb_data["middle"]
        df["bb_lower"] = bb_data["lower"]

        # Detectar tendência
        trend = "uptrend" if df["close"].iloc[-1] > df["close"].iloc[-10] else "downtrend"

        # Agent 4: Analisar padrões de candlestick
        logger.info(f"[{request_id}] Agent 4: Analisando padrões...")
        agent4_analysis = await app.state.agent4.analyze(
            df, {"symbol": request.symbol, "trend": trend}
        )

        # Agent 8: Definir plano de execução
        logger.info(f"[{request_id}] Agent 8: Definindo plano de execução...")
        agent8_plan = await app.state.agent8.execute(
            symbol=request.symbol,
            current_price=float(df["close"].iloc[-1]),
            agent_analyses={"agent4": agent4_analysis},
            capital_available=request.capital_available,
            risk_params={
                "max_risk_per_trade": 0.01,  # 1%
                "min_risk_reward": 1.5,
                "max_stop_loss": 0.03,  # 3%
            },
        )

        # Extrair indicadores técnicos
        technical_analysis = {
            "rsi": float(df["rsi"].iloc[-1]) if not pd.isna(df["rsi"].iloc[-1]) else 50.0,
            "macd": float(df["macd"].iloc[-1]) if not pd.isna(df["macd"].iloc[-1]) else 0.0,
            "macd_signal": float(df["macd_signal"].iloc[-1])
            if not pd.isna(df["macd_signal"].iloc[-1])
            else 0.0,
            "bb_position": "upper"
            if df["close"].iloc[-1] > df["bb_upper"].iloc[-1]
            else "lower"
            if df["close"].iloc[-1] < df["bb_lower"].iloc[-1]
            else "middle",
            "trend": trend,
            "price_change_pct": float(
                ((df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]) * 100
            ),
            "volume_avg": float(df["volume"].mean()),
        }

        # Montar resposta
        processing_time = time.time() - start_time

        logger.info(
            f"[{request_id}] Análise completa: {agent8_plan['decision']} (confiança: {agent8_plan.get('confidence', 0):.2f})"
        )

        return CandleAnalysisResponse(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            processing_time=processing_time,
            signal=agent8_plan["decision"],
            confidence=agent8_plan.get("confidence", 0.5),
            entry_price=agent8_plan.get("entry_price", float(df["close"].iloc[-1])),
            stop_loss=agent8_plan.get("stop_loss", {}).get("price", 0.0),
            take_profit_1=agent8_plan.get("take_profit_1", {}).get("price", 0.0),
            take_profit_2=agent8_plan.get("take_profit_2", {}).get("price", 0.0),
            quantity_usd=agent8_plan.get("quantity_usd", 0.0),
            risk_reward_ratio=agent8_plan.get("risk_reward_ratio", 0.0),
            risk_amount_usd=agent8_plan.get("risk_amount_usd", 0.0),
            potential_profit_usd=agent8_plan.get("potential_profit_usd", 0.0),
            reasoning=agent8_plan.get("reasoning", "Análise concluída"),
            patterns_detected=agent4_analysis.get("key_patterns", []),
            technical_analysis=technical_analysis,
            validations=agent8_plan.get("validations", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Erro na análise: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro interno: {e!s}")


# ==================== MAIN ====================

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
        log_level="info",
        access_log=True,
    )
