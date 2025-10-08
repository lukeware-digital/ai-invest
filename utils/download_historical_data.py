"""
CeciAI - Download Historical Data
Script para baixar dados históricos da CoinAPI e salvar localmente

Features:
- Download de dados históricos (30-90 dias)
- Salvamento no DuckDB
- Backup automático em Parquet
- Múltiplos símbolos e timeframes
- Progress bar
- Retry automático

Autor: CeciAI Team
Data: 2025-10-07
"""

import argparse
import asyncio
from datetime import datetime

from loguru import logger
from tqdm import tqdm

from utils.backup_manager import BackupManager
from utils.coinapi_client import CoinAPIClient, CoinAPIMode
from utils.database import AsyncDatabaseManager


class HistoricalDataDownloader:
    """
    Baixa dados históricos da CoinAPI e salva localmente.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
        days: int = 30,
        api_key: str | None = None,
    ):
        """
        Inicializa downloader.

        Args:
            symbols: Lista de símbolos (default: BTC/USD, ETH/USD)
            timeframes: Lista de timeframes (default: 1min, 5min, 1h, 4h, 1d)
            days: Dias de histórico a baixar (default: 30)
            api_key: Chave da API CoinAPI
        """
        self.symbols = symbols or ["BTC/USD", "ETH/USD"]
        self.timeframes = timeframes or ["1min", "5min", "1h", "4h", "1d"]
        self.days = days
        self.api_key = api_key

        # Estatísticas
        self.stats = {
            "total_candles": 0,
            "total_requests": 0,
            "failed_requests": 0,
            "start_time": None,
            "end_time": None,
        }

        logger.info("HistoricalDataDownloader inicializado:")
        logger.info(f"  • Símbolos: {self.symbols}")
        logger.info(f"  • Timeframes: {self.timeframes}")
        logger.info(f"  • Dias: {days}")

    async def download_all(self):
        """
        Baixa todos os dados históricos.
        """
        logger.info("╔══════════════════════════════════════════════════════╗")
        logger.info("║                                                      ║")
        logger.info("║     📥 Download de Dados Históricos                 ║")
        logger.info("║                                                      ║")
        logger.info("╚══════════════════════════════════════════════════════╝")
        logger.info("")

        self.stats["start_time"] = datetime.now()

        # Calcular total de operações
        total_ops = len(self.symbols) * len(self.timeframes)

        async with CoinAPIClient(api_key=self.api_key, mode=CoinAPIMode.PRODUCTION) as client:
            db = AsyncDatabaseManager()
            backup_mgr = BackupManager()

            # Progress bar
            with tqdm(total=total_ops, desc="Baixando dados") as pbar:
                for symbol in self.symbols:
                    for timeframe in self.timeframes:
                        try:
                            # Calcular limite de candles
                            limit = self._calculate_limit(timeframe, self.days)

                            logger.info(f"📊 Baixando {symbol} {timeframe} ({limit} candles)...")

                            # Buscar dados
                            df = await client.get_ohlcv_data(
                                symbol=symbol, timeframe=timeframe, limit=limit, force_refresh=True
                            )

                            if df.empty:
                                logger.warning(
                                    f"  ⚠️  Nenhum dado retornado para {symbol} {timeframe}"
                                )
                                self.stats["failed_requests"] += 1
                            else:
                                # Salvar no DuckDB
                                rows = await db.save_ohlcv_data(symbol, timeframe, df)
                                self.stats["total_candles"] += rows

                                logger.success(f"  ✅ {rows} candles salvos")

                                # Backup em Parquet
                                backup_mgr.backup_ohlcv_data(
                                    symbol, timeframe, df, backup_type="historical"
                                )

                                logger.info("  💾 Backup criado")

                            self.stats["total_requests"] += 1

                        except Exception as e:
                            logger.error(f"  ❌ Erro ao baixar {symbol} {timeframe}: {e}")
                            self.stats["failed_requests"] += 1

                        finally:
                            pbar.update(1)
                            # Aguardar 2s entre requests (rate limit)
                            await asyncio.sleep(2)

            # Fechar conexões
            await db.close()

        self.stats["end_time"] = datetime.now()

        # Exibir resumo
        self._print_summary()

    def _calculate_limit(self, timeframe: str, days: int) -> int:
        """
        Calcula quantos candles buscar baseado no timeframe.

        Args:
            timeframe: Timeframe (1min, 1h, etc)
            days: Dias de histórico

        Returns:
            Número de candles
        """
        # Candles por dia para cada timeframe
        candles_per_day = {"1min": 1440, "5min": 288, "15min": 96, "1h": 24, "4h": 6, "1d": 1}

        cpd = candles_per_day.get(timeframe, 24)
        return cpd * days

    def _print_summary(self):
        """Exibe resumo do download"""
        elapsed = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

        logger.info("")
        logger.info("╔══════════════════════════════════════════════════════╗")
        logger.info("║                                                      ║")
        logger.info("║     ✅ Download Concluído                           ║")
        logger.info("║                                                      ║")
        logger.info("╚══════════════════════════════════════════════════════╝")
        logger.info("")
        logger.info("📊 Estatísticas:")
        logger.info(f"  • Total de candles: {self.stats['total_candles']:,}")
        logger.info(f"  • Total de requests: {self.stats['total_requests']}")
        logger.info(f"  • Requests falhados: {self.stats['failed_requests']}")
        logger.info(f"  • Tempo total: {elapsed:.2f}s")
        logger.info(f"  • Símbolos: {len(self.symbols)}")
        logger.info(f"  • Timeframes: {len(self.timeframes)}")
        logger.info(f"  • Dias de histórico: {self.days}")
        logger.info("")

        # Calcular tamanho estimado
        estimated_size_mb = self.stats["total_candles"] * 0.0001  # ~100 bytes por candle
        logger.info(f"💾 Tamanho estimado: {estimated_size_mb:.2f} MB")
        logger.info("")

        # Taxa de sucesso
        success_rate = (
            (self.stats["total_requests"] - self.stats["failed_requests"])
            / self.stats["total_requests"]
            * 100
            if self.stats["total_requests"] > 0
            else 0
        )
        logger.info(f"✅ Taxa de sucesso: {success_rate:.1f}%")
        logger.info("")


# ==================== CLI ====================


async def main():
    """Entry point"""
    parser = argparse.ArgumentParser(
        description="CeciAI - Download Historical Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Download padrão (30 dias, BTC/USD e ETH/USD, todos timeframes)
  python download_historical_data.py

  # Download de 90 dias
  python download_historical_data.py --days 90

  # Download de símbolos específicos
  python download_historical_data.py --symbols BTC/USD ETH/USD

  # Download de timeframes específicos
  python download_historical_data.py --timeframes 1h 4h 1d

  # Personalizado
  python download_historical_data.py \\
    --symbols BTC/USD ETH/USD \\
    --timeframes 1h 4h 1d \\
    --days 60 \\
    --api-key YOUR_API_KEY
        """,
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USD", "ETH/USD"],
        help="Símbolos a baixar (default: BTC/USD ETH/USD)",
    )

    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1min", "5min", "1h", "4h", "1d"],
        help="Timeframes a baixar (default: 1min 5min 1h 4h 1d)",
    )

    parser.add_argument("--days", type=int, default=30, help="Dias de histórico (default: 30)")

    parser.add_argument(
        "--api-key", type=str, default=None, help="CoinAPI key (ou use variável COINAPI_KEY)"
    )

    args = parser.parse_args()

    # Banner
    print("╔══════════════════════════════════════════════════════╗")
    print("║                                                      ║")
    print("║     CeciAI - Download Historical Data               ║")
    print("║                                                      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print(f"Símbolos: {args.symbols}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Dias: {args.days}")
    print()

    # Confirmar
    print("⚠️  ATENÇÃO:")
    print(f"  • Total de requests: {len(args.symbols) * len(args.timeframes)}")
    print(f"  • Tempo estimado: {len(args.symbols) * len(args.timeframes) * 2 / 60:.1f} minutos")
    print()

    response = input("Continuar? (s/N): ")
    if response.lower() != "s":
        print("❌ Cancelado")
        return

    print()

    # Download
    downloader = HistoricalDataDownloader(
        symbols=args.symbols, timeframes=args.timeframes, days=args.days, api_key=args.api_key
    )

    await downloader.download_all()

    print()
    print("✅ Download concluído!")
    print()
    print("💡 Próximos passos:")
    print("  1. Verifique os dados: make shell")
    print("  2. Inicie o sistema: make up")
    print("  3. Teste a API: curl http://localhost:8000/health")
    print()


if __name__ == "__main__":
    asyncio.run(main())
