"""
CeciAI - Data Updater
Sistema de atualização automática da base local

Features:
- Atualização agendada (padrão: 12h em 12h)
- Hora inicial configurável
- Atualização incremental (apenas dados novos)
- Retry automático em caso de falha
- Logs detalhados

Autor: CeciAI Team
Data: 2025-10-07
"""

import asyncio
from datetime import datetime, timedelta

from loguru import logger

from utils.backup_manager import BackupManager
from utils.coinapi_client import CoinAPIClient, CoinAPIMode
from utils.database import AsyncDatabaseManager


class DataUpdater:
    """
    Atualizador automático de dados.

    Busca dados da CoinAPI e atualiza base local periodicamente.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
        update_interval_hours: int = 12,
        start_time: str = "03:00",  # Hora inicial (formato HH:MM)
        enable_backup: bool = True,
        max_retries: int = 3,
    ):
        """
        Inicializa o atualizador.

        Args:
            symbols: Lista de símbolos (default: BTC/USD, ETH/USD)
            timeframes: Lista de timeframes (default: 1min, 5min, 1h, 4h, 1d)
            update_interval_hours: Intervalo de atualização em horas (default: 12)
            start_time: Hora inicial no formato HH:MM (default: 03:00)
            enable_backup: Se True, faz backup após atualização
            max_retries: Número máximo de tentativas em caso de erro
        """
        self.symbols = symbols or ["BTC/USD", "ETH/USD"]
        self.timeframes = timeframes or ["1min", "5min", "1h", "4h", "1d"]
        self.update_interval_hours = update_interval_hours
        self.start_time = start_time
        self.enable_backup = enable_backup
        self.max_retries = max_retries

        # Estatísticas
        self.stats = {
            "last_update": None,
            "next_update": None,
            "total_updates": 0,
            "total_candles_updated": 0,
            "failed_updates": 0,
        }

        logger.info("DataUpdater inicializado:")
        logger.info(f"  • Símbolos: {self.symbols}")
        logger.info(f"  • Timeframes: {self.timeframes}")
        logger.info(f"  • Intervalo: {update_interval_hours}h")
        logger.info(f"  • Hora inicial: {start_time}")

    async def start(self):
        """
        Inicia o atualizador em modo daemon.

        Roda indefinidamente, atualizando a base conforme agendamento.
        """
        logger.info("🚀 Iniciando Data Updater...")

        # Calcular próxima execução
        next_run = self._calculate_next_run()
        self.stats["next_update"] = next_run

        logger.info(f"⏰ Próxima atualização: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")

        # Executar imediatamente na primeira vez (opcional)
        # await self.update_data()

        # Loop principal
        while True:
            now = datetime.now()

            # Verificar se é hora de atualizar
            if now >= next_run:
                logger.info("⏰ Hora de atualizar!")

                # Executar atualização
                success = await self.update_data()

                if success:
                    self.stats["total_updates"] += 1
                else:
                    self.stats["failed_updates"] += 1

                # Calcular próxima execução
                next_run = self._calculate_next_run()
                self.stats["next_update"] = next_run

                logger.info(f"⏰ Próxima atualização: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")

            # Aguardar 1 minuto antes de verificar novamente
            await asyncio.sleep(60)

    async def update_data(self) -> bool:
        """
        Atualiza dados da base local com dados da CoinAPI.

        Returns:
            True se sucesso, False se falhou
        """
        logger.info("╔══════════════════════════════════════════════════════╗")
        logger.info("║                                                      ║")
        logger.info("║     📥 Atualizando Base Local                       ║")
        logger.info("║                                                      ║")
        logger.info("╚══════════════════════════════════════════════════════╝")

        start_time = datetime.now()
        total_candles = 0

        try:
            async with CoinAPIClient(mode=CoinAPIMode.PRODUCTION) as client:
                db = AsyncDatabaseManager()
                backup_mgr = BackupManager() if self.enable_backup else None

                # Atualizar cada símbolo e timeframe
                for symbol in self.symbols:
                    for timeframe in self.timeframes:
                        retries = 0

                        while retries < self.max_retries:
                            try:
                                logger.info(f"📊 Atualizando {symbol} {timeframe}...")

                                # Buscar última data na base local
                                last_date = await self._get_last_date(db, symbol, timeframe)

                                if last_date:
                                    logger.info(f"  Última data: {last_date}")
                                    # Buscar apenas dados novos (incrementais)
                                    limit = self._calculate_limit(timeframe, last_date)
                                else:
                                    logger.info("  Primeira atualização (sem dados)")
                                    # Buscar últimos 30 dias
                                    limit = self._calculate_limit(timeframe, days=30)

                                # Buscar dados da CoinAPI
                                df = await client.get_ohlcv_data(
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    limit=limit,
                                    force_refresh=True,
                                )

                                if df.empty:
                                    logger.warning("  ⚠️  Nenhum dado novo")
                                    break

                                # Filtrar apenas dados novos
                                if last_date:
                                    df = df[df["timestamp"] > last_date]

                                if df.empty:
                                    logger.info("  ✅ Base já está atualizada")
                                    break

                                # Salvar na base local
                                rows = await db.save_ohlcv_data(symbol, timeframe, df)
                                total_candles += rows

                                logger.success(f"  ✅ {rows} novos candles salvos")

                                # Backup (se habilitado)
                                if backup_mgr:
                                    backup_mgr.backup_ohlcv_data(
                                        symbol, timeframe, df, backup_type="daily"
                                    )

                                # Sucesso, sair do retry loop
                                break

                            except Exception as e:
                                retries += 1
                                logger.error(
                                    f"  ❌ Erro (tentativa {retries}/{self.max_retries}): {e}"
                                )

                                if retries < self.max_retries:
                                    logger.info("  ⏳ Aguardando 30s antes de tentar novamente...")
                                    await asyncio.sleep(30)
                                else:
                                    logger.error(f"  ❌ Falha após {self.max_retries} tentativas")

                        # Aguardar 2s entre requests (rate limit)
                        await asyncio.sleep(2)

                # Fechar conexões
                await db.close()

            # Estatísticas
            elapsed = (datetime.now() - start_time).total_seconds()
            self.stats["last_update"] = datetime.now()
            self.stats["total_candles_updated"] += total_candles

            logger.info("╔══════════════════════════════════════════════════════╗")
            logger.info("║                                                      ║")
            logger.info("║     ✅ Atualização Concluída                        ║")
            logger.info("║                                                      ║")
            logger.info("╚══════════════════════════════════════════════════════╝")
            logger.info("📊 Estatísticas:")
            logger.info(f"  • Total de candles: {total_candles}")
            logger.info(f"  • Tempo: {elapsed:.2f}s")
            logger.info(f"  • Símbolos: {len(self.symbols)}")
            logger.info(f"  • Timeframes: {len(self.timeframes)}")

            return True

        except Exception as e:
            logger.error(f"❌ Erro crítico na atualização: {e}")
            return False

    async def _get_last_date(
        self, db: AsyncDatabaseManager, symbol: str, timeframe: str
    ) -> datetime | None:
        """Busca a última data disponível na base local"""
        try:
            df = await db.get_ohlcv_data(symbol, timeframe, limit=1)

            if not df.empty:
                return df["timestamp"].iloc[0]

            return None

        except Exception as e:
            logger.warning(f"Erro ao buscar última data: {e}")
            return None

    def _calculate_limit(
        self, timeframe: str, last_date: datetime | None = None, days: int | None = None
    ) -> int:
        """
        Calcula quantos candles buscar baseado no timeframe.

        Args:
            timeframe: Timeframe (1min, 1h, etc)
            last_date: Última data na base (para incremental)
            days: Dias de histórico (para primeira vez)

        Returns:
            Número de candles a buscar
        """
        # Candles por dia para cada timeframe
        candles_per_day = {"1min": 1440, "5min": 288, "15min": 96, "1h": 24, "4h": 6, "1d": 1}

        cpd = candles_per_day.get(timeframe, 24)

        if last_date:
            # Incremental: buscar desde última data até agora
            days_diff = (datetime.now() - last_date).days + 1
            return cpd * days_diff
        elif days:
            # Primeira vez: buscar N dias
            return cpd * days
        else:
            # Default: 30 dias
            return cpd * 30

    def _calculate_next_run(self) -> datetime:
        """
        Calcula próxima execução baseado no intervalo e hora inicial.

        Returns:
            Datetime da próxima execução
        """
        now = datetime.now()

        # Parse hora inicial (formato HH:MM)
        hour, minute = map(int, self.start_time.split(":"))

        # Criar datetime para hoje na hora inicial
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        # Se já passou, adicionar intervalo
        while next_run <= now:
            next_run += timedelta(hours=self.update_interval_hours)

        return next_run

    def get_stats(self) -> dict[str, Any]:
        """Retorna estatísticas do atualizador"""
        return {
            **self.stats,
            "symbols": self.symbols,
            "timeframes": self.timeframes,
            "update_interval_hours": self.update_interval_hours,
            "start_time": self.start_time,
        }


# ==================== STANDALONE RUNNER ====================


async def run_updater(
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    interval_hours: int = 12,
    start_time: str = "03:00",
):
    """
    Executa o atualizador como processo standalone.

    Args:
        symbols: Lista de símbolos
        timeframes: Lista de timeframes
        interval_hours: Intervalo em horas
        start_time: Hora inicial (HH:MM)
    """
    updater = DataUpdater(
        symbols=symbols,
        timeframes=timeframes,
        update_interval_hours=interval_hours,
        start_time=start_time,
    )

    await updater.start()


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CeciAI Data Updater")

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USD", "ETH/USD"],
        help="Símbolos a atualizar (default: BTC/USD ETH/USD)",
    )

    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1min", "5min", "1h", "4h", "1d"],
        help="Timeframes a atualizar (default: 1min 5min 1h 4h 1d)",
    )

    parser.add_argument(
        "--interval", type=int, default=12, help="Intervalo de atualização em horas (default: 12)"
    )

    parser.add_argument(
        "--start-time",
        type=str,
        default="03:00",
        help="Hora inicial no formato HH:MM (default: 03:00)",
    )

    parser.add_argument("--run-once", action="store_true", help="Executar apenas uma vez e sair")

    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║                                                      ║")
    print("║     CeciAI - Data Updater                           ║")
    print("║                                                      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print(f"Símbolos: {args.symbols}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Intervalo: {args.interval}h")
    print(f"Hora inicial: {args.start_time}")
    print()

    if args.run_once:
        # Executar apenas uma vez
        updater = DataUpdater(
            symbols=args.symbols,
            timeframes=args.timeframes,
            update_interval_hours=args.interval,
            start_time=args.start_time,
        )

        asyncio.run(updater.update_data())
    else:
        # Executar em modo daemon
        asyncio.run(
            run_updater(
                symbols=args.symbols,
                timeframes=args.timeframes,
                interval_hours=args.interval,
                start_time=args.start_time,
            )
        )
