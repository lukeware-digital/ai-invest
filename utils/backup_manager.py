"""
CeciAI - Backup Manager
Sistema de backup robusto para garantir que dados NUNCA sejam perdidos

Estratégia de Backup (3-2-1):
- 3 cópias dos dados
- 2 tipos diferentes de mídia
- 1 cópia off-site (opcional)

Camadas:
1. DuckDB (primário)
2. Parquet (backup local comprimido)
3. JSON (backup legível)
4. Checksum (validação de integridade)

Autor: CeciAI Team
Data: 2025-10-07
"""

import gzip
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from utils.database import DatabaseManager


class BackupManager:
    """
    Gerenciador de backups com múltiplas camadas de proteção.

    Garante que dados do CoinAPI nunca sejam perdidos.
    """

    def __init__(
        self,
        backup_dir: str = "data/backups",
        enable_compression: bool = True,
        enable_checksums: bool = True,
    ):
        """
        Inicializa o gerenciador de backups.

        Args:
            backup_dir: Diretório para backups
            enable_compression: Se True, comprime backups
            enable_checksums: Se True, gera checksums para validação
        """
        self.backup_dir = Path(backup_dir)
        self.enable_compression = enable_compression
        self.enable_checksums = enable_checksums

        # Criar estrutura de diretórios
        self._create_backup_structure()

        logger.info(f"BackupManager inicializado: {backup_dir}")

    def _create_backup_structure(self):
        """Cria estrutura de diretórios para backups"""
        dirs = [
            self.backup_dir / "daily",  # Backups diários
            self.backup_dir / "weekly",  # Backups semanais
            self.backup_dir / "monthly",  # Backups mensais
            self.backup_dir / "parquet",  # Backups em Parquet
            self.backup_dir / "json",  # Backups em JSON
            self.backup_dir / "checksums",  # Checksums para validação
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    # ==================== BACKUP AUTOMÁTICO ====================

    def backup_ohlcv_data(
        self, symbol: str, timeframe: str, df: pd.DataFrame, backup_type: str = "daily"
    ) -> dict[str, str]:
        """
        Faz backup de dados OHLCV em múltiplas camadas.

        Args:
            symbol: Par de trading
            timeframe: Timeframe
            df: DataFrame com dados
            backup_type: Tipo de backup (daily, weekly, monthly)

        Returns:
            Dict com caminhos dos backups criados
        """
        if df.empty:
            logger.warning(f"DataFrame vazio, pulando backup de {symbol} {timeframe}")
            return {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_symbol = symbol.replace("/", "_")
        base_name = f"{safe_symbol}_{timeframe}_{timestamp}"

        backups = {}

        # 1. Backup em Parquet (comprimido, eficiente)
        parquet_path = self.backup_dir / "parquet" / f"{base_name}.parquet"
        df.to_parquet(parquet_path, compression="snappy", index=False)
        backups["parquet"] = str(parquet_path)
        logger.info(f"✅ Backup Parquet: {parquet_path}")

        # 2. Backup em JSON (legível, portável)
        json_path = self.backup_dir / "json" / f"{base_name}.json"
        if self.enable_compression:
            json_path = json_path.with_suffix(".json.gz")
            with gzip.open(json_path, "wt", encoding="utf-8") as f:
                df.to_json(f, orient="records", date_format="iso", indent=2)
        else:
            df.to_json(json_path, orient="records", date_format="iso", indent=2)
        backups["json"] = str(json_path)
        logger.info(f"✅ Backup JSON: {json_path}")

        # 3. Backup periódico (daily/weekly/monthly)
        periodic_path = self.backup_dir / backup_type / f"{base_name}.parquet"
        shutil.copy2(parquet_path, periodic_path)
        backups["periodic"] = str(periodic_path)
        logger.info(f"✅ Backup {backup_type}: {periodic_path}")

        # 4. Checksum para validação de integridade
        if self.enable_checksums:
            checksum = self._generate_checksum(df)
            checksum_path = self.backup_dir / "checksums" / f"{base_name}.checksum"

            with open(checksum_path, "w") as f:
                json.dump(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "timestamp": timestamp,
                        "rows": len(df),
                        "checksum": checksum,
                        "files": backups,
                    },
                    f,
                    indent=2,
                )

            backups["checksum"] = str(checksum_path)
            logger.info(f"✅ Checksum: {checksum_path}")

        # 5. Metadados
        self._save_metadata(symbol, timeframe, df, backups)

        return backups

    def backup_database(self, db_path: str = "data/ceciai.duckdb") -> str:
        """
        Faz backup completo do banco DuckDB.

        Args:
            db_path: Caminho do banco DuckDB

        Returns:
            Caminho do backup
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Banco não encontrado: {db_path}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / "daily" / f"ceciai_{timestamp}.duckdb"

        # Copiar banco
        shutil.copy2(db_path, backup_path)

        # Comprimir se habilitado
        if self.enable_compression:
            compressed_path = backup_path.with_suffix(".duckdb.gz")
            with open(backup_path, "rb") as f_in, gzip.open(compressed_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

            # Remover original não comprimido
            backup_path.unlink()
            backup_path = compressed_path

        size_mb = os.path.getsize(backup_path) / (1024 * 1024)
        logger.success(f"✅ Backup do banco: {backup_path} ({size_mb:.2f} MB)")

        return str(backup_path)

    # ==================== RECUPERAÇÃO ====================

    def restore_from_backup(
        self, backup_path: str, target_path: str | None = None
    ) -> pd.DataFrame:
        """
        Restaura dados de um backup.

        Args:
            backup_path: Caminho do backup
            target_path: Caminho de destino (opcional)

        Returns:
            DataFrame com dados restaurados
        """
        backup_path = Path(backup_path)

        if not backup_path.exists():
            raise FileNotFoundError(f"Backup não encontrado: {backup_path}")

        logger.info(f"📥 Restaurando de: {backup_path}")

        # Detectar formato
        if backup_path.suffix == ".parquet":
            df = pd.read_parquet(backup_path)

        elif backup_path.suffix == ".gz":
            if ".json.gz" in backup_path.name:
                with gzip.open(backup_path, "rt", encoding="utf-8") as f:
                    df = pd.read_json(f)
            elif ".duckdb.gz" in backup_path.name:
                # Descomprimir banco
                if target_path:
                    with gzip.open(backup_path, "rb") as f_in, open(target_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    logger.success(f"✅ Banco restaurado: {target_path}")
                    return pd.DataFrame()  # Retornar vazio para banco
                else:
                    raise ValueError("target_path necessário para restaurar banco")

        elif backup_path.suffix == ".json":
            df = pd.read_json(backup_path)

        else:
            raise ValueError(f"Formato não suportado: {backup_path.suffix}")

        logger.success(f"✅ Restaurado: {len(df)} linhas")

        # Validar checksum se disponível
        if self.enable_checksums:
            self._validate_checksum(backup_path, df)

        return df

    def find_latest_backup(
        self, symbol: str, timeframe: str, backup_type: str = "daily"
    ) -> str | None:
        """
        Encontra o backup mais recente de um símbolo.

        Args:
            symbol: Par de trading
            timeframe: Timeframe
            backup_type: Tipo de backup

        Returns:
            Caminho do backup mais recente ou None
        """
        safe_symbol = symbol.replace("/", "_")
        pattern = f"{safe_symbol}_{timeframe}_*.parquet"

        backup_dir = self.backup_dir / backup_type
        backups = list(backup_dir.glob(pattern))

        if not backups:
            return None

        # Ordenar por data (mais recente primeiro)
        backups.sort(reverse=True)

        return str(backups[0])

    # ==================== VALIDAÇÃO ====================

    def _generate_checksum(self, df: pd.DataFrame) -> str:
        """Gera checksum MD5 dos dados"""
        data_str = df.to_json(orient="records", date_format="iso")
        return hashlib.md5(data_str.encode()).hexdigest()

    def _validate_checksum(self, backup_path: Path, df: pd.DataFrame) -> bool:
        """Valida integridade dos dados usando checksum"""
        checksum_path = self.backup_dir / "checksums" / f"{backup_path.stem}.checksum"

        if not checksum_path.exists():
            logger.warning(f"⚠️  Checksum não encontrado: {checksum_path}")
            return False

        with open(checksum_path) as f:
            metadata = json.load(f)

        expected_checksum = metadata["checksum"]
        actual_checksum = self._generate_checksum(df)

        if expected_checksum == actual_checksum:
            logger.success("✅ Checksum válido")
            return True
        else:
            logger.error("❌ Checksum inválido! Dados podem estar corrompidos")
            return False

    def _save_metadata(
        self, symbol: str, timeframe: str, df: pd.DataFrame, backups: dict[str, str]
    ):
        """Salva metadados do backup"""
        metadata = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "rows": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": df["timestamp"].min().isoformat() if "timestamp" in df else None,
                "end": df["timestamp"].max().isoformat() if "timestamp" in df else None,
            },
            "backups": backups,
        }

        metadata_path = (
            self.backup_dir / "metadata" / f"{symbol.replace('/', '_')}_{timeframe}.json"
        )
        metadata_path.parent.mkdir(exist_ok=True)

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    # ==================== LIMPEZA ====================

    def cleanup_old_backups(
        self, keep_daily: int = 7, keep_weekly: int = 4, keep_monthly: int = 12
    ):
        """
        Remove backups antigos para economizar espaço.

        Args:
            keep_daily: Manter últimos N backups diários
            keep_weekly: Manter últimos N backups semanais
            keep_monthly: Manter últimos N backups mensais
        """
        logger.info("🧹 Limpando backups antigos...")

        def cleanup_dir(dir_path: Path, keep: int):
            backups = sorted(dir_path.glob("*"), reverse=True)

            if len(backups) > keep:
                for backup in backups[keep:]:
                    backup.unlink()
                    logger.info(f"🗑️  Removido: {backup.name}")

        cleanup_dir(self.backup_dir / "daily", keep_daily)
        cleanup_dir(self.backup_dir / "weekly", keep_weekly)
        cleanup_dir(self.backup_dir / "monthly", keep_monthly)

        logger.success("✅ Limpeza concluída")

    def get_backup_statistics(self) -> dict[str, object]:
        """Retorna estatísticas dos backups"""
        stats = {"total_backups": 0, "total_size_mb": 0, "by_type": {}}

        for backup_type in ["daily", "weekly", "monthly", "parquet", "json"]:
            dir_path = self.backup_dir / backup_type

            if dir_path.exists():
                files = list(dir_path.glob("*"))
                count = len(files)
                size = sum(f.stat().st_size for f in files if f.is_file())

                stats["by_type"][backup_type] = {"count": count, "size_mb": size / (1024 * 1024)}

                stats["total_backups"] += count
                stats["total_size_mb"] += size / (1024 * 1024)

        return stats


# ==================== INTEGRAÇÃO COM DATABASE ====================


class DatabaseWithBackup(DatabaseManager):
    """
    DatabaseManager com backup automático.

    Todos os dados salvos são automaticamente backupeados.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backup_manager = BackupManager()

    def save_ohlcv_data(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        """Salva dados com backup automático"""
        # Salvar no banco
        rows = super().save_ohlcv_data(symbol, timeframe, df)

        # Backup automático
        try:
            self.backup_manager.backup_ohlcv_data(symbol, timeframe, df)
            logger.info(f"💾 Backup automático criado para {symbol} {timeframe}")
        except Exception as e:
            logger.error(f"❌ Erro ao criar backup: {e}")

        return rows


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    print("CeciAI - Backup Manager")
    print("=" * 50)

    # Criar backup manager
    backup_mgr = BackupManager()

    # Estatísticas
    stats = backup_mgr.get_backup_statistics()
    print("\n📊 Estatísticas de Backup:")
    print(f"  Total de backups: {stats['total_backups']}")
    print(f"  Tamanho total: {stats['total_size_mb']:.2f} MB")

    for backup_type, data in stats["by_type"].items():
        print(f"  {backup_type}: {data['count']} backups ({data['size_mb']:.2f} MB)")

    print("\n✅ Backup Manager funcionando!")
