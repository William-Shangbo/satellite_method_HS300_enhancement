from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeSettings:
    project_root: Path
    raw_data_root: Path
    artifact_root: Path
    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mysql_database: str
    tushare_token: str
    benchmark_index: str


def load_settings(project_root: str | Path | None = None) -> RuntimeSettings:
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[3]
    return RuntimeSettings(
        project_root=root,
        raw_data_root=root / "data" / "raw",
        artifact_root=root / "artifacts",
        mysql_host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        mysql_port=int(os.getenv("MYSQL_PORT", "3306")),
        mysql_user=os.getenv("MYSQL_USER", "root"),
        mysql_password=os.getenv("MYSQL_PASSWORD", "root"),
        mysql_database=os.getenv("MYSQL_DATABASE", "index_enforcement"),
        tushare_token=os.getenv("TUSHARE_TOKEN", ""),
        benchmark_index=os.getenv("BENCHMARK_INDEX", "399300.SZ"),
    )
