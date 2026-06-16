from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetContract:
    dataset: str
    api_name: str
    required: bool
    layer: str
    target_table: str
    refresh: str
    purpose: str
    primary_key: tuple[str, ...]
    partition_key: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    fields: tuple[str, ...] = ()
    fields_hint: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineContracts:
    version: int
    pipeline: str
    provider: str
    endpoint: str
    benchmark_index: str
    datasets: tuple[DatasetContract, ...]


def _coerce_contract(item: dict[str, Any]) -> DatasetContract:
    return DatasetContract(
        dataset=item["dataset"],
        api_name=item["api_name"],
        required=bool(item["required"]),
        layer=item["layer"],
        target_table=item["target_table"],
        refresh=item["refresh"],
        purpose=item["purpose"],
        primary_key=tuple(item.get("primary_key", [])),
        partition_key=item.get("partition_key"),
        params=dict(item.get("params", {})),
        fields=tuple(item.get("fields", [])),
        fields_hint=tuple(item.get("fields_hint", [])),
    )


def load_dataset_contracts(path: str | Path) -> PipelineContracts:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    datasets = tuple(_coerce_contract(item) for item in payload["datasets"])
    return PipelineContracts(
        version=int(payload["version"]),
        pipeline=payload["pipeline"],
        provider=payload["provider"],
        endpoint=payload["endpoint"],
        benchmark_index=payload["benchmark_index"],
        datasets=datasets,
    )
